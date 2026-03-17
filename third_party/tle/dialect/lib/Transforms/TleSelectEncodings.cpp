// MIT License

// Copyright (c) 2025 The FlagOS Contributors

// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

// flagtree tle

#include "tle/dialect/include/IR/Dialect.h"
#include "tle/dialect/include/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/ADT/DenseSet.h"

namespace mlir::triton::tle {

#define GEN_PASS_DEF_TRITONTLESELECTENCODINGS
#include "tle/dialect/include/Transforms/Passes.h.inc"

namespace {

// Triton shared-memory pointers use LLVM address space 3 (NVVM shared).
constexpr int kSharedMemoryAddressSpace = 3;
constexpr StringLiteral kBarrierGroupAttr = "tle.barrier_group";
constexpr StringLiteral kTTContiguityAttr = "tt.contiguity";
constexpr StringLiteral kTTDivisibilityAttr = "tt.divisibility";
constexpr StringLiteral kTTConstancyAttr = "tt.constancy";

static Value stripConvertLayouts(Value value) {
  Value current = value;
  while (auto convert = current.getDefiningOp<triton::gpu::ConvertLayoutOp>())
    current = convert.getSrc();
  return current;
}

static Value stripIndexValueWrappers(Value value) {
  Value current = value;
  while (true) {
    if (auto convert = current.getDefiningOp<triton::gpu::ConvertLayoutOp>()) {
      current = convert.getSrc();
      continue;
    }
    if (auto ext = current.getDefiningOp<arith::ExtSIOp>()) {
      current = ext.getIn();
      continue;
    }
    if (auto ext = current.getDefiningOp<arith::ExtUIOp>()) {
      current = ext.getIn();
      continue;
    }
    if (auto trunc = current.getDefiningOp<arith::TruncIOp>()) {
      current = trunc.getIn();
      continue;
    }
    if (auto cast = current.getDefiningOp<arith::IndexCastOp>()) {
      current = cast.getIn();
      continue;
    }
    break;
  }
  return current;
}

static bool matchZeroStartMakeRange(Value value, int64_t extent) {
  Value current = stripIndexValueWrappers(value);
  auto range = current.getDefiningOp<triton::MakeRangeOp>();
  if (!range)
    return false;
  return range.getStart() == 0 && range.getEnd() == extent;
}

static bool matchFullIndexTensorForAxis(Value index, size_t axis,
                                        ArrayRef<int64_t> shape) {
  auto indexTy = dyn_cast<RankedTensorType>(index.getType());
  if (!indexTy || !indexTy.getElementType().isInteger())
    return false;
  if (indexTy.getShape() != shape)
    return false;

  Value current = stripIndexValueWrappers(index);
  if (shape.size() == 1)
    return matchZeroStartMakeRange(current, shape.front());

  auto bcast = current.getDefiningOp<triton::BroadcastOp>();
  if (!bcast)
    return false;

  auto bcastSrcTy = dyn_cast<RankedTensorType>(bcast.getSrc().getType());
  if (!bcastSrcTy || bcastSrcTy.getRank() != static_cast<int64_t>(shape.size()))
    return false;
  for (auto [dim, dimSize] : llvm::enumerate(shape)) {
    const int64_t expected = dim == axis ? dimSize : 1;
    if (bcastSrcTy.getShape()[dim] != expected)
      return false;
  }

  current = stripIndexValueWrappers(bcast.getSrc());
  while (auto expand = current.getDefiningOp<triton::ExpandDimsOp>())
    current = stripIndexValueWrappers(expand.getSrc());

  auto rangeTy = dyn_cast<RankedTensorType>(current.getType());
  if (!rangeTy || rangeTy.getRank() != 1)
    return false;
  if (rangeTy.getShape()[0] != shape[axis])
    return false;

  return matchZeroStartMakeRange(current, shape[axis]);
}

// Loads of full-view local_pointers are later rewritten to ttg.local_load.
// They should not bias local_pointers encoding inference toward load layouts.
static bool isRewritableFullViewLocalPointerLoad(triton::LoadOp load) {
  if (load.getMask() || load.getOther())
    return false;
  if (load.getIsVolatile())
    return false;
  if (load.getCache() != triton::CacheModifier::NONE ||
      load.getEvict() != triton::EvictionPolicy::NORMAL)
    return false;

  auto loadTy = dyn_cast<RankedTensorType>(load.getType());
  if (!loadTy)
    return false;

  Value ptr = stripConvertLayouts(load.getPtr());
  auto localPointers = ptr.getDefiningOp<triton::tle::LocalPointersOp>();
  if (!localPointers)
    return false;

  auto ptrTy = dyn_cast<RankedTensorType>(localPointers.getResult().getType());
  if (!ptrTy)
    return false;

  auto memDescTy = dyn_cast<triton::gpu::MemDescType>(localPointers.getSrc().getType());
  if (!memDescTy)
    return false;

  auto memDescShape = memDescTy.getShape();
  if (loadTy.getShape() != memDescShape || ptrTy.getShape() != memDescShape)
    return false;
  if (loadTy.getElementType() != memDescTy.getElementType())
    return false;

  auto indices = localPointers.getIndices();
  if (indices.empty())
    return true;
  if (indices.size() != memDescShape.size())
    return false;

  for (auto [axis, index] : llvm::enumerate(indices))
    if (!matchFullIndexTensorForAxis(index, axis, memDescShape))
      return false;
  return true;
}

static int64_t getScfLoopDepth(Operation *op) {
  int64_t depth = 0;
  for (Operation *cur = op; cur; cur = cur->getParentOp())
    if (isa<scf::ForOp>(cur))
      ++depth;
  return depth;
}

static bool valueFeedsDot(Value root) {
  llvm::SmallVector<Value> worklist;
  llvm::DenseSet<Value> visited;
  auto enqueue = [&](Value v) {
    if (v && visited.insert(v).second)
      worklist.push_back(v);
  };
  enqueue(root);
  while (!worklist.empty()) {
    Value current = worklist.pop_back_val();
    for (OpOperand &use : current.getUses()) {
      Operation *owner = use.getOwner();
      if (isa<triton::DotOpInterface>(owner))
        return true;
      if (auto convert = dyn_cast<triton::gpu::ConvertLayoutOp>(owner)) {
        enqueue(convert.getResult());
        continue;
      }
      if (auto trans = dyn_cast<triton::TransOp>(owner)) {
        enqueue(trans.getResult());
        continue;
      }
      if (auto bcast = dyn_cast<triton::BroadcastOp>(owner)) {
        enqueue(bcast.getResult());
        continue;
      }
      if (auto expand = dyn_cast<triton::ExpandDimsOp>(owner)) {
        enqueue(expand.getResult());
        continue;
      }
      if (auto reshape = dyn_cast<triton::ReshapeOp>(owner)) {
        enqueue(reshape.getResult());
        continue;
      }
    }
  }
  return false;
}

struct EncodingVote {
  Attribute encoding;
  int64_t score;
};

static Operation *peelAxisInfoCarrier(Value value) {
  llvm::DenseSet<Value> visited;
  Value current = value;
  while (current && visited.insert(current).second) {
    Operation *def = current.getDefiningOp();
    if (!def)
      break;
    if (auto convert = dyn_cast<triton::gpu::ConvertLayoutOp>(def)) {
      current = convert.getSrc();
      continue;
    }
    if (auto bcast = dyn_cast<triton::BroadcastOp>(def)) {
      current = bcast.getSrc();
      continue;
    }
    if (auto expand = dyn_cast<triton::ExpandDimsOp>(def)) {
      current = expand.getSrc();
      continue;
    }
    if (auto reshape = dyn_cast<triton::ReshapeOp>(def)) {
      current = reshape.getSrc();
      continue;
    }
    return def;
  }
  return current ? current.getDefiningOp() : nullptr;
}

static void copyAxisInfoAttrs(Operation *src, Operation *dst) {
  if (!src || !dst)
    return;
  auto tryCopy = [&](StringRef name) {
    if (dst->getDiscardableAttr(name))
      return;
    if (auto attr = src->getDiscardableAttr(name))
      dst->setDiscardableAttr(name, attr);
  };
  tryCopy(kTTContiguityAttr);
  tryCopy(kTTDivisibilityAttr);
  tryCopy(kTTConstancyAttr);
}

static void
collectConsumerEncodingVotes(Value root,
                             llvm::SmallVectorImpl<EncodingVote> &votes) {
  llvm::SmallVector<Value> worklist;
  llvm::DenseSet<Value> visited;
  auto enqueue = [&](Value v) {
    if (!v)
      return;
    if (!visited.insert(v).second)
      return;
    worklist.push_back(v);
  };

  enqueue(root);
  while (!worklist.empty()) {
    Value current = worklist.pop_back_val();
    for (OpOperand &use : current.getUses()) {
      Operation *owner = use.getOwner();
      if (auto load = dyn_cast<triton::LoadOp>(owner)) {
        if (isRewritableFullViewLocalPointerLoad(load))
          continue;
        auto loadTy = dyn_cast<RankedTensorType>(load.getResult().getType());
        if (loadTy && loadTy.getEncoding()) {
          const int64_t depthFactor = 1 + getScfLoopDepth(owner);
          int64_t score = 8 * depthFactor;
          if (valueFeedsDot(load.getResult()))
            score += 128 * depthFactor;
          votes.push_back({loadTy.getEncoding(), score});
        }
        continue;
      }
      if (auto store = dyn_cast<triton::StoreOp>(owner)) {
        auto valueTy = dyn_cast<RankedTensorType>(store.getValue().getType());
        if (valueTy && valueTy.getEncoding()) {
          const int64_t depthFactor = 1 + getScfLoopDepth(owner);
          int64_t score = 2 * depthFactor;
          if (Operation *def = store.getValue().getDefiningOp();
              def && isa<triton::DotOpInterface>(def))
            score += 8 * depthFactor;
          votes.push_back({valueTy.getEncoding(), score});
        }
        continue;
      }
      if (auto convert = dyn_cast<triton::gpu::ConvertLayoutOp>(owner)) {
        enqueue(convert.getResult());
        continue;
      }
      if (auto bcast = dyn_cast<triton::BroadcastOp>(owner)) {
        enqueue(bcast.getResult());
        continue;
      }
      if (auto expand = dyn_cast<triton::ExpandDimsOp>(owner)) {
        enqueue(expand.getResult());
        continue;
      }
      if (auto reshape = dyn_cast<triton::ReshapeOp>(owner)) {
        enqueue(reshape.getResult());
        continue;
      }
      if (auto remote = dyn_cast<triton::tle::RemotePointersOp>(owner)) {
        enqueue(remote.getResult());
        continue;
      }
    }
  }
}

static Attribute pickDominantEncoding(ArrayRef<EncodingVote> votes,
                                      Attribute fallback) {
  if (votes.empty())
    return fallback;

  llvm::DenseMap<Attribute, int64_t> scoreByEncoding;
  llvm::SmallVector<Attribute> order;
  for (const EncodingVote &vote : votes) {
    if (!vote.encoding)
      continue;
    auto [it, inserted] = scoreByEncoding.try_emplace(vote.encoding, 0);
    if (inserted)
      order.push_back(vote.encoding);
    it->second += vote.score;
  }
  if (order.empty())
    return fallback;

  Attribute best = order.front();
  int64_t bestScore = scoreByEncoding.lookup(best);
  for (Attribute encoding : order) {
    int64_t score = scoreByEncoding.lookup(encoding);
    if (score > bestScore) {
      best = encoding;
      bestScore = score;
      continue;
    }
    if (score == bestScore && encoding == fallback)
      best = encoding;
  }
  return best;
}

class SelectEncodingsPass
    : public impl::TritonTleSelectEncodingsBase<SelectEncodingsPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    OpBuilder builder(module.getContext());
    module.walk([&](triton::tle::LocalPointersOp op) {
      // Always tag local pointer ops so barrier insertion can track hazards
      // across different pointer views of the same alloc.
      tagDependencyGroup(op, builder);

      auto tensorTy = dyn_cast<RankedTensorType>(op.getResult().getType());
      auto scalarPtrTy =
          dyn_cast<triton::PointerType>(op.getResult().getType());
      if (!tensorTy && !scalarPtrTy)
        return;
      auto ptrTy =
          tensorTy ? dyn_cast<triton::PointerType>(tensorTy.getElementType())
                   : scalarPtrTy;
      if (!ptrTy)
        return;
      bool updated = false;
      Type updatedResultTy = op.getResult().getType();
      const auto desiredAddrSpace = kSharedMemoryAddressSpace;
      if (ptrTy.getAddressSpace() != desiredAddrSpace) {
        ptrTy =
            triton::PointerType::get(ptrTy.getPointeeType(), desiredAddrSpace);
        updated = true;
      }

      if (!tensorTy) {
        if (updated)
          op.getResult().setType(ptrTy);
        return;
      }

      auto encoding = tensorTy.getEncoding();
      SmallVector<EncodingVote> votes;
      collectConsumerEncodingVotes(op.getResult(), votes);
      Attribute userEncoding = pickDominantEncoding(votes, encoding);
      if (userEncoding && userEncoding != encoding) {
        encoding = userEncoding;
        updated = true;
      }
      if (!encoding) {
        OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPoint(op);
        int numWarps = triton::gpu::maybeLookupNumWarps(op).value_or(1);
        int threadsPerWarp = triton::gpu::lookupThreadsPerWarp(builder);
        int numCTAs = triton::gpu::lookupNumCTAs(builder);
        encoding = triton::gpu::getDefaultBlockedEncoding(
            module.getContext(), tensorTy.getShape(), numWarps, threadsPerWarp,
            numCTAs);
        updated = true;
      }

      if (updated)
        updatedResultTy =
            RankedTensorType::get(tensorTy.getShape(), ptrTy, encoding);

      if (updated)
        op.getResult().setType(updatedResultTy);

      if (updated) {
        llvm::DenseSet<Value> visited;
        auto updateUserResultTypes = [&](auto &&self, Value ptrVal) -> void {
          if (!ptrVal || !visited.insert(ptrVal).second)
            return;
          auto ptrTensorTy = cast<RankedTensorType>(ptrVal.getType());
          auto ptrEncoding = ptrTensorTy.getEncoding();
          auto ptrElemTy =
              cast<triton::PointerType>(ptrTensorTy.getElementType())
                  .getPointeeType();
          auto loadTy = RankedTensorType::get(ptrTensorTy.getShape(), ptrElemTy,
                                              ptrTensorTy.getEncoding());
          auto convertOperandEncoding = [&](Operation *insertBefore, Value v,
                                            Attribute encoding) -> Value {
            auto vTy = dyn_cast<RankedTensorType>(v.getType());
            if (!vTy)
              return v;
            if (vTy.getEncoding() == encoding)
              return v;
            auto convertedTy = RankedTensorType::get(
                vTy.getShape(), vTy.getElementType(), encoding);
            OpBuilder::InsertionGuard guard(builder);
            builder.setInsertionPoint(insertBefore);
            auto converted = builder.create<triton::gpu::ConvertLayoutOp>(
                insertBefore->getLoc(), convertedTy, v);
            return converted.getResult();
          };
          for (OpOperand &use : ptrVal.getUses()) {
            Operation *owner = use.getOwner();
            if (auto load = dyn_cast<triton::LoadOp>(owner)) {
              if (isRewritableFullViewLocalPointerLoad(load))
                continue;
              if (Value mask = load.getMask()) {
                Value convertedMask =
                    convertOperandEncoding(owner, mask, ptrEncoding);
                if (convertedMask != mask)
                  load.getMaskMutable().assign(convertedMask);
              }
              if (Value other = load.getOther()) {
                Value convertedOther =
                    convertOperandEncoding(owner, other, ptrEncoding);
                if (convertedOther != other)
                  load.getOtherMutable().assign(convertedOther);
              }
              auto oldLoadTy =
                  dyn_cast<RankedTensorType>(load.getResult().getType());
              if (oldLoadTy != loadTy) {
                load.getResult().setType(loadTy);
                if (oldLoadTy) {
                  OpBuilder::InsertionGuard guard(builder);
                  builder.setInsertionPointAfter(load);
                  auto bridge = builder.create<triton::gpu::ConvertLayoutOp>(
                      load.getLoc(), oldLoadTy, load.getResult());
                  load.getResult().replaceAllUsesExcept(bridge.getResult(),
                                                        bridge.getOperation());
                }
              }
              continue;
            }
            if (auto store = dyn_cast<triton::StoreOp>(owner)) {
              auto valueTy =
                  dyn_cast<RankedTensorType>(store.getValue().getType());
              if (valueTy) {
                Value convertedValue = convertOperandEncoding(
                    owner, store.getValue(), ptrEncoding);
                if (convertedValue != store.getValue())
                  store.getValueMutable().assign(convertedValue);
              }
              if (Value mask = store.getMask()) {
                Value convertedMask =
                    convertOperandEncoding(owner, mask, ptrEncoding);
                if (convertedMask != mask)
                  store.getMaskMutable().assign(convertedMask);
              }
              continue;
            }
            if (auto atomic = dyn_cast<triton::AtomicRMWOp>(owner)) {
              Value val = atomic.getVal();
              Value convertedVal =
                  convertOperandEncoding(owner, val, ptrEncoding);
              if (convertedVal != val)
                atomic.getValMutable().assign(convertedVal);
              if (Value mask = atomic.getMask()) {
                Value convertedMask =
                    convertOperandEncoding(owner, mask, ptrEncoding);
                if (convertedMask != mask)
                  atomic.getMaskMutable().assign(convertedMask);
              }
              atomic.getResult().setType(loadTy);
              continue;
            }
            if (auto cas = dyn_cast<triton::AtomicCASOp>(owner)) {
              Value cmp = cas.getCmp();
              Value convertedCmp =
                  convertOperandEncoding(owner, cmp, ptrEncoding);
              if (convertedCmp != cmp)
                cas.getCmpMutable().assign(convertedCmp);
              Value val = cas.getVal();
              Value convertedVal =
                  convertOperandEncoding(owner, val, ptrEncoding);
              if (convertedVal != val)
                cas.getValMutable().assign(convertedVal);
              cas.getResult().setType(loadTy);
              continue;
            }
            if (auto remote = dyn_cast<triton::tle::RemotePointersOp>(owner)) {
              if (remote.getResult().getType() != ptrTensorTy)
                remote.getResult().setType(ptrTensorTy);
              self(self, remote.getResult());
              continue;
            }
          }
        };
        updateUserResultTypes(updateUserResultTypes, op.getResult());
      }

      auto desiredEncoding =
          cast<RankedTensorType>(updatedResultTy).getEncoding();
      if (desiredEncoding) {
        OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPoint(op);
        SmallVector<Value> newOperands;
        newOperands.reserve(op->getNumOperands());
        newOperands.push_back(op.getSrc());
        bool updatedOperands = false;
        for (Value operand : op.getIndices()) {
          auto operandTy = dyn_cast<RankedTensorType>(operand.getType());
          if (!operandTy) {
            newOperands.push_back(operand);
            continue;
          }
          if (operandTy.getEncoding() == desiredEncoding) {
            newOperands.push_back(operand);
            continue;
          }
          auto convertedTy = RankedTensorType::get(operandTy.getShape(),
                                                   operandTy.getElementType(),
                                                   desiredEncoding);
          auto converted = builder.create<triton::gpu::ConvertLayoutOp>(
              op.getLoc(), convertedTy, operand);
          newOperands.push_back(converted);
          updatedOperands = true;
        }
        if (updatedOperands)
          op->setOperands(newOperands);
      }
    });

    // remote_pointers should preserve source pointer axis properties so later
    // passes can reason about remote operands without dialect-specific
    // visitors.
    module.walk([&](triton::tle::RemotePointersOp op) {
      Operation *srcDef = peelAxisInfoCarrier(op.getSrc());
      copyAxisInfoAttrs(srcDef, op.getOperation());
    });
  }

  void tagDependencyGroup(triton::tle::LocalPointersOp op, OpBuilder &builder) {
    auto alloc = op.getSrc().getDefiningOp<triton::gpu::LocalAllocOp>();
    if (!alloc)
      return;
    auto groupAttr = alloc->getAttrOfType<IntegerAttr>(kBarrierGroupAttr);
    if (!groupAttr) {
      groupAttr = builder.getI64IntegerAttr(nextBarrierGroupId++);
      alloc->setAttr(kBarrierGroupAttr, groupAttr);
    }
    op->setAttr(kBarrierGroupAttr, groupAttr);
  }

  int64_t nextBarrierGroupId = 0;
};

} // namespace
} // namespace mlir::triton::tle
