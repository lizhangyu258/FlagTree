/*
 * Copyright 2026 FlagOS Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/Transforms/Passes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"

#include <limits>

namespace mlir::triton {

#define GEN_PASS_DEF_TRITONLOOPVECTORIZE
#include "triton/Dialect/Triton/Transforms/Passes.h.inc"

#define DEBUG_TYPE "triton-loop-vectorize"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace {

constexpr StringLiteral kVectorFactorAttr = "tt.loop_vectorize_factor";

static std::optional<int64_t> getConstantInt(Value value) {
  if (auto constant = value.getDefiningOp<arith::ConstantOp>())
    if (auto integer = dyn_cast<IntegerAttr>(constant.getValue()))
      return integer.getInt();
  return std::nullopt;
}

static RankedTensorType prependIterationDim(Type type, int64_t width) {
  if (auto tensor = dyn_cast<RankedTensorType>(type)) {
    SmallVector<int64_t> shape(tensor.getShape());
    shape.insert(shape.begin(), width);
    return RankedTensorType::get(shape, tensor.getElementType());
  }
  return RankedTensorType::get({width}, type);
}

enum class VectorizationKind {
  CloneAndBroadcast,
  Splat,
  ExpandDims,
  Reduce,
  Reshape,
  Identity,
  Generic
};

struct OperandVectorizationPlan {
  Value source;
  RankedTensorType type;
  bool broadcastScalar = false;
};

struct OperationVectorizationPlan {
  Operation *op;
  VectorizationKind kind = VectorizationKind::Generic;
  SmallVector<OperandVectorizationPlan> operands;
  SmallVector<Type> resultTypes;
  DictionaryAttr attributes;
  int32_t axis = 0;
};

struct LoopVectorizationPlan {
  scf::ForOp loop;
  int64_t width;
  int64_t step;
  bool singleGroup = false;
  SmallVector<OperationVectorizationPlan> operations;
};

class LoopVectorizer {
private:
  scf::ForOp loop;
  int64_t width;
  OpBuilder builder;
  IRMapping mapping;
  StringRef failureReason;
  Operation *failureOp = nullptr;

public:
  LoopVectorizer(scf::ForOp loop, int64_t width)
      : loop(loop), width(width), builder(loop) {}

  LogicalResult run() {
    auto plan = buildPlan();
    if (failed(plan)) {
      if (!failureReason.empty()) {
        auto diagnostic = loop.emitWarning("loop vectorization with factor ");
        diagnostic << width << " was not applied: " << failureReason;
        if (failureOp)
          diagnostic << " (operation '" << failureOp->getName() << "')";
        diagnostic << "; keeping the original loop";
      }
      return failure();
    }

    LDBG("Built vectorization plan with factor " << plan->width << " for "
                                                 << loop.getLoc());
    // Keep the original loop until the whole rewrite succeeds. Both paths
    // insert before it, so rollback also removes any strip-mining bounds.
    Operation *previous = loop->getPrevNode();
    LogicalResult result = plan->singleGroup
                               ? emitBody(*plan, loop.getLowerBound())
                               : emitStripMinedLoop(*plan);
    if (failed(result)) {
      Operation *current =
          previous ? previous->getNextNode() : &loop->getBlock()->front();
      SmallVector<Operation *> created;
      while (current != loop) {
        created.push_back(current);
        current = current->getNextNode();
      }
      for (Operation *op : llvm::reverse(created))
        op->erase();
      loop.emitWarning() << "loop vectorization with factor " << width
                         << " was not applied: plan execution failed; keeping "
                            "the original loop";
      return failure();
    }

    LDBG(
        "Applied vectorization with factor "
        << plan->width << " using "
        << (plan->singleGroup ? "direct replacement" : "main loop and epilogue")
        << "; processed " << plan->operations.size() << " operations at "
        << loop.getLoc());
    loop.erase();
    return success();
  }

private:
  LogicalResult reject(StringRef reason, Operation *op = nullptr) {
    failureReason = reason;
    failureOp = op;
    LDBG("Cannot vectorize loop with factor " << width << ": " << reason);
    if (op)
      LDBG("Unsupported operation '" << op->getName() << "' at "
                                     << op->getLoc());
    return failure();
  }

  // The plan only describes unencoded TTIR tensors and scalar element types.
  // Block pointers, descriptors, vectors and tokens cannot be splatted.
  static bool isLiftableType(Type type) {
    if (auto tensor = dyn_cast<RankedTensorType>(type)) {
      if (!tensor.hasStaticShape() || tensor.getEncoding())
        return false;
      type = tensor.getElementType();
    }
    if (auto ptr = dyn_cast<PointerType>(type))
      return isa<IntegerType, FloatType>(ptr.getPointeeType());
    return isa<IntegerType, FloatType>(type);
  }

  LogicalResult validateReduce(ReduceOp reduce) {
    if (reduce.getNumOperands() != 1 || reduce.getNumResults() != 1)
      return reject("reductions must have one operand and one result", reduce);
    Region &combine = reduce.getCombineOp();
    for (Operation &op : combine.front()) {
      if (op.getNumRegions() != 0 || !isMemoryEffectFree(&op))
        return reject(
            "reduction combiners must be region-free and memory-effect-free",
            &op);
      for (Value operand : op.getOperands()) {
        Region *parent = operand.getParentRegion();
        // A loop-local capture would still refer to an erased value after
        // cloning, and an iteration-varying value is not a scalar combiner
        // input.
        if (!combine.isAncestor(parent) && loop.getRegion().isAncestor(parent))
          return reject("reduction combiner captures a loop-local value", &op);
      }
    }
    return success();
  }

  LogicalResult validateOperation(Operation *op) {
    if (!llvm::all_of(op->getOperandTypes(), isLiftableType) ||
        !llvm::all_of(op->getResultTypes(), isLiftableType))
      return reject("unsupported operand or result type", op);
    if (auto reduce = dyn_cast<ReduceOp>(op))
      return validateReduce(reduce);
    if (op->getNumRegions() != 0)
      return reject("nested regions are not supported", op);
    if (op->getNumResults() > 1)
      return reject("multiple results are not supported", op);
    if (auto load = dyn_cast<LoadOp>(op)) {
      if (!load.getBoundaryCheck().empty() || load.getPadding())
        return reject("load boundary checks and padding are not supported", op);
      if (load.getIsVolatile())
        return reject("volatile loads are not supported", op);
      return success();
    }
    if (auto store = dyn_cast<StoreOp>(op)) {
      if (!store.getBoundaryCheck().empty())
        return reject("store boundary checks are not supported", op);
      return success();
    }
    if (isa<arith::ConstantOp, MakeRangeOp, SplatOp, ExpandDimsOp, BroadcastOp,
            ReshapeOp, AddPtrOp, BitcastOp, ClampFOp, FpToFpOp>(op))
      return success();
    // Tensorizable also permits scalar operands to become tensors, such as a
    // scalar select condition. Elementwise alone does not provide this promise.
    if (!op->hasTrait<OpTrait::Elementwise>() ||
        !op->hasTrait<OpTrait::Tensorizable>() || !isMemoryEffectFree(op) ||
        op->getNumResults() != 1)
      return reject("operation has no supported tensorization", op);
    return success();
  }

  DictionaryAttr planAttributes(Operation *op) {
    NamedAttrList attributes(op->getAttrs());
    if (isa<ReshapeOp>(op))
      attributes.erase("allow_reorder");
    if (op->getNumResults() != 1)
      return attributes.getDictionary(loop.getContext());
    auto tensor = dyn_cast<RankedTensorType>(op->getResult(0).getType());
    if (!tensor)
      return attributes.getDictionary(loop.getContext());
    // Rank-dependent analysis hints gain a conservative iteration axis.
    for (StringRef name :
         {"tt.contiguity", "tt.divisibility", "tt.constancy"}) {
      Attribute hint = attributes.get(name);
      if (!hint)
        continue;
      SmallVector<int64_t> values{1};
      if (auto dense = dyn_cast<DenseIntElementsAttr>(hint)) {
        for (const APInt &value : dense.getValues<APInt>())
          values.push_back(value.getSExtValue());
      } else if (auto integer = dyn_cast<IntegerAttr>(hint)) {
        values.push_back(integer.getInt());
      }
      if (values.size() != static_cast<size_t>(tensor.getRank() + 1))
        attributes.erase(name);
      else
        attributes.set(name, builder.getI64TensorAttr(values));
    }
    return attributes.getDictionary(loop.getContext());
  }

  LogicalResult planOperands(OperationVectorizationPlan &plan,
                             const DenseMap<Value, Type> &types) {
    if (plan.kind == VectorizationKind::CloneAndBroadcast)
      return success();
    RankedTensorType tensorResult;
    if (plan.kind == VectorizationKind::Generic &&
        plan.op->hasTrait<OpTrait::Elementwise>() &&
        plan.op->getNumResults() == 1)
      tensorResult =
          dyn_cast<RankedTensorType>(plan.op->getResult(0).getType());
    for (Value operand : plan.op->getOperands()) {
      Type sourceType = types.lookup(operand);
      if (!sourceType) {
        if (loop.getRegion().isAncestor(operand.getParentRegion()))
          return reject("operand has no planned definition", plan.op);
        sourceType = operand.getType();
      }
      bool scalar = plan.kind == VectorizationKind::Splat ||
                    (tensorResult && !isa<RankedTensorType>(operand.getType()));
      RankedTensorType target = prependIterationDim(operand.getType(), width);
      if (plan.kind == VectorizationKind::Splat)
        target = cast<RankedTensorType>(plan.resultTypes.front());
      else if (scalar)
        target =
            prependIterationDim(tensorResult, width).clone(operand.getType());
      if (scalar) {
        if (auto tensor = dyn_cast<RankedTensorType>(sourceType)) {
          if (tensor.getRank() != 1 || tensor.getShape()[0] != width ||
              tensor.getElementType() != target.getElementType())
            return reject(
                "scalar operand cannot broadcast to the planned shape",
                plan.op);
        } else if (sourceType != target.getElementType()) {
          return reject("scalar operand element type does not match", plan.op);
        }
      } else if (types.contains(operand) && sourceType != target) {
        return reject("operand type does not match the planned shape", plan.op);
      }
      plan.operands.push_back({operand, target, scalar});
    }
    return success();
  }

  FailureOr<LoopVectorizationPlan> buildPlan() {
    if (width <= 1) {
      LDBG("Loop vectorization is disabled at " << loop.getLoc());
      return failure();
    }
    if (!llvm::isPowerOf2_64(width))
      return reject("factor must be a power of two");
    if (!loop.getInductionVar().getType().isInteger(32))
      return reject("only i32 induction variables are supported");
    auto step = getConstantInt(loop.getStep());
    if (!step || *step <= 0)
      return reject("step must be a positive constant");
    if (width > std::numeric_limits<int32_t>::max() / *step)
      return reject("vectorized step does not fit in i32");
    if (loop.getUnsignedCmp())
      return reject("unsigned loop bounds are not supported");

    LoopVectorizationPlan plan{loop, width, *step};
    auto lower = getConstantInt(loop.getLowerBound());
    auto upper = getConstantInt(loop.getUpperBound());
    if (lower && upper) {
      int64_t distance = *upper - *lower;
      if (distance <= 0) {
        LDBG("Skipping an empty loop at " << loop.getLoc());
        return failure();
      }
      int64_t tripCount = llvm::divideCeil(distance, *step);
      if (tripCount < width)
        return reject("iteration count is smaller than vectorization factor");
      plan.singleGroup = tripCount == width;
    }
    if (!loop.getInitArgs().empty() || loop->getNumResults() != 0)
      return reject("loop-carried values are not supported");

    DenseMap<Value, Type> types;
    types[loop.getInductionVar()] =
        prependIterationDim(loop.getInductionVar().getType(), width);
    for (Operation &op : loop.getBody()->without_terminator()) {
      if (failed(validateOperation(&op)))
        return failure();
      OperationVectorizationPlan operation;
      operation.op = &op;
      for (Type type : op.getResultTypes())
        operation.resultTypes.push_back(prependIterationDim(type, width));
      if (isa<arith::ConstantOp, MakeRangeOp>(op)) {
        operation.kind = VectorizationKind::CloneAndBroadcast;
      } else if (isa<SplatOp>(op)) {
        operation.kind = VectorizationKind::Splat;
      } else if (auto expand = dyn_cast<ExpandDimsOp>(op)) {
        operation.kind = VectorizationKind::ExpandDims;
        operation.axis = expand.getAxis() + 1;
      } else if (auto reduce = dyn_cast<ReduceOp>(op)) {
        operation.kind = VectorizationKind::Reduce;
        operation.axis = reduce.getAxis() + 1;
      } else if (auto reshape = dyn_cast<ReshapeOp>(op)) {
        // Reordering must stay within each original iteration. Identity
        // reshapes emitted by tl.max can use their planned input directly.
        operation.kind =
            reshape.getSrc().getType() == reshape.getResult().getType()
                ? VectorizationKind::Identity
                : VectorizationKind::Reshape;
      }
      if (failed(planOperands(operation, types)))
        return failure();
      if (operation.kind == VectorizationKind::Generic ||
          operation.kind == VectorizationKind::Reshape)
        operation.attributes = planAttributes(&op);
      for (auto [result, type] :
           llvm::zip(op.getResults(), operation.resultTypes))
        types[result] = type;
      plan.operations.push_back(std::move(operation));
    }
    return plan;
  }

  // Broadcast a scalar, or its widened [VF] value, to the planned tensor type.
  Value broadcastScalar(Value value, RankedTensorType target) {
    if (!isa<RankedTensorType>(value.getType()))
      return SplatOp::create(builder, loop.getLoc(), target, value);
    while (cast<RankedTensorType>(value.getType()).getRank() <
           target.getRank()) {
      auto type = cast<RankedTensorType>(value.getType());
      SmallVector<int64_t> shape(type.getShape());
      shape.push_back(1);
      auto expanded = RankedTensorType::get(shape, type.getElementType());
      value = ExpandDimsOp::create(builder, loop.getLoc(), expanded, value,
                                   type.getRank());
    }
    if (value.getType() == target)
      return value;
    return BroadcastOp::create(builder, loop.getLoc(), target, value);
  }

  Value broadcastMajor(Value value, RankedTensorType target) {
    if (!isa<RankedTensorType>(value.getType()))
      return SplatOp::create(builder, loop.getLoc(), target, value);
    auto source = cast<RankedTensorType>(value.getType());
    SmallVector<int64_t> shape(source.getShape());
    shape.insert(shape.begin(), 1);
    auto expandedType = RankedTensorType::get(shape, source.getElementType());
    Value expanded =
        ExpandDimsOp::create(builder, loop.getLoc(), expandedType, value, 0);
    return BroadcastOp::create(builder, loop.getLoc(), target, expanded);
  }

  LogicalResult emitOperation(const OperationVectorizationPlan &plan) {
    IRMapping operands;
    SmallVector<Value> inputs;
    for (const auto &operand : plan.operands) {
      Value value = mapping.lookupOrNull(operand.source);
      if (operand.broadcastScalar)
        value = broadcastScalar(value ? value : operand.source, operand.type);
      else if (!value)
        value = broadcastMajor(operand.source, operand.type);
      if (value.getType() != operand.type)
        return failure();
      operands.map(operand.source, value);
      inputs.push_back(value);
    }

    Value result;
    switch (plan.kind) {
    case VectorizationKind::CloneAndBroadcast: {
      Operation *copy = builder.clone(*plan.op);
      result = broadcastMajor(copy->getResult(0),
                              cast<RankedTensorType>(plan.resultTypes.front()));
      break;
    }
    case VectorizationKind::Splat:
    case VectorizationKind::Identity:
      result = inputs.front();
      break;
    case VectorizationKind::ExpandDims:
      result = ExpandDimsOp::create(builder, plan.op->getLoc(),
                                    plan.resultTypes.front(), inputs.front(),
                                    plan.axis);
      break;
    case VectorizationKind::Reduce: {
      auto wide =
          ReduceOp::create(builder, plan.op->getLoc(), inputs, plan.axis);
      auto reduce = cast<ReduceOp>(plan.op);
      builder.cloneRegionBefore(reduce.getCombineOp(), wide.getCombineOp(),
                                wide.getCombineOp().end());
      result = wide.getResult().front();
      break;
    }
    case VectorizationKind::Reshape:
    case VectorizationKind::Generic: {
      Operation *copy = builder.clone(*plan.op, operands);
      copy->setAttrs(plan.attributes);
      for (auto [newResult, type] :
           llvm::zip(copy->getResults(), plan.resultTypes))
        newResult.setType(type);
      if (copy->getNumResults() == 1)
        result = copy->getResult(0);
      break;
    }
    }
    if (!plan.resultTypes.empty()) {
      if (!result || result.getType() != plan.resultTypes.front())
        return failure();
      mapping.map(plan.op->getResult(0), result);
    }
    return success();
  }

  LogicalResult emitBody(const LoopVectorizationPlan &plan, Value base) {
    auto loop = plan.loop;
    auto type =
        RankedTensorType::get({plan.width}, loop.getInductionVar().getType());
    Value indices = MakeRangeOp::create(builder, loop.getLoc(), type, 0,
                                        static_cast<int32_t>(plan.width));
    Value step = SplatOp::create(builder, loop.getLoc(), type, loop.getStep());
    Value lower = SplatOp::create(builder, loop.getLoc(), type, base);
    Value scaled =
        arith::MulIOp::create(builder, loop.getLoc(), type, indices, step);
    Value vectorIV =
        arith::AddIOp::create(builder, loop.getLoc(), type, lower, scaled);
    mapping.map(loop.getInductionVar(), vectorIV);
    for (const auto &operation : plan.operations)
      if (failed(emitOperation(operation)))
        return failure();
    return success();
  }

  LogicalResult emitStripMinedLoop(const LoopVectorizationPlan &plan) {
    auto loop = plan.loop;
    // Count iterations in i64 to avoid subtraction overflow and include full
    // groups when the upper bound is not aligned to the original step.
    Location loc = loop.getLoc();
    Value zero = arith::ConstantIntOp::create(builder, loc, 0, 64);
    Value factor64 = arith::ConstantIntOp::create(builder, loc, plan.width, 64);
    Value step64 = arith::ConstantIntOp::create(builder, loc, plan.step, 64);
    Value lower64 = arith::ExtSIOp::create(builder, loc, builder.getI64Type(),
                                           loop.getLowerBound());
    Value upper64 = arith::ExtSIOp::create(builder, loc, builder.getI64Type(),
                                           loop.getUpperBound());
    Value vectorStep =
        arith::ConstantIntOp::create(builder, loc, plan.width * plan.step, 32);
    Value distance = arith::SubIOp::create(builder, loc, upper64, lower64);
    Value safeDistance = arith::MaxSIOp::create(builder, loc, distance, zero);
    Value tripCount =
        arith::CeilDivUIOp::create(builder, loc, safeDistance, step64);
    Value remainder = arith::RemUIOp::create(builder, loc, tripCount, factor64);
    Value fullTrips = arith::SubIOp::create(builder, loc, tripCount, remainder);
    Value vectorDistance =
        arith::MulIOp::create(builder, loc, fullTrips, step64);
    Value end64 = arith::AddIOp::create(builder, loc, lower64, vectorDistance);
    end64 = arith::MinSIOp::create(builder, loc, end64, upper64);
    Value vectorEnd =
        arith::TruncIOp::create(builder, loc, builder.getI32Type(), end64);

    auto vectorLoop = scf::ForOp::create(builder, loc, loop.getLowerBound(),
                                         vectorEnd, vectorStep);
    vectorLoop->setAttrs(loop->getAttrs());
    vectorLoop->removeAttr(kVectorFactorAttr);
    builder.setInsertionPoint(vectorLoop.getBody()->getTerminator());
    if (failed(emitBody(plan, vectorLoop.getInductionVar())))
      return failure();

    builder.setInsertionPointAfter(vectorLoop);
    auto tailLoop = scf::ForOp::create(builder, loc, vectorEnd,
                                       loop.getUpperBound(), loop.getStep());
    tailLoop->setAttrs(loop->getAttrs());
    tailLoop->removeAttr(kVectorFactorAttr);
    tailLoop->setAttr("tt.num_stages", builder.getI32IntegerAttr(1));
    builder.setInsertionPoint(tailLoop.getBody()->getTerminator());
    IRMapping tailMapping;
    tailMapping.map(loop.getInductionVar(), tailLoop.getInductionVar());
    for (Operation &op : loop.getBody()->without_terminator())
      builder.clone(op, tailMapping);
    return success();
  }
};

class LoopVectorizePass
    : public impl::TritonLoopVectorizeBase<LoopVectorizePass> {
public:
  void runOnOperation() override {
    SmallVector<scf::ForOp> loops;
    getOperation()->walk([&](scf::ForOp loop) {
      if (loop->hasAttr(kVectorFactorAttr))
        loops.push_back(loop);
    });

    for (auto loop : loops) {
      if (auto factor = loop->getAttrOfType<IntegerAttr>(kVectorFactorAttr)) {
        if (failed(LoopVectorizer(loop, factor.getInt()).run()))
          continue;
      }
    }
  }
};

} // namespace
} // namespace mlir::triton
