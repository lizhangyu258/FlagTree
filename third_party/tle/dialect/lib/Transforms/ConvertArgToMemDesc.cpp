/*
 * Copyright 2025-     FlagOS Contributors
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files
 * (the "Software"), to deal in the Software without restriction,
 * including without limitation the rights to use, copy, modify, merge,
 * publish, distribute, sublicense, and/or sell copies of the Software,
 * and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#include "tle/dialect/include/Transforms/ConvertArgToMemDesc.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "tle/dialect/include/IR/Dialect.h"
#include "tle/dialect/include/Transforms/Passes.h"
#include "tle/dialect/include/Transforms/TleUtility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Types.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/Casting.h"

namespace mlir::triton::tle {
#define GEN_PASS_DEF_TLECONVERTARGTOMEMDESC
#include "tle/dialect/include/Transforms/Passes.h.inc"
} // namespace mlir::triton::tle

using namespace mlir;
namespace ttg = mlir::triton::gpu;
namespace tle = mlir::triton::tle;

namespace {

ttg::MemDescType getPlainMemDesc(RankedTensorType ty) {
  ttg::CTAEncodingAttr ctaLayout = ttg::getCTALayout(ty.getEncoding());
  llvm::iota_range<uint32_t> rOrderRange =
      llvm::iota_range<uint32_t>(0, ty.getRank(), false);
  llvm::SmallVector<uint32_t> order = ttg::getOrder(ty);
  return ttg::MemDescType::get(ty.getShape(), ty.getElementType(),
                               ttg::SwizzledSharedEncodingAttr::get(
                                   ty.getContext(), 1, 1, 1, order, ctaLayout),
                               ttg::SharedMemorySpaceAttr::get(ty.getContext()),
                               true);
}

Value stripConvertLayouts(Value value) {
  while (auto convert = value.getDefiningOp<ttg::ConvertLayoutOp>())
    value = convert.getSrc();
  return value;
}

// Recover an existing full-tile shared-memory descriptor from a tensor read.
// Keep this deliberately narrow: arbitrary local-pointer indices and masked
// loads do not necessarily describe the complete backing memdesc.
Value getReusableMemDesc(Value value, RankedTensorType tensorTy) {
  Value source = stripConvertLayouts(value);
  if (auto localLoad = source.getDefiningOp<ttg::LocalLoadOp>()) {
    auto memDescTy = dyn_cast<ttg::MemDescType>(localLoad.getSrc().getType());
    if (memDescTy && memDescTy.getShape() == tensorTy.getShape() &&
        memDescTy.getElementType() == tensorTy.getElementType())
      return localLoad.getSrc();
    return {};
  }

  if (auto load = source.getDefiningOp<triton::LoadOp>()) {
    if (load.getMask() || load.getIsVolatile())
      return {};

    auto localPointers = stripConvertLayouts(load.getPtr())
                             .getDefiningOp<tle::LocalPointersOp>();
    if (!localPointers || !localPointers.getIndices().empty())
      return {};

    auto memDescTy =
        dyn_cast<ttg::MemDescType>(localPointers.getSrc().getType());
    if (memDescTy && memDescTy.getShape() == tensorTy.getShape() &&
        memDescTy.getElementType() == tensorTy.getElementType())
      return localPointers.getSrc();
  }

  return {};
}

struct TleArgConversion : public OpRewritePattern<tle::DSLRegionOp> {
  using OpRewritePattern::OpRewritePattern;

  TleArgConversion(MLIRContext *context);
  LogicalResult matchAndRewrite(tle::DSLRegionOp op,
                                PatternRewriter &rewriter) const override;
};

struct TleConvertArgToMemDesc
    : public tle::impl::TleConvertArgToMemDescBase<TleConvertArgToMemDesc> {
  void runOnOperation() override;
};

} // namespace

TleArgConversion::TleArgConversion(MLIRContext *context)
    : OpRewritePattern(context) {}

LogicalResult
TleArgConversion::matchAndRewrite(tle::DSLRegionOp op,
                                  PatternRewriter &rewriter) const {
  bool hasConversion = false;
  for (Type type : op->getOperandTypes())
    hasConversion |= isa<RankedTensorType>(type);
  for (Type type : op->getResultTypes())
    hasConversion |= isa<RankedTensorType>(type);
  if (!hasConversion)
    return failure();

  SmallVector<Value> newOperands;
  IRMapping mapper;
  bool needSync = false;
  for (const auto &operand : op->getOperands()) {
    if (RankedTensorType tensorTy =
            dyn_cast<RankedTensorType>(operand.getType())) {
      if (Value memDesc = getReusableMemDesc(operand, tensorTy)) {
        newOperands.push_back(memDesc);
        mapper.map(operand, memDesc);
        needSync = true;
        continue;
      }

      PatternRewriter::InsertionGuard guard(rewriter);
      rewriter.setInsertionPoint(op);
      ttg::LocalAllocOp allocOp = rewriter.create<ttg::LocalAllocOp>(
          op->getLoc(), getPlainMemDesc(tensorTy));
      rewriter.create<ttg::LocalStoreOp>(op->getLoc(), operand, allocOp);
      rewriter.setInsertionPointAfter(op);
      rewriter.create<ttg::LocalDeallocOp>(op->getLoc(), allocOp);

      newOperands.push_back(allocOp);
      mapper.map(operand, allocOp);
      needSync = true;
    } else {
      if (isa<ttg::MemDescType>(operand.getType())) {
        needSync = true;
      }
      newOperands.push_back(operand);
    }
  }
  if (needSync) {
    PatternRewriter::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(op);
    rewriter.create<NVVM::Barrier0Op>(op.getLoc());
  }
  SmallVector<Type> newRetTys;
  auto outputIndices = op.getOutputOperandIndices();
  for (auto [resultIdx, result] : llvm::enumerate(op.getResults())) {
    if (RankedTensorType tensorTy =
            dyn_cast<RankedTensorType>(result.getType())) {
      Type resultTy = getPlainMemDesc(tensorTy);
      if (resultIdx < outputIndices.size()) {
        int64_t operandIdx = outputIndices[resultIdx];
        if (operandIdx >= 0 &&
            operandIdx < static_cast<int64_t>(newOperands.size()) &&
            isa<ttg::MemDescType>(newOperands[operandIdx].getType()))
          resultTy = newOperands[operandIdx].getType();
      }
      newRetTys.push_back(resultTy);
    } else {
      newRetTys.push_back(result.getType());
    }
  }
  tle::DSLRegionOp newOp = rewriter.create<tle::DSLRegionOp>(
      op.getLoc(), newRetTys, newOperands, op.getRegionDialectAttr(),
      op.getArgDialectAttr(), op.getOutputOperandIndicesAttr(),
      op->getAttrOfType<StringAttr>("hint"));
  newOp->setAttrs(op->getAttrs());
  PatternRewriter::InsertionGuard guard(rewriter);
  DenseMap<Value, Type> yieldedTypes;
  for (Block &block : op.getBody()) {
    auto yield = dyn_cast<tle::YieldOp>(block.getTerminator());
    if (!yield)
      continue;
    for (auto [idx, value] : llvm::enumerate(yield.getInputs())) {
      if (idx < newRetTys.size())
        yieldedTypes[value] = newRetTys[idx];
    }
  }
  for (auto [idx, oldBlock] : llvm::enumerate(op.getBody().getBlocks())) {
    Block *newBlock = nullptr;
    if (idx == 0) {
      newBlock = rewriter.createBlock(
          &newOp.getBody(), {}, newOp->getOperandTypes(),
          SmallVector<Location>(newOp->getNumOperands(), op.getLoc()));
    } else {
      newBlock = rewriter.createBlock(
          &newOp.getBody(), {}, oldBlock.getArgumentTypes(),
          SmallVector<Location>(oldBlock.getNumArguments(), op.getLoc()));
    }
    for (auto [oldArg, newArg] :
         llvm::zip(oldBlock.getArguments(), newBlock->getArguments())) {
      mapper.map(oldArg, newArg);
    }
    mapper.map(&oldBlock, newBlock);
  }
  for (auto [oldBlock, newBlock] :
       llvm::zip(op.getBody().getBlocks(), newOp.getBody().getBlocks())) {
    rewriter.setInsertionPointToEnd(&newBlock);
    for (Operation &operation : oldBlock.getOperations()) {
      if (tle::PackOp packOp = dyn_cast<tle::PackOp>(operation)) {
        if (auto tensorTy =
                dyn_cast<RankedTensorType>(packOp.getOutput().getType())) {
          Type packTy = getPlainMemDesc(tensorTy);
          if (auto it = yieldedTypes.find(packOp.getOutput());
              it != yieldedTypes.end())
            packTy = it->second;
          tle::PackOp newPackOp = rewriter.create<tle::PackOp>(
              packOp.getLoc(), packTy, mapper.lookup(packOp.getInput()));
          mapper.map(packOp.getOutput(), newPackOp.getOutput());
          continue;
        }
      }
      rewriter.clone(operation, mapper);
    }
  }
  rewriter.setInsertionPointAfter(newOp);
  SmallVector<Value> results;
  for (auto [oldResult, newResult] :
       llvm::zip(op.getResults(), newOp.getResults())) {
    if (RankedTensorType tensorTy =
            dyn_cast<RankedTensorType>(oldResult.getType())) {
      ttg::LocalLoadOp loadOp =
          rewriter.create<ttg::LocalLoadOp>(op.getLoc(), tensorTy, newResult);
      results.push_back(loadOp);
    } else {
      results.push_back(newResult);
    }
  }
  rewriter.replaceOp(op, results);
  return success();
}

void mlir::triton::tle::populateConvertArgToMemDescPatterns(
    RewritePatternSet &patterns) {
  patterns.add<TleArgConversion>(patterns.getContext());
}

void TleConvertArgToMemDesc::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  tle::populateConvertArgToMemDescPatterns(patterns);
  if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
    signalPassFailure();
  }
}
