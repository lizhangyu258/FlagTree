/*
 * Copyright 2026 FlagOS Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/Transforms/Passes.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/MathExtras.h"

namespace mlir::triton {

#define GEN_PASS_DEF_TRITONLOOPVECTORIZE
#include "triton/Dialect/Triton/Transforms/Passes.h.inc"

namespace {

constexpr StringLiteral kVectorFactorAttr = "tt.loop_vectorize_factor";

static std::optional<int64_t> getConstantInt(Value value) {
  if (auto constant = value.getDefiningOp<arith::ConstantOp>())
    if (auto integer = dyn_cast<IntegerAttr>(constant.getValue()))
      return integer.getInt();
  return std::nullopt;
}

static Type prependLane(Type type, int64_t width) {
  if (auto tensor = dyn_cast<RankedTensorType>(type)) {
    SmallVector<int64_t> shape(tensor.getShape());
    shape.insert(shape.begin(), width);
    return RankedTensorType::get(shape, tensor.getElementType(),
                                 tensor.getEncoding());
  }
  return RankedTensorType::get({width}, type);
}

class DirectLoopVectorizer {
public:
  DirectLoopVectorizer(scf::ForOp loop, int64_t width)
      : loop(loop), width(width), builder(loop) {}

  LogicalResult run() {
    if (!loop.getInitArgs().empty() || loop->getNumResults() != 0)
      return failure();
    if (failed(validateBody()))
      return failure();

    Operation *previous = loop->getPrevNode();
    auto rollback = [&] {
      Operation *current =
          previous ? previous->getNextNode() : &loop->getBlock()->front();
      SmallVector<Operation *> created;
      while (current != loop) {
        created.push_back(current);
        current = current->getNextNode();
      }
      for (Operation *op : llvm::reverse(created))
        op->erase();
    };

    if (failed(buildVectorIV(loop.getLowerBound()))) {
      rollback();
      return failure();
    }

    for (Operation &op : loop.getBody()->without_terminator()) {
      if (failed(liftOperation(&op))) {
        rollback();
        return failure();
      }
    }

    loop.erase();
    return success();
  }

  LogicalResult runStripMined(Value vectorEnd, Value vectorStep) {
    if (!loop.getInitArgs().empty() || loop->getNumResults() != 0)
      return failure();
    if (failed(validateBody()))
      return failure();

    auto vectorLoop = scf::ForOp::create(
        builder, loop.getLoc(), loop.getLowerBound(), vectorEnd, vectorStep);
    vectorLoop->setAttrs(loop->getAttrs());
    vectorLoop->removeAttr(kVectorFactorAttr);
    builder.setInsertionPoint(vectorLoop.getBody()->getTerminator());

    if (failed(buildVectorIV(vectorLoop.getInductionVar()))) {
      vectorLoop.erase();
      return failure();
    }
    for (Operation &op : loop.getBody()->without_terminator()) {
      if (failed(liftOperation(&op))) {
        vectorLoop.erase();
        return failure();
      }
    }

    builder.setInsertionPointAfter(vectorLoop);
    auto tailLoop = scf::ForOp::create(builder, loop.getLoc(), vectorEnd,
                                       loop.getUpperBound(), loop.getStep());
    tailLoop->setAttrs(loop->getAttrs());
    tailLoop->removeAttr(kVectorFactorAttr);
    builder.setInsertionPoint(tailLoop.getBody()->getTerminator());
    IRMapping tailMapping;
    tailMapping.map(loop.getInductionVar(), tailLoop.getInductionVar());
    for (Operation &op : loop.getBody()->without_terminator())
      builder.clone(op, tailMapping);

    loop.erase();
    return success();
  }

private:
  LogicalResult validateBody() {
    for (Operation &op : loop.getBody()->without_terminator()) {
      if (op.getNumRegions() != 0 && !isa<ReduceOp>(op))
        return failure();
      if (op.getNumResults() > 1)
        return failure();
      if (isa<arith::ConstantOp, MakeRangeOp, SplatOp, ExpandDimsOp,
              BroadcastOp, ReshapeOp, LoadOp, StoreOp, AddPtrOp, ReduceOp,
              BitcastOp, ClampFOp, FpToFpOp, arith::AddFOp, arith::SubFOp,
              arith::MulFOp, arith::DivFOp, arith::AddIOp, arith::SubIOp,
              arith::MulIOp, arith::CmpFOp, arith::CmpIOp, arith::SelectOp,
              arith::ExtFOp, arith::TruncFOp, arith::ExtSIOp, arith::ExtUIOp,
              arith::TruncIOp, arith::UIToFPOp, arith::MaxNumFOp, math::AbsFOp,
              math::SqrtOp, math::Log2Op, math::CeilOp, math::Exp2Op>(op))
        continue;
      return failure();
    }
    return success();
  }

  LogicalResult buildVectorIV(Value base) {
    Type ivType = loop.getInductionVar().getType();
    if (!ivType.isInteger(32))
      return failure();
    auto laneType = RankedTensorType::get({width}, ivType);
    Value lanes = MakeRangeOp::create(builder, loop.getLoc(), laneType, 0,
                                      static_cast<int32_t>(width));
    Value step =
        SplatOp::create(builder, loop.getLoc(), laneType, loop.getStep());
    Value lower = SplatOp::create(builder, loop.getLoc(), laneType, base);
    Value scaled =
        arith::MulIOp::create(builder, loop.getLoc(), laneType, lanes, step);
    Value vectorIV =
        arith::AddIOp::create(builder, loop.getLoc(), laneType, lower, scaled);
    mapping.map(loop.getInductionVar(), vectorIV);
    return success();
  }

  Value broadcastMajor(Value value) {
    Type target = prependLane(value.getType(), width);
    if (value.getType() == target)
      return value;
    auto targetTensor = cast<RankedTensorType>(target);
    if (!isa<RankedTensorType>(value.getType()))
      return SplatOp::create(builder, loop.getLoc(), targetTensor, value);

    auto source = cast<RankedTensorType>(value.getType());
    SmallVector<int64_t> expandedShape(source.getShape());
    expandedShape.insert(expandedShape.begin(), 1);
    auto expandedType = RankedTensorType::get(
        expandedShape, source.getElementType(), source.getEncoding());
    Value expanded =
        ExpandDimsOp::create(builder, loop.getLoc(), expandedType, value, 0);
    return BroadcastOp::create(builder, loop.getLoc(), targetTensor, expanded);
  }

  Value mappedOrBroadcast(Value value) {
    if (Value mapped = mapping.lookupOrNull(value))
      return mapped;
    return broadcastMajor(value);
  }

  LogicalResult liftConstant(arith::ConstantOp constant) {
    Operation *copy = builder.clone(*constant);
    Value result = broadcastMajor(copy->getResult(0));
    mapping.map(constant.getResult(), result);
    return success();
  }

  LogicalResult liftMakeRange(MakeRangeOp range) {
    Operation *copy = builder.clone(*range);
    Value result = broadcastMajor(copy->getResult(0));
    mapping.map(range.getResult(), result);
    return success();
  }

  LogicalResult liftSplat(SplatOp splat) {
    Value source = mapping.lookupOrDefault(splat.getSrc());
    auto resultType =
        cast<RankedTensorType>(prependLane(splat.getResult().getType(), width));
    Value result;
    if (!isa<RankedTensorType>(source.getType())) {
      result = SplatOp::create(builder, splat.getLoc(), resultType, source);
    } else {
      auto sourceType = cast<RankedTensorType>(source.getType());
      if (sourceType.getRank() != 1 || sourceType.getShape()[0] != width)
        return failure();
      auto expandedType = RankedTensorType::get(
          {width, 1}, sourceType.getElementType(), sourceType.getEncoding());
      result = ExpandDimsOp::create(builder, splat.getLoc(), expandedType,
                                    source, 1);
      result = BroadcastOp::create(builder, splat.getLoc(), resultType, result);
    }
    mapping.map(splat.getResult(), result);
    return success();
  }

  LogicalResult liftExpandDims(ExpandDimsOp expand) {
    Value source = mappedOrBroadcast(expand.getSrc());
    auto resultType = cast<RankedTensorType>(
        prependLane(expand.getResult().getType(), width));
    Value result = ExpandDimsOp::create(builder, expand.getLoc(), resultType,
                                        source, expand.getAxis() + 1);
    mapping.map(expand.getResult(), result);
    return success();
  }

  LogicalResult liftReduce(ReduceOp reduce) {
    if (reduce->getNumOperands() != 1 || reduce->getNumResults() != 1)
      return failure();
    Value source = mappedOrBroadcast(reduce.getSrcs().front());
    auto wide = ReduceOp::create(builder, reduce.getLoc(), ValueRange{source},
                                 reduce.getAxis() + 1);
    builder.cloneRegionBefore(reduce.getCombineOp(), wide.getCombineOp(),
                              wide.getCombineOp().end());
    mapping.map(reduce.getResult().front(), wide.getResult().front());
    return success();
  }

  LogicalResult liftReshape(ReshapeOp reshape) {
    Value source = mappedOrBroadcast(reshape.getSrc());
    // A reorderable reshape inside one scalar iteration must not be allowed
    // to exchange elements between newly introduced lanes. Identity reshapes
    // emitted by tl.max can be removed outright.
    if (reshape.getSrc().getType() == reshape.getResult().getType()) {
      mapping.map(reshape.getResult(), source);
      return success();
    }
    IRMapping operands;
    operands.map(reshape.getSrc(), source);
    Operation *copy = builder.clone(*reshape, operands);
    copy->removeAttr("allow_reorder");
    copy->getResult(0).setType(
        prependLane(reshape.getResult().getType(), width));
    mapping.map(reshape.getResult(), copy->getResult(0));
    return success();
  }

  LogicalResult liftGeneric(Operation *op) {
    IRMapping operands;
    for (Value operand : op->getOperands())
      operands.map(operand, mappedOrBroadcast(operand));
    Operation *copy = builder.clone(*op, operands);
    for (auto [oldResult, newResult] :
         llvm::zip(op->getResults(), copy->getResults())) {
      newResult.setType(prependLane(oldResult.getType(), width));
      mapping.map(oldResult, newResult);
    }
    return success();
  }

  LogicalResult liftOperation(Operation *op) {
    if (auto constant = dyn_cast<arith::ConstantOp>(op))
      return liftConstant(constant);
    if (auto range = dyn_cast<MakeRangeOp>(op))
      return liftMakeRange(range);
    if (auto splat = dyn_cast<SplatOp>(op))
      return liftSplat(splat);
    if (auto expand = dyn_cast<ExpandDimsOp>(op))
      return liftExpandDims(expand);
    if (auto reduce = dyn_cast<ReduceOp>(op))
      return liftReduce(reduce);
    if (auto reshape = dyn_cast<ReshapeOp>(op))
      return liftReshape(reshape);
    return liftGeneric(op);
  }

  scf::ForOp loop;
  int64_t width;
  OpBuilder builder;
  IRMapping mapping;
};

class LoopVectorizePass
    : public impl::TritonLoopVectorizeBase<LoopVectorizePass> {
public:
  void runOnOperation() override {
    SmallVector<scf::ForOp> loops;
    getOperation().walk([&](scf::ForOp loop) {
      if (loop->hasAttr(kVectorFactorAttr))
        loops.push_back(loop);
    });

    for (scf::ForOp loop : loops) {
      auto factor =
          dyn_cast_or_null<IntegerAttr>(loop->getAttr(kVectorFactorAttr));
      auto lower = getConstantInt(loop.getLowerBound());
      auto upper = getConstantInt(loop.getUpperBound());
      auto step = getConstantInt(loop.getStep());
      if (!factor || !step || *step <= 0)
        continue;
      int64_t width = factor.getInt();
      if (width < 2 || !llvm::isPowerOf2_64(width) ||
          !loop.getInductionVar().getType().isInteger(32))
        continue;

      if (lower && upper) {
        int64_t distance = *upper - *lower;
        if (distance <= 0)
          continue;
        int64_t tripCount = llvm::divideCeil(distance, *step);
        if (tripCount < width)
          continue;
        if (tripCount == width) {
          (void)DirectLoopVectorizer(loop, width).run();
          continue;
        }
      }

      // Strip-mine the loop into groups of `width` iterations.  The main loop
      // executes [base + lane * step] as one leading tensor dimension, while
      // the scalar epilogue preserves the final 0 .. width-1 iterations.
      OpBuilder builder(loop);
      Location loc = loop.getLoc();
      Value zero = arith::ConstantIntOp::create(builder, loc, 0, 32);
      Value chunkStep =
          arith::ConstantIntOp::create(builder, loc, width * *step, 32);
      Value distance = arith::SubIOp::create(builder, loc, loop.getUpperBound(),
                                             loop.getLowerBound());
      Value hasIterations = arith::CmpIOp::create(
          builder, loc, arith::CmpIPredicate::sgt, distance, zero);
      Value safeDistance =
          arith::SelectOp::create(builder, loc, hasIterations, distance, zero);
      Value remainder =
          arith::RemUIOp::create(builder, loc, safeDistance, chunkStep);
      Value vectorDistance =
          arith::SubIOp::create(builder, loc, safeDistance, remainder);
      Value vectorEnd = arith::AddIOp::create(
          builder, loc, loop.getLowerBound(), vectorDistance);
      (void)DirectLoopVectorizer(loop, width)
          .runStripMined(vectorEnd, chunkStep);
    }
  }
};

} // namespace
} // namespace mlir::triton
