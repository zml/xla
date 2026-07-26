/* Copyright 2025 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "xla/service/spmd/shardy/extensions/mhlo_extensions.h"

#include <algorithm>
#include <cstdint>
#include <numeric>

#include "absl/log/check.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/ir/enums.h"
#include "shardy/dialect/sdy/ir/utils.h"
#include "shardy/dialect/sdy/transforms/propagation/op_sharding_rule_builder.h"
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"

namespace mhlo = ::mlir::mhlo;

namespace xla {
namespace sdy {

namespace {

using ::mlir::ArrayRef;
using ::mlir::sdy::FactorType;
using ::mlir::sdy::getTensorShape;
using ::mlir::sdy::kNullDim;
using ::mlir::sdy::OpShardingRuleBuilder;

struct CopyShardingRuleOpInterface
    : public mlir::sdy::ShardingRuleOpInterface::ExternalModel<
          CopyShardingRuleOpInterface, mhlo::CopyOp> {
  mlir::sdy::OpShardingRuleAttr getShardingRule(mlir::Operation* op) const {
    return OpShardingRuleBuilder::buildPointwise(op);
  }
};

enum RaggedDotMode {
  // Ragged non-contracting (m): [b,m,k], [g,b,k,n], [b,g] -> [b,m,n].
  kNonContracting,
  // Ragged contracting (k):     [b,m,k], [b,k,n],   [b,g] -> [g,b,m,n].
  kContracting,
  // Ragged batch (b):           [b,m,k], [b,k,n],   [g]   -> [b,m,n].
  kBatch,
};

struct RaggedDotShardingRuleOpInterface
    : public mlir::sdy::ShardingRuleOpInterface::ExternalModel<
          RaggedDotShardingRuleOpInterface, mhlo::RaggedDotOp> {
  mlir::sdy::OpShardingRuleAttr getShardingRule(mlir::Operation* op) const {
    mhlo::RaggedDotOp raggedDot = llvm::cast<mhlo::RaggedDotOp>(op);
    mhlo::RaggedDotDimensionNumbersAttr raggedDotDimNumbers =
        raggedDot.getRaggedDotDimensionNumbers();
    mhlo::DotDimensionNumbersAttr dotDimNumbers =
        raggedDotDimNumbers.getDotDimensionNumbers();

    ArrayRef<int64_t> lhsBatchingDims =
        dotDimNumbers.getLhsBatchingDimensions();
    ArrayRef<int64_t> rhsBatchingDims =
        dotDimNumbers.getRhsBatchingDimensions();
    ArrayRef<int64_t> lhsContractingDims =
        dotDimNumbers.getLhsContractingDimensions();
    ArrayRef<int64_t> rhsContractingDims =
        dotDimNumbers.getRhsContractingDimensions();

    ArrayRef<int64_t> lhsRaggedDims =
        raggedDotDimNumbers.getLhsRaggedDimensions();
    CHECK_EQ(lhsRaggedDims.size(), 1);
    int64_t lhsRaggedDim = lhsRaggedDims.front();
    ArrayRef<int64_t> rhsGroupDims =
        raggedDotDimNumbers.getRhsGroupDimensions();

    RaggedDotMode mode;
    if (llvm::is_contained(lhsContractingDims, lhsRaggedDim)) {
      CHECK(rhsGroupDims.empty());
      mode = RaggedDotMode::kContracting;
    } else if (llvm::is_contained(lhsBatchingDims, lhsRaggedDim)) {
      CHECK(rhsGroupDims.empty());
      mode = RaggedDotMode::kBatch;
    } else {
      CHECK_EQ(rhsGroupDims.size(), 1);
      mode = RaggedDotMode::kNonContracting;
    }

    OpShardingRuleBuilder builder(raggedDot);

    mlir::RankedTensorType lhsType = raggedDot.getLhs().getType();
    mlir::RankedTensorType rhsType = raggedDot.getRhs().getType();
    mlir::RankedTensorType groupSizesType = raggedDot.getGroupSizes().getType();

    int64_t groupSizesDim = 0;
    int64_t outputDim = (mode == RaggedDotMode::kContracting) ? 1 : 0;

    // batching dimensions
    for (auto [lhsDim, rhsDim] :
         llvm::zip_equal(lhsBatchingDims, rhsBatchingDims)) {
      builder.addFactor(
          {lhsDim, rhsDim,
           mode != RaggedDotMode::kBatch ? groupSizesDim++ : kNullDim},
          outputDim++, lhsType.getDimSize(lhsDim));
    }

    // lhs non-contracting dimensions
    bool addLhsNonContractingDimsInGroupSizes =
        mode == RaggedDotMode::kNonContracting;
    for (int64_t i = 0; i < lhsType.getRank(); i++) {
      if (!llvm::is_contained(lhsContractingDims, i) &&
          !llvm::is_contained(lhsBatchingDims, i)) {
        FactorType factorType = FactorType::kPassThrough;
        if (i == lhsRaggedDim) {
          // We only add the non-contracting dimensions before the lhs ragged
          // dimension to the group sizes in the kNonContracting mode.
          addLhsNonContractingDimsInGroupSizes = false;
          factorType = FactorType::kNeedReplication;
        }
        builder.addFactor(
            {i, kNullDim,
             addLhsNonContractingDimsInGroupSizes ? groupSizesDim++ : kNullDim},
            outputDim++, lhsType.getDimSize(i), factorType);
      }
    }

    // rhs non-contracting dimensions
    for (int64_t i = 0; i < rhsType.getRank(); i++) {
      if (!llvm::is_contained(rhsContractingDims, i) &&
          !llvm::is_contained(rhsBatchingDims, i) &&
          !llvm::is_contained(rhsGroupDims, i)) {
        builder.addFactor({kNullDim, i, kNullDim}, outputDim++,
                          rhsType.getDimSize(i));
      }
    }

    // contracting dimensions
    bool addContractingDimsInGroupSizes = mode == RaggedDotMode::kContracting;
    for (auto [lhsDim, rhsDim] :
         llvm::zip_equal(lhsContractingDims, rhsContractingDims)) {
      FactorType factorType = FactorType::kReduction;
      if (lhsDim == lhsRaggedDim) {
        // We only add the contracting dimensions before the lhs ragged
        // dimension to the group sizes in the kContracting mode.
        addContractingDimsInGroupSizes = false;
        factorType = FactorType::kNeedReplication;
      }
      builder.addFactor(
          {lhsDim, rhsDim,
           addContractingDimsInGroupSizes ? groupSizesDim++ : kNullDim},
          kNullDim, lhsType.getDimSize(lhsDim), factorType);
    }

    switch (mode) {
      case RaggedDotMode::kNonContracting: {
        CHECK_EQ(groupSizesDim, groupSizesType.getRank() - 1);
        int64_t rhsGroupDim = rhsGroupDims.front();
        builder.addFactor({kNullDim, rhsGroupDim, groupSizesDim}, kNullDim,
                          rhsType.getDimSize(rhsGroupDim),
                          FactorType::kNeedReplication);
        break;
      }
      case RaggedDotMode::kContracting: {
        CHECK_EQ(groupSizesDim, groupSizesType.getRank() - 1);
        builder.addFactor({kNullDim, kNullDim, groupSizesDim}, 0,
                          groupSizesType.getDimSize(groupSizesDim),
                          FactorType::kNeedReplication);
        break;
      }
      case RaggedDotMode::kBatch: {
        break;
      }
    }

    return builder.build();
  }
};

// Sharding rule for `mhlo.scaled_dot` (block/group-scaled dot).
//
// Operand order is (lhs, rhs, lhs_scale, rhs_scale). The dot part behaves like
// `dot_general`; the two scale operands share the dim order of their data
// operand (the HLO verifier guarantees scale rank == data rank, and each data
// dim is an integer multiple of the matching scale dim).
//
// For every dot factor covering a (lhsDim, rhsDim) pair we emit a shared
// "group" factor across {lhs, rhs, matching real scale dims, result}. When a
// scale block-scales the dim (its extent is a proper divisor of the data dim),
// the group factor is sized to the block count (= scale extent) and an extra
// data-only "block" factor of size data/blockCount is added and marked
// kNeedReplication + isBlocked, so Shardy only ever shards that dim on
// block-count boundaries and never splits a scale block across devices. A
// scale extent of 1 along a dim is a broadcast (per-tensor along it): the data
// dim can shard freely while the scale stays replicated.
struct ScaledDotShardingRuleOpInterface
    : public mlir::sdy::ShardingRuleOpInterface::ExternalModel<
          ScaledDotShardingRuleOpInterface, mhlo::ScaledDotOp> {
  mlir::sdy::OpShardingRuleAttr getShardingRule(mlir::Operation* op) const {
    mhlo::ScaledDotOp scaledDot = llvm::cast<mhlo::ScaledDotOp>(op);
    mhlo::DotDimensionNumbersAttr dimNumbers =
        scaledDot.getDotDimensionNumbers();
    ArrayRef<int64_t> lhsBatchingDims = dimNumbers.getLhsBatchingDimensions();
    ArrayRef<int64_t> rhsBatchingDims = dimNumbers.getRhsBatchingDimensions();
    ArrayRef<int64_t> lhsContractingDims =
        dimNumbers.getLhsContractingDimensions();
    ArrayRef<int64_t> rhsContractingDims =
        dimNumbers.getRhsContractingDimensions();

    mlir::RankedTensorType lhsType = scaledDot.getLhs().getType();
    mlir::RankedTensorType rhsType = scaledDot.getRhs().getType();
    auto lhsScaleType = llvm::dyn_cast<mlir::RankedTensorType>(
        scaledDot.getLhsScale().getType());
    auto rhsScaleType = llvm::dyn_cast<mlir::RankedTensorType>(
        scaledDot.getRhsScale().getType());

    const int64_t lhsRank = lhsType.getRank();
    const int64_t rhsRank = rhsType.getRank();

    OpShardingRuleBuilder builder(scaledDot);

    // NVFP4 carries two trailing per-tensor scalar globals (operands 4,5) that
    // are rank-0 and never participate in any factor (fully replicated).
    // addFactor's operand array is zip_equal'd against the operand mappings, so
    // it must cover every operand: pad with kNullDim when the globals are
    // present.
    const bool hasGlobals = !scaledDot.getGlobalScales().empty();
    auto operandDims = [&](int64_t lhsDim, int64_t rhsDim, int64_t lhsScaleDim,
                           int64_t rhsScaleDim) -> llvm::SmallVector<int64_t> {
      llvm::SmallVector<int64_t> dims{lhsDim, rhsDim, lhsScaleDim, rhsScaleDim};
      if (hasGlobals) {
        dims.push_back(kNullDim);
        dims.push_back(kNullDim);
      }
      return dims;
    };

    // Emits the shared group factor (and, for a genuinely blocked dim, a
    // data-only block factor) for a single dot factor. `reduce` marks a
    // contracting factor (kReduction, absent from the result).
    auto addScaledFactor = [&](int64_t lhsDim, int64_t rhsDim, int64_t outDim,
                               int64_t dataSize, bool reduce) {
      int64_t groupSize = dataSize;
      int64_t lhsScaleDim = kNullDim;
      int64_t rhsScaleDim = kNullDim;
      // A scale shares this factor only when it has real structure on the
      // matching dim (1:1 when scaleSize == dataSize, or a genuine block when
      // 1 < scaleSize and scaleSize divides dataSize).
      auto consider = [&](mlir::RankedTensorType scaleType, int64_t dim,
                          int64_t& scaleSlot) {
        if (!scaleType || dim == kNullDim || dim >= scaleType.getRank()) {
          return;
        }
        int64_t s = scaleType.getDimSize(dim);
        if (s > 1 && dataSize % s == 0) {
          // Both operands may block the same factor at different granularity
          // (e.g. lhs per 32 elements, rhs per 16). The shared factor is then
          // the coarsest split both agree on -- their gcd -- and each operand
          // gets its own finer factor below. Taking either block count alone
          // would over-shard the other operand.
          groupSize = std::gcd(groupSize, s);
          scaleSlot = dim;
        }
      };
      // lhs_scale shares lhs's dim index; rhs_scale shares rhs's dim index.
      consider(lhsScaleType, lhsDim, lhsScaleDim);
      consider(rhsScaleType, rhsDim, rhsScaleDim);

      FactorType groupType =
          reduce ? FactorType::kReduction : FactorType::kPassThrough;
      int64_t resultDim = reduce ? kNullDim : outDim;

      // Shared group factor, at the granularity both scales agree on.
      builder.addFactor(operandDims(lhsDim, rhsDim, lhsScaleDim, rhsScaleDim),
                        resultDim, groupSize, groupType);

      // Per-operand refinement: a scale that blocks this dim more finely than
      // the shared factor keeps the remainder as its own never-sharded factor,
      // so a shard can never fall inside one of its blocks. Derived from the
      // operand, never hardcoded.
      auto addScaleRefinement = [&](mlir::RankedTensorType scaleType,
                                    int64_t dim, bool isLhs) {
        if (!scaleType || dim == kNullDim || dim >= scaleType.getRank()) return;
        const int64_t s = scaleType.getDimSize(dim);
        if (s <= groupSize || s % groupSize != 0) return;
        builder.addFactor(
            operandDims(isLhs ? lhsDim : kNullDim, isLhs ? kNullDim : rhsDim,
                        isLhs ? dim : kNullDim, isLhs ? kNullDim : dim),
            resultDim, s / groupSize, FactorType::kNeedReplication,
            /*isBlocked=*/true);
      };
      addScaleRefinement(lhsScaleType, lhsScaleDim, /*isLhs=*/true);
      addScaleRefinement(rhsScaleType, rhsScaleDim, /*isLhs=*/false);

      // Data-only block / minor factor: never shard across a scale block.
      if (dataSize > groupSize) {
        builder.addFactor(operandDims(lhsDim, rhsDim, kNullDim, kNullDim),
                          resultDim, dataSize / groupSize,
                          FactorType::kNeedReplication,
                          /*isBlocked=*/true);
      }
    };

    int64_t outputDim = 0;
    // Batch dimensions.
    for (auto [lhsDim, rhsDim] :
         llvm::zip_equal(lhsBatchingDims, rhsBatchingDims)) {
      addScaledFactor(lhsDim, rhsDim, outputDim++, lhsType.getDimSize(lhsDim),
                      /*reduce=*/false);
    }
    // LHS non-contracting dimensions.
    for (int64_t i = 0; i < lhsRank; ++i) {
      if (!llvm::is_contained(lhsContractingDims, i) &&
          !llvm::is_contained(lhsBatchingDims, i)) {
        addScaledFactor(/*lhsDim=*/i, /*rhsDim=*/kNullDim, outputDim++,
                        lhsType.getDimSize(i), /*reduce=*/false);
      }
    }
    // RHS non-contracting dimensions.
    for (int64_t i = 0; i < rhsRank; ++i) {
      if (!llvm::is_contained(rhsContractingDims, i) &&
          !llvm::is_contained(rhsBatchingDims, i)) {
        addScaledFactor(/*lhsDim=*/kNullDim, /*rhsDim=*/i, outputDim++,
                        rhsType.getDimSize(i), /*reduce=*/false);
      }
    }
    // Contracting dimensions.
    for (auto [lhsDim, rhsDim] :
         llvm::zip_equal(lhsContractingDims, rhsContractingDims)) {
      addScaledFactor(lhsDim, rhsDim, /*outDim=*/kNullDim,
                      lhsType.getDimSize(lhsDim), /*reduce=*/true);
    }


    return builder.build();
  }
};

struct TopKShardingRuleOpInterface
    : public mlir::sdy::ShardingRuleOpInterface::ExternalModel<
          TopKShardingRuleOpInterface, mhlo::TopKOp> {
  mlir::sdy::OpShardingRuleAttr getShardingRule(mlir::Operation* op) const {
    mhlo::TopKOp topK = llvm::cast<mhlo::TopKOp>(op);
    return OpShardingRuleBuilder(topK)
        .addPointwiseWithDiffTypeForMismatch(getTensorShape(topK.getOperand()),
                                             getTensorShape(topK.getResult(0)),
                                             FactorType::kNeedReplication,
                                             /*mismatchFactorIsBlocked=*/true)
        .build();
  }
};

// Sharding rule for `mhlo.scan`.
//
// `mhlo.scan` has variadic operands `(inputs..., inits...)` and variadic
// results `(outputs..., carries...)`. In the general case the body may
// produce per-iteration outputs of a different shape than its per-iteration
// inputs (the result types are inferred from the body terminator), so
// `outputs[i].shape` is not guaranteed to equal `inputs[i].shape` and the
// init/carry shapes need not relate to the input shapes at all.
//
// In practice (chlo.ScanOp from JAX, etc.) the body is shape-preserving:
//   * #inputs == #outputs == #inits == #carries,
//   * `inputs[i].shape == outputs[i].shape`,
//   * `inits[j].shape == inputs[j].shape` with the scan dim removed
//     (so `carries[j].shape` matches as well).
// In that "shape-preserving" pattern, all non-scan dimensions are
// independent across iterations and propagate point-wise between
// input/output/init/carry. The scan dimension's treatment depends on
// `is_associative`:
//   * associative: a parallel-prefix implementation exists (local scans per
//     shard plus inter-shard combine). The dimension is permutable
//     (`kPermutation`) so propagation may shard it at the cost of data
//     movement, mirroring how `stablehlo.reduce_window` handles window dims
//     with size > 1.
//   * non-associative or unspecified: the body has a true sequential
//     dependency along the scan dim, so it must be replicated
//     (`kNeedReplication`).
//
// For shape-changing scans, we fall back to a conservative rule that only
// constrains the scan dim and blocks propagation through every other dim.
struct ScanShardingRuleOpInterface
    : public mlir::sdy::ShardingRuleOpInterface::ExternalModel<
          ScanShardingRuleOpInterface, mhlo::ScanOp> {
  mlir::sdy::OpShardingRuleAttr getShardingRule(mlir::Operation* op) const {
    mhlo::ScanOp scan = llvm::cast<mhlo::ScanOp>(op);
    OpShardingRuleBuilder builder(scan);

    // The `mhlo.scan` verifier rejects the both-empty case
    // (`inputs.empty() && outputs.empty()`); defensively bail out with an
    // empty (no-factor) rule for malformed IR rather than crashing on
    // `.front()`.
    mlir::ValueRange inputs = scan.getInputs();
    mlir::ValueRange outputs = scan.getOutputs();
    if (inputs.empty() && outputs.empty()) {
      return builder.build();
    }
    mlir::ValueRange inits = scan.getInits();
    mlir::ValueRange carries = scan.getCarries();
    int64_t scanDim = scan.getDimension();
    int64_t numInputs = inputs.size();
    int64_t numInits = inits.size();
    bool isAssociative = scan.getIsAssociative().value_or(false);
    FactorType scanDimFactorType =
        isAssociative ? FactorType::kPermutation : FactorType::kNeedReplication;

    // Pick a representative input/output tensor to derive the rank and the
    // scan dim size. Bail out with an empty (no-factor) rule if the
    // representative is not a ranked tensor; in well-formed `mhlo.scan` IR
    // it always is.
    auto representative = llvm::dyn_cast<mlir::RankedTensorType>(
        (!inputs.empty() ? inputs.front() : outputs.front()).getType());
    if (!representative) {
      return builder.build();
    }
    int64_t rank = representative.getRank();
    int64_t scanDimSize = representative.getDimSize(scanDim);

    // Detect the "shape-preserving" pattern produced by chlo.ScanOp/JAX. In
    // that pattern the per-iteration body output has the same shape as the
    // per-iteration body input, so each input[i] has a corresponding output
    // of identical shape, and each init[j] has shape equal to input[j] with
    // the scan dim removed.
    auto isShapePreserving = [&]() {
      if (numInputs != outputs.size() || numInputs != numInits) {
        return false;
      }
      for (int64_t i = 0; i < numInputs; ++i) {
        auto inT = llvm::dyn_cast<mlir::RankedTensorType>(inputs[i].getType());
        auto outT =
            llvm::dyn_cast<mlir::RankedTensorType>(outputs[i].getType());
        if (!inT || !outT || inT.getShape() != outT.getShape()) {
          return false;
        }
      }
      for (int64_t j = 0; j < numInits; ++j) {
        auto initT = llvm::dyn_cast<mlir::RankedTensorType>(inits[j].getType());
        auto inT = llvm::dyn_cast<mlir::RankedTensorType>(inputs[j].getType());
        if (!initT || !inT || inT.getRank() != initT.getRank() + 1) {
          return false;
        }
        for (int64_t d = 0, m = 0; d < inT.getRank(); ++d) {
          if (d == scanDim) {
            continue;
          }
          if (inT.getDimSize(d) != initT.getDimSize(m++)) {
            return false;
          }
        }
      }
      return true;
    }();

    // Layout of the operand/result mapping vectors mirrors the variadic op:
    //   operandDims = [input_0, ..., input_{N-1}, init_0, ..., init_{M-1}]
    //   resultDims  = [output_0, ..., output_{N-1}, carry_0, ..., carry_{M-1}]
    int64_t numOperands = numInputs + numInits;
    int64_t numResults = outputs.size() + carries.size();

    if (!isShapePreserving) {
      // Conservative fallback: only model the scan dim factor (which we know
      // is shared across all inputs and outputs at position `scanDim`), and
      // leave every other dim unmapped. Unmapped dims are treated as needing
      // replication by Shardy, which is safe for arbitrary body shapes.
      mlir::SmallVector<int64_t> operandDims(numOperands, kNullDim);
      mlir::SmallVector<int64_t> resultDims(numResults, kNullDim);
      std::fill_n(operandDims.begin(), numInputs, scanDim);
      std::fill_n(resultDims.begin(), outputs.size(), scanDim);
      builder.addFactor(operandDims, resultDims, scanDimSize,
                        scanDimFactorType);
      return builder.build();
    }

    // Shape-preserving common case: one factor per input dim, shared across
    // input[i]/output[i] (and init[j]/carry[j] for the non-scan dims).
    mlir::SmallVector<int64_t> operandDims(numOperands, kNullDim);
    mlir::SmallVector<int64_t> resultDims(numResults, kNullDim);

    int64_t initDim = 0;
    for (int64_t inDim = 0; inDim < rank; ++inDim) {
      int64_t dimSize = representative.getDimSize(inDim);
      // Inputs and outputs always map to inDim.
      std::fill_n(operandDims.begin(), numInputs, inDim);
      std::fill_n(resultDims.begin(), numInputs, inDim);
      if (inDim == scanDim) {
        // Scan dim: not present on inits/carries.
        std::fill_n(operandDims.begin() + numInputs, numInits, kNullDim);
        std::fill_n(resultDims.begin() + numInputs, numInits, kNullDim);
        builder.addFactor(operandDims, resultDims, dimSize, scanDimFactorType);
      } else {
        // Non-scan dim: pass-through across inputs/outputs/inits/carries.
        std::fill_n(operandDims.begin() + numInputs, numInits, initDim);
        std::fill_n(resultDims.begin() + numInputs, numInits, initDim);
        builder.addFactor(operandDims, resultDims, dimSize);
        ++initDim;
      }
    }

    return builder.build();
  }
};

}  // namespace

void registerMhloExtensions(mlir::DialectRegistry& registry) {
  registry.addExtension(+[](mlir::MLIRContext* ctx, mhlo::MhloDialect*) {
    mhlo::CopyOp::attachInterface<CopyShardingRuleOpInterface>(*ctx);
    mhlo::RaggedDotOp::attachInterface<RaggedDotShardingRuleOpInterface>(*ctx);
    mhlo::ScaledDotOp::attachInterface<ScaledDotShardingRuleOpInterface>(*ctx);
    mhlo::ScanOp::attachInterface<ScanShardingRuleOpInterface>(*ctx);
    mhlo::TopKOp::attachInterface<TopKShardingRuleOpInterface>(*ctx);
  });
}

}  // namespace sdy
}  // namespace xla
