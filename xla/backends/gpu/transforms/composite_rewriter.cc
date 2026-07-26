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

#include "xla/backends/gpu/transforms/composite_rewriter.h"

#include <cstdint>

#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/platform/status_macros.h"
#include "mlir/AsmParser/AsmParser.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/LLVM.h"
#include "xla/backends/gpu/transforms/nvfp4_scale_swizzle.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/literal.h"
#include "xla/service/shape_inference.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {

namespace {

absl::StatusOr<DotDimensionNumbers> ParseDimensionNumbers(
    absl::string_view composite_attributes) {
  mlir::MLIRContext context;
  mlir::Attribute attr = mlir::parseAttribute(composite_attributes, &context);
  mlir::DictionaryAttr dict_attrs = mlir::dyn_cast<mlir::DictionaryAttr>(attr);
  if (!dict_attrs) {
    return absl::InvalidArgumentError(
        "composite.attributes must be an MLIR dictionary attribute");
  }
  if (!dict_attrs.contains("dimension_numbers")) {
    return absl::InvalidArgumentError(
        "dimension_numbers are not set in composite attributes");
  }

  mlir::ArrayAttr dim_numbers =
      mlir::dyn_cast<mlir::ArrayAttr>(dict_attrs.get("dimension_numbers"));
  if (!dim_numbers || dim_numbers.size() != 2) {
    return absl::InvalidArgumentError(
        "dimension_numbers must be array of size 2");
  }

  mlir::ArrayAttr contracting = mlir::dyn_cast<mlir::ArrayAttr>(dim_numbers[0]);
  mlir::ArrayAttr batch = mlir::dyn_cast<mlir::ArrayAttr>(dim_numbers[1]);
  if (!contracting || contracting.size() != 2 || !batch || batch.size() != 2) {
    return absl::InvalidArgumentError(
        "invalid contracting or batch dimensions");
  }

  mlir::ArrayAttr lhs_contracting =
      mlir::dyn_cast<mlir::ArrayAttr>(contracting[0]);
  mlir::ArrayAttr rhs_contracting =
      mlir::dyn_cast<mlir::ArrayAttr>(contracting[1]);
  mlir::ArrayAttr lhs_batch = mlir::dyn_cast<mlir::ArrayAttr>(batch[0]);
  mlir::ArrayAttr rhs_batch = mlir::dyn_cast<mlir::ArrayAttr>(batch[1]);

  if (!lhs_contracting || !rhs_contracting || !lhs_batch || !rhs_batch) {
    return absl::InvalidArgumentError("Invalid dimension_numbers structure");
  }

  DotDimensionNumbers dnums;
  for (mlir::Attribute dim : lhs_contracting) {
    mlir::IntegerAttr integer = mlir::dyn_cast<mlir::IntegerAttr>(dim);
    if (!integer) {
      return absl::InvalidArgumentError(
          "lhs contracting dimensions must contain only integers");
    }
    dnums.add_lhs_contracting_dimensions(integer.getInt());
  }
  for (mlir::Attribute dim : rhs_contracting) {
    mlir::IntegerAttr integer = mlir::dyn_cast<mlir::IntegerAttr>(dim);
    if (!integer) {
      return absl::InvalidArgumentError(
          "rhs contracting dimensions must contain only integers");
    }
    dnums.add_rhs_contracting_dimensions(integer.getInt());
  }
  for (mlir::Attribute dim : lhs_batch) {
    mlir::IntegerAttr integer = mlir::dyn_cast<mlir::IntegerAttr>(dim);
    if (!integer) {
      return absl::InvalidArgumentError(
          "lhs batch dimensions must contain only integers");
    }
    dnums.add_lhs_batch_dimensions(integer.getInt());
  }
  for (mlir::Attribute dim : rhs_batch) {
    mlir::IntegerAttr integer = mlir::dyn_cast<mlir::IntegerAttr>(dim);
    if (!integer) {
      return absl::InvalidArgumentError(
          "rhs batch dimensions must contain only integers");
    }
    dnums.add_rhs_batch_dimensions(integer.getInt());
  }
  return dnums;
}

absl::Status ValidateDotShape(const HloCallInstruction& call,
                              const DotDimensionNumbers& dnums) {
  if (!call.shape().IsArray()) {
    return absl::InvalidArgumentError(
        "xla.scaled_dot composite result must be an array");
  }

  absl::StatusOr<Shape> inferred = ShapeInference::InferDotOpShape(
      call.operand(0)->shape(), call.operand(1)->shape(), dnums,
      call.shape().element_type());
  if (!inferred.ok()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "invalid xla.scaled_dot dimension numbers or operand shapes: ",
        inferred.status().message()));
  }
  if (!ShapeUtil::Compatible(*inferred, call.shape())) {
    return absl::InvalidArgumentError(absl::StrCat(
        "xla.scaled_dot result shape ", call.shape().ToString(),
        " does not match the inferred dot shape ", inferred->ToString()));
  }
  return absl::OkStatus();
}

}  // namespace

absl::StatusOr<bool> CompositeRewriter::RewriteComputation(
    HloComputation* computation) {
  bool changed = false;
  for (HloInstruction* instruction : computation->MakeInstructionPostOrder()) {
    if (instruction->opcode() != HloOpcode::kCall) {
      continue;
    }
    auto call = Cast<HloCallInstruction>(instruction);
    if (!call->is_composite()) {
      continue;
    }
    if (!call->has_frontend_attributes()) {
      VLOG(3) << "No frontend attributes";
      continue;
    }
    const auto& frontend_attrs = call->frontend_attributes().map();
    constexpr char key[] = "composite.name";
    auto name = frontend_attrs.find(key);
    if (name == frontend_attrs.end()) {
      VLOG(3) << key << " is not set";
      continue;
    }
    if (name->second != "xla.scaled_dot") {
      VLOG(3) << key << " is not xla.scaled_dot: " << name->second;
      continue;
    }
    if (!frontend_attrs.contains("composite.attributes")) {
      return absl::InvalidArgumentError(
          "composite.attributes is not set for xla.scaled_dot");
    }
    const bool with_globals =
        call->operand_count() == HloScaledDotInstruction::kOperandsWithGlobals;
    if (call->operand_count() != HloScaledDotInstruction::kOperands &&
        !with_globals) {
      return absl::InvalidArgumentError(absl::StrCat(
          "xla.scaled_dot composite expects ",
          HloScaledDotInstruction::kOperands, " or ",
          HloScaledDotInstruction::kOperandsWithGlobals, " operands, got ",
          call->operand_count()));
    }
    // Operand 3 may arrive in either spelling. The blocked
    // [N/128, kg/4, 512] form is the hardware SF layout; resolve it here, at
    // the boundary, so the op itself is always defined on the natural [N, kg]
    // scale and no pass downstream needs to know the layout exists. A backend
    // arm that wants the swizzled bytes matches the chain back
    // (MatchNvfp4ScaleUnswizzle); everything else just sees reshape/transpose.
    if (call->operand(3)->shape().dimensions().size() == 3 &&
        call->operand(1)->shape().dimensions().size() == 2) {
      const Shape& rhs = call->operand(1)->shape();
      const int64_t n = rhs.dimensions(0);
      const Shape& sf = call->operand(3)->shape();
      const int64_t kg = sf.dimensions(1) * kSfGroupsPerBlock;
      if (rhs.dimensions(1) % kg != 0) {
        return absl::InvalidArgumentError(absl::StrCat(
            "xla.scaled_dot: blocked rhs_scale ", sf.ToString(),
            " does not divide the contracting dimension of ", rhs.ToString()));
      }
      HloInstruction* natural =
          EmitNvfp4ScaleUnswizzle(call->mutable_operand(3), n, kg);
      if (natural == nullptr) {
        return absl::InvalidArgumentError(absl::StrCat(
            "xla.scaled_dot: rhs_scale ", sf.ToString(),
            " is rank-3 but not a valid [N/128, kg/4, 512] block scale for ",
            rhs.ToString()));
      }
      RETURN_IF_ERROR(call->ReplaceOperandWithDifferentShape(3, natural));
    }

    ASSIGN_OR_RETURN(
        DotDimensionNumbers dot_dimension_numbers,
        ParseDimensionNumbers(frontend_attrs.at("composite.attributes")));

    RETURN_IF_ERROR(ValidateDotShape(*call, dot_dimension_numbers));

    if (dot_dimension_numbers.lhs_contracting_dimensions_size() != 1 ||
        dot_dimension_numbers.rhs_contracting_dimensions_size() != 1 ||
        dot_dimension_numbers.lhs_batch_dimensions_size() > 1 ||
        dot_dimension_numbers.rhs_batch_dimensions_size() > 1) {
      LOG(ERROR) << "Unsupported dimension numbers: "
                 << dot_dimension_numbers.DebugString();
      continue;
    }

    const HloInstruction* lhs = call->operand(0);
    const HloInstruction* rhs = call->operand(1);
    const HloInstruction* lhs_scale = call->operand(2);
    const HloInstruction* rhs_scale = call->operand(3);

    auto is_supported = [&](const HloInstruction* operand,
                            const HloInstruction* scale) {
      auto op_type = operand->shape().element_type();
      auto scale_type = scale->shape().element_type();
      if (op_type == F8E4M3FN || op_type == F8E5M2 || op_type == F4E2M1FN) {
        if (scale_type != F8E8M0FNU && scale_type != F8E4M3FN &&
            scale_type != BF16 && scale_type != F32) {
          return false;
        }

        // scalar
        if (scale->shape().dimensions().empty()) {
          return true;
        }

        if (scale->shape().dimensions().size() !=
            operand->shape().dimensions().size()) {
          return false;
        }

        // Every scale dimension must divide the matching operand dimension
        // (block-128, group-16 NVFP4, group-32 MX, per-channel, ...).
        for (int64_t d = 0; d < operand->shape().dimensions().size(); ++d) {
          int64_t o = operand->shape().dimensions(d);
          int64_t s = scale->shape().dimensions(d);
          if (s == 0 || o % s != 0) {
            return false;
          }
        }

        return true;
      }

      if (op_type == BF16 && scale_type == BF16) {
        if (scale->shape().dimensions().size() !=
            operand->shape().dimensions().size()) {
          return false;
        }
        for (int64_t dim : scale->shape().dimensions()) {
          if (dim != 1) {
            return false;
          }
        }
        if (scale->opcode() != HloOpcode::kConstant) {
          return false;
        }
        return scale->literal().IsAllFloat(1.0);
      }
      return false;
    };

    if (!is_supported(lhs, lhs_scale) || !is_supported(rhs, rhs_scale)) {
      continue;
    }
    if (ShapeUtil::IsScalar(lhs_scale->shape()) &&
        ShapeUtil::IsScalar(rhs_scale->shape())) {
      return absl::InvalidArgumentError(
          "xla.scaled_dot requires at least one non-scalar scale");
    }

    PrecisionConfig precision{};
    precision.mutable_operand_precision()->Resize(2, PrecisionConfig::DEFAULT);
    auto* scaled_dot =
        computation->AddInstruction(HloInstruction::CreateScaledDot(
            call->shape(), call->mutable_operand(0), call->mutable_operand(1),
            call->mutable_operand(2), call->mutable_operand(3),
            dot_dimension_numbers, precision,
            with_globals ? call->mutable_operand(4) : nullptr,
            with_globals ? call->mutable_operand(5) : nullptr));
    call->SetupDerivedInstruction(scaled_dot);
    TF_ASSIGN_OR_RETURN(
        bool replaced,
        computation->ReplaceInstruction(
            call, scaled_dot, /*preserve_sharding=*/true,
            /*relay_control_dependency=*/true));
    if (!replaced) {
      return absl::InternalError(
          "failed to replace xla.scaled_dot composite call");
    }
    changed = true;
  }
  return changed;
}

absl::StatusOr<bool> CompositeRewriter::RunImpl(
    HloModule* module, const absl::flat_hash_set<absl::string_view>&) {
  bool changed = false;
  for (HloComputation* computation : module->computations()) {
    ASSIGN_OR_RETURN(bool result, RewriteComputation(computation));
    changed |= result;
  }
  return changed;
}

}  // namespace gpu
}  // namespace xla
