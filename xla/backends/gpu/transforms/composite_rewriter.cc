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

bool IsSupportedScaledOperand(const HloInstruction* operand,
                              const HloInstruction* scale) {
  const PrimitiveType op_type = operand->shape().element_type();
  const PrimitiveType scale_type = scale->shape().element_type();
  if (op_type == F8E4M3FN || op_type == F8E5M2 || op_type == F4E2M1FN) {
    if (scale_type != F8E8M0FNU && scale_type != F8E4M3FN &&
        scale_type != BF16 && scale_type != F32) {
      return false;
    }
    // ShapeVerifier::HandleScaledDot requires equal ranks; IsNoOpScale only
    // applies to BF16 operands, so a rank-0 scale on quantized data fails later.
    if (scale->shape().dimensions().size() !=
        operand->shape().dimensions().size()) {
      return false;
    }
    for (int64_t d = 0; d < operand->shape().dimensions().size(); ++d) {
      const int64_t o = operand->shape().dimensions(d);
      const int64_t s = scale->shape().dimensions(d);
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
      if (dim != 1) return false;
    }
    return scale->opcode() == HloOpcode::kConstant &&
           scale->literal().IsAllFloat(1.0);
  }
  return false;
}

// rhs_scale rank > rhs rank is the hardware SF block layout ([N/128, kg/4, 512]
// swizzle); compare ranks relative to the rhs, not a fixed rank-3, so batched
// dots (natural rank-3 scale) still work.
absl::Status RejectSwizzledRhsScale(HloCallInstruction* call) {
  const int64_t rhs_rank = call->operand(1)->shape().dimensions().size();
  const int64_t scale_rank = call->operand(3)->shape().dimensions().size();
  if (scale_rank <= rhs_rank) {
    return absl::OkStatus();
  }
  return absl::InvalidArgumentError(absl::StrCat(
      "xla.scaled_dot: rhs_scale ", call->operand(3)->shape().ToString(),
      " has more dimensions than rhs ", call->operand(1)->shape().ToString(),
      ", i.e. the hardware SF block layout. That layout is no longer supported: "
      "emit the natural [N, kg] scale instead."));
}

absl::StatusOr<bool> TryRewriteScaledDotComposite(HloComputation* computation,
                                                  HloCallInstruction* call) {
  if (!call->is_composite()) return false;
  if (!call->has_frontend_attributes()) {
    VLOG(3) << "No frontend attributes";
    return false;
  }
  const auto& frontend_attrs = call->frontend_attributes().map();
  constexpr char kNameKey[] = "composite.name";
  auto name = frontend_attrs.find(kNameKey);
  if (name == frontend_attrs.end()) {
    VLOG(3) << kNameKey << " is not set";
    return false;
  }
  if (name->second != "xla.scaled_dot") {
    VLOG(3) << kNameKey << " is not xla.scaled_dot: " << name->second;
    return false;
  }
  if (!frontend_attrs.contains("composite.attributes")) {
    return absl::InvalidArgumentError(
        "composite.attributes is not set for xla.scaled_dot");
  }
  if (call->operand_count() != HloScaledDotInstruction::kOperands) {
    return absl::InvalidArgumentError(
        absl::StrCat("xla.scaled_dot composite expects exactly ",
                     HloScaledDotInstruction::kOperands, " operands, got ",
                     call->operand_count()));
  }

  RETURN_IF_ERROR(RejectSwizzledRhsScale(call));

  ASSIGN_OR_RETURN(
      DotDimensionNumbers dnums,
      ParseDimensionNumbers(frontend_attrs.at("composite.attributes")));
  RETURN_IF_ERROR(ValidateDotShape(*call, dnums));

  if (dnums.lhs_contracting_dimensions_size() != 1 ||
      dnums.rhs_contracting_dimensions_size() != 1 ||
      dnums.lhs_batch_dimensions_size() > 1 ||
      dnums.rhs_batch_dimensions_size() > 1) {
    LOG(ERROR) << "Unsupported dimension numbers: " << dnums.DebugString();
    return false;
  }

  // Both scales scalar is malformed (not merely unsupported).
  if (ShapeUtil::IsScalar(call->operand(2)->shape()) &&
      ShapeUtil::IsScalar(call->operand(3)->shape())) {
    return absl::InvalidArgumentError(
        "xla.scaled_dot requires at least one non-scalar scale");
  }
  if (!IsSupportedScaledOperand(call->operand(0), call->operand(2)) ||
      !IsSupportedScaledOperand(call->operand(1), call->operand(3))) {
    return false;
  }

  PrecisionConfig precision{};
  precision.mutable_operand_precision()->Resize(2, PrecisionConfig::DEFAULT);
  auto* scaled_dot =
      computation->AddInstruction(HloInstruction::CreateScaledDot(
          call->shape(), call->mutable_operand(0), call->mutable_operand(1),
          call->mutable_operand(2), call->mutable_operand(3), dnums,
          precision));
  call->SetupDerivedInstruction(scaled_dot);
  TF_ASSIGN_OR_RETURN(
      bool replaced,
      computation->ReplaceInstruction(call, scaled_dot,
                                      /*preserve_sharding=*/true,
                                      /*relay_control_dependency=*/true));
  if (!replaced) {
    return absl::InternalError(
        "failed to replace xla.scaled_dot composite call");
  }
  return true;
}

}  // namespace

absl::StatusOr<bool> CompositeRewriter::RewriteComputation(
    HloComputation* computation) {
  bool changed = false;
  for (HloInstruction* instruction : computation->MakeInstructionPostOrder()) {
    if (instruction->opcode() != HloOpcode::kCall) continue;
    ASSIGN_OR_RETURN(
        bool rewritten,
        TryRewriteScaledDotComposite(computation,
                                     Cast<HloCallInstruction>(instruction)));
    changed |= rewritten;
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
