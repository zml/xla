/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/service/gpu/fly_call.h"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "mlir/AsmParser/AsmParser.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "xla/tsl/platform/status_macros.h"

namespace xla::gpu {
namespace {

absl::StatusOr<int64_t> GetInteger(mlir::DictionaryAttr attrs,
                                   absl::string_view name) {
  auto attr = attrs.getAs<mlir::IntegerAttr>(name);
  if (!attr) {
    return absl::InvalidArgumentError(
        absl::StrCat("Fly custom call requires integer attribute '", name,
                     "'"));
  }
  return attr.getValue().getSExtValue();
}

}  // namespace

absl::StatusOr<FlyCall> FlyCall::Parse(absl::string_view backend_config,
                                       mlir::MLIRContext* mlir_context) {
  mlir::Attribute parsed = mlir::parseAttribute(backend_config, mlir_context);
  auto attrs = mlir::dyn_cast_if_present<mlir::DictionaryAttr>(parsed);
  if (!attrs) {
    return absl::InvalidArgumentError(
        "Fly custom call backend config must be an MLIR dictionary");
  }
  auto name = attrs.getAs<mlir::StringAttr>("name");
  auto ir = attrs.getAs<mlir::StringAttr>("ir");
  if (!name || !ir) {
    return absl::InvalidArgumentError(
        "Fly custom call requires string attributes 'name' and 'ir'");
  }

  TF_ASSIGN_OR_RETURN(int64_t num_warps, GetInteger(attrs, "num_warps"));
  TF_ASSIGN_OR_RETURN(int64_t grid_x, GetInteger(attrs, "grid_x"));
  TF_ASSIGN_OR_RETURN(int64_t grid_y, GetInteger(attrs, "grid_y"));
  TF_ASSIGN_OR_RETURN(int64_t grid_z, GetInteger(attrs, "grid_z"));
  if (num_warps <= 0 || grid_x <= 0 || grid_y <= 0 || grid_z <= 0) {
    return absl::InvalidArgumentError(
        "Fly custom call launch dimensions must be positive");
  }

  int32_t waves_per_eu = 0;
  if (auto attr = attrs.getAs<mlir::IntegerAttr>("waves_per_eu")) {
    waves_per_eu = static_cast<int32_t>(attr.getValue().getSExtValue());
    if (waves_per_eu < 0) {
      return absl::InvalidArgumentError(
          "Fly custom call waves_per_eu must be non-negative");
    }
  }

  int64_t shared_mem_bytes = 0;
  if (auto attr = attrs.getAs<mlir::IntegerAttr>("shared_mem_bytes")) {
    shared_mem_bytes = attr.getValue().getSExtValue();
    if (shared_mem_bytes < 0) {
      return absl::InvalidArgumentError(
          "Fly custom call shared_mem_bytes must be non-negative");
    }
  }

  std::vector<int64_t> zeroed_outputs;
  if (auto attr = attrs.getAs<mlir::ArrayAttr>("zeroed_outputs")) {
    zeroed_outputs.reserve(attr.size());
    for (mlir::Attribute value : attr) {
      auto integer = mlir::dyn_cast<mlir::IntegerAttr>(value);
      if (!integer) {
        return absl::InvalidArgumentError(
            "Fly custom call zeroed_outputs must contain integers");
      }
      zeroed_outputs.push_back(integer.getValue().getSExtValue());
    }
  }

  return FlyCall{name.getValue().str(),
                 ir.getValue().str(),
                 num_warps,
                 static_cast<int32_t>(grid_x),
                 static_cast<int32_t>(grid_y),
                 static_cast<int32_t>(grid_z),
                 waves_per_eu,
                 shared_mem_bytes,
                 std::move(zeroed_outputs)};
}

}  // namespace xla::gpu
