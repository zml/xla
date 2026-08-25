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

#include "xla/backends/gpu/codegen/flydsl/scan_support.h"

#include <cstdint>
#include <limits>
#include <optional>

#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/shape_util.h"

namespace xla::gpu::flydsl {
namespace {

constexpr int64_t kMaxNativeRowLength = 1024;

bool IsSupportedElementType(PrimitiveType type) {
  return type == F16 || type == BF16 || type == F32 || type == S32 ||
         type == U32;
}

int64_t ElementBytes(PrimitiveType type) {
  return type == F16 || type == BF16 ? 2 : 4;
}

}  // namespace

bool IsFlyScanSupported(const Shape& shape, const CubScanOptions& options) {
  if (!shape.IsArray() || !shape.has_layout() ||
      !IsSupportedElementType(shape.element_type()) ||
      options.kind() != CubScanOptions::SUM || options.vector_length() != 1 ||
      options.row_length() <= 0 ||
      options.row_length() > kMaxNativeRowLength ||
      options.column_length() <= 0 ||
      ShapeUtil::ElementsIn(shape) !=
          options.row_length() * options.column_length()) {
    return false;
  }
  const int64_t elements = ShapeUtil::ElementsIn(shape);
  return elements <= std::numeric_limits<uint32_t>::max() /
                         ElementBytes(shape.element_type());
}

std::optional<FlyScanDescriptor> GetFlyScanDescriptor(
    const HloInstruction& call) {
  if (call.opcode() != HloOpcode::kCustomCall ||
      call.custom_call_target() != kFlyScanCallTarget ||
      call.operand_count() != 1 || !call.shape().IsArray() ||
      call.operand(0)->shape() != call.shape()) {
    return std::nullopt;
  }
  auto options = call.backend_config<CubScanOptions>();
  if (!options.ok() || !IsFlyScanSupported(call.shape(), *options)) {
    return std::nullopt;
  }
  return FlyScanDescriptor{
      options->row_length(), options->column_length(), options->is_reverse(),
      call.shape().element_type(), call.operand(0), &call};
}

std::optional<FlyScanDescriptor> GetFlyScanDescriptor(
    const HloFusionAnalysis& analysis) {
  if (analysis.fusion_root_count() != 1) {
    return std::nullopt;
  }
  return GetFlyScanDescriptor(analysis.fusion_root(0).instruction());
}

}  // namespace xla::gpu::flydsl
