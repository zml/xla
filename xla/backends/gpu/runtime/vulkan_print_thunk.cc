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

#include "xla/backends/gpu/runtime/vulkan_print_thunk.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "xla/runtime/buffer_use.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/stream.h"
#include "xla/tsl/platform/errors.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {
namespace {

double DecodeElement(const uint8_t* data, int64_t index, PrimitiveType type) {
  auto read_u16 = [&]() {
    uint16_t value;
    std::memcpy(&value, data + 2 * index, sizeof(value));
    return value;
  };

  switch (type) {
    case F32: {
      float value;
      std::memcpy(&value, data + 4 * index, sizeof(value));
      return value;
    }
    case S32: {
      int32_t value;
      std::memcpy(&value, data + 4 * index, sizeof(value));
      return value;
    }
    case U32: {
      uint32_t value;
      std::memcpy(&value, data + 4 * index, sizeof(value));
      return value;
    }
    case S16: {
      int16_t value;
      std::memcpy(&value, data + 2 * index, sizeof(value));
      return value;
    }
    case U16:
      return read_u16();
    case S8:
      return static_cast<int8_t>(data[index]);
    case U8:
    case PRED:
      return data[index];
    case BF16: {
      uint32_t bits = static_cast<uint32_t>(read_u16()) << 16;
      float value;
      std::memcpy(&value, &bits, sizeof(value));
      return value;
    }
    case F16: {
      uint16_t half = read_u16();
      uint32_t sign = (half & 0x8000u) << 16;
      uint32_t exponent = (half >> 10) & 0x1f;
      uint32_t mantissa = half & 0x3ffu;
      uint32_t bits;
      if (exponent == 0) {
        if (mantissa == 0) {
          bits = sign;
        } else {
          exponent = 127 - 15 + 1;
          while ((mantissa & 0x400u) == 0) {
            mantissa <<= 1;
            --exponent;
          }
          mantissa &= 0x3ffu;
          bits = sign | (exponent << 23) | (mantissa << 13);
        }
      } else if (exponent == 0x1f) {
        bits = sign | 0x7f800000u | (mantissa << 13);
      } else {
        bits = sign | ((exponent - 15 + 127) << 23) | (mantissa << 13);
      }
      float value;
      std::memcpy(&value, &bits, sizeof(value));
      return value;
    }
    default:
      return 0.0;
  }
}

}  // namespace

VulkanPrintThunk::VulkanPrintThunk(ThunkInfo thunk_info, std::string label,
                                   BufferAllocation::Slice operand,
                                   Shape operand_shape)
    : Thunk(Kind::kCustomCall, std::move(thunk_info)),
      label_(std::move(label)),
      operand_(operand),
      operand_shape_(std::move(operand_shape)) {}

absl::Status VulkanPrintThunk::ExecuteOnStream(
    const ExecuteParams& /*params*/) {
  static_cast<void>(&DecodeElement);
  return absl::OkStatus();
}

Thunk::BufferUses VulkanPrintThunk::buffer_uses() const {
  return {BufferUse::Read(operand_, operand_shape_)};
}

absl::StatusOr<ThunkProto> VulkanPrintThunk::ToProto() const {
  return absl::UnimplementedError(
      "VulkanPrintThunk does not support serialization.");
}

}  // namespace xla::gpu
