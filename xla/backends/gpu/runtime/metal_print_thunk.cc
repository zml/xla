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

#include "xla/backends/gpu/runtime/metal_print_thunk.h"

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

namespace xla {
namespace gpu {

namespace se = ::stream_executor;

namespace {
// Decode element `i` of a raw host buffer of primitive type `t` to a double for
// printing. Floats widen to double; ints to their signed/unsigned value. bf16 is
// bits<<16 reinterpreted; f16 is a manual IEEE binary16 -> float.
double DecodeElement(const uint8_t* data, int64_t i, PrimitiveType t) {
  auto u16 = [&]() { uint16_t v; std::memcpy(&v, data + 2 * i, 2); return v; };
  switch (t) {
    case F32: { float v; std::memcpy(&v, data + 4 * i, 4); return v; }
    case S32: { int32_t v; std::memcpy(&v, data + 4 * i, 4); return v; }
    case U32: { uint32_t v; std::memcpy(&v, data + 4 * i, 4); return v; }
    case S16: { int16_t v; std::memcpy(&v, data + 2 * i, 2); return v; }
    case U16: return u16();
    case S8:  return static_cast<int8_t>(data[i]);
    case U8: case PRED: return data[i];
    case BF16: {
      uint32_t bits = static_cast<uint32_t>(u16()) << 16;
      float f; std::memcpy(&f, &bits, 4); return f;
    }
    case F16: {
      uint16_t h = u16();
      uint32_t sign = (h & 0x8000u) << 16;
      uint32_t exp = (h >> 10) & 0x1f, man = h & 0x3ffu, bits;
      if (exp == 0) {
        if (man == 0) { bits = sign; }
        else {  // subnormal -> normalize
          exp = 127 - 15 + 1;
          while ((man & 0x400u) == 0) { man <<= 1; --exp; }
          man &= 0x3ffu;
          bits = sign | (exp << 23) | (man << 13);
        }
      } else if (exp == 0x1f) {
        bits = sign | 0x7f800000u | (man << 13);
      } else {
        bits = sign | ((exp - 15 + 127) << 23) | (man << 13);
      }
      float f; std::memcpy(&f, &bits, 4); return f;
    }
    default: return 0.0;
  }
}
}  // namespace

MetalPrintThunk::MetalPrintThunk(ThunkInfo thunk_info, std::string label,
                                 BufferAllocation::Slice operand,
                                 Shape operand_shape)
    : Thunk(Kind::kCustomCall, std::move(thunk_info)),
      label_(std::move(label)),
      operand_(operand),
      operand_shape_(std::move(operand_shape)) {}

absl::Status MetalPrintThunk::ExecuteOnStream(const ExecuteParams& params) {
  se::Stream* stream = params.stream;
  se::DeviceAddressBase src =
      params.buffer_allocations->GetDeviceAddress(operand_);

  const PrimitiveType t = operand_shape_.element_type();
  const int64_t n = ShapeUtil::ElementsIn(operand_shape_);
  const uint64_t bytes = src.size();
  std::vector<uint8_t> host(bytes);
  TF_RETURN_IF_ERROR(stream->Memcpy(host.data(), src, bytes));
  TF_RETURN_IF_ERROR(stream->BlockHostUntilDone());

  const int64_t kMax = 32;  // how many leading values to print
  std::string vals;
  double mn = 0, mx = 0;
  int64_t nz = 0;
  for (int64_t i = 0; i < n; ++i) {
    double v = DecodeElement(host.data(), i, t);
    if (i == 0 || v < mn) mn = v;
    if (i == 0 || v > mx) mx = v;
    if (v != 0.0) ++nz;
    if (i < kMax) absl::StrAppend(&vals, i ? ", " : "", absl::StrCat(v));
  }
  fprintf(stderr,
          "ZMLPRINT[%s] %s n=%lld first%lld=[%s] min=%g max=%g nonzero=%lld/%lld\n",
          label_.c_str(), operand_shape_.ToString().c_str(),
          static_cast<long long>(n), static_cast<long long>(kMax), vals.c_str(),
          mn, mx, static_cast<long long>(nz), static_cast<long long>(n));
  fflush(stderr);
  return absl::OkStatus();
}

Thunk::BufferUses MetalPrintThunk::buffer_uses() const {
  return {BufferUse::Read(operand_, operand_shape_)};
}

absl::StatusOr<ThunkProto> MetalPrintThunk::ToProto() const {
  return absl::UnimplementedError(
      "MetalPrintThunk does not support serialization.");
}

}  // namespace gpu
}  // namespace xla
