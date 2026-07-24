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

#ifndef XLA_STREAM_EXECUTOR_MUSA_MUBLAS_SHIM_MUBLAS_SHIM_FLOAT16_H_
#define XLA_STREAM_EXECUTOR_MUSA_MUBLAS_SHIM_MUBLAS_SHIM_FLOAT16_H_

#include <cstdint>
#include <cstring>

namespace stream_executor::musa::mublas_shim_internal {

// Converts IEEE binary32 to IEEE binary16 using round-to-nearest, ties-to-even.
// NaN payloads are truncated but remain nonzero.
inline uint16_t Float32ToFloat16Bits(float value) {
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));

  const uint32_t sign = (bits >> 16) & 0x8000u;
  const uint32_t exponent = (bits >> 23) & 0xffu;
  const uint32_t fraction = bits & 0x007fffffu;

  if (exponent == 0xffu) {
    if (fraction == 0) return static_cast<uint16_t>(sign | 0x7c00u);
    uint32_t payload = fraction >> 13;
    if (payload == 0) payload = 1;
    return static_cast<uint16_t>(sign | 0x7c00u | payload);
  }

  int32_t half_exponent = static_cast<int32_t>(exponent) - 127 + 15;
  if (half_exponent <= 0) {
    if (half_exponent < -10) return static_cast<uint16_t>(sign);

    const uint32_t significand = fraction | 0x00800000u;
    const uint32_t shift = static_cast<uint32_t>(14 - half_exponent);
    uint32_t half_fraction = significand >> shift;
    const uint32_t remainder = significand & ((uint32_t{1} << shift) - 1);
    const uint32_t halfway = uint32_t{1} << (shift - 1);
    if (remainder > halfway ||
        (remainder == halfway && (half_fraction & 1u) != 0)) {
      ++half_fraction;
    }
    return static_cast<uint16_t>(sign | half_fraction);
  }

  if (half_exponent >= 31) {
    return static_cast<uint16_t>(sign | 0x7c00u);
  }

  uint32_t half_fraction = fraction >> 13;
  const uint32_t remainder = fraction & 0x1fffu;
  if (remainder > 0x1000u ||
      (remainder == 0x1000u && (half_fraction & 1u) != 0)) {
    ++half_fraction;
    if (half_fraction == 0x400u) {
      half_fraction = 0;
      ++half_exponent;
      if (half_exponent == 31) {
        return static_cast<uint16_t>(sign | 0x7c00u);
      }
    }
  }

  return static_cast<uint16_t>(
      sign | (static_cast<uint32_t>(half_exponent) << 10) | half_fraction);
}

}  // namespace stream_executor::musa::mublas_shim_internal

#endif  // XLA_STREAM_EXECUTOR_MUSA_MUBLAS_SHIM_MUBLAS_SHIM_FLOAT16_H_
