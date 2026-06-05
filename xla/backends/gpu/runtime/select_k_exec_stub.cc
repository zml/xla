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

#include <cstdint>
#include <algorithm>
#include <numeric>
#include <vector>

#include "absl/status/status.h"
#include "xla/backends/gpu/runtime/select_k_exec.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/device_address_allocator.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/types.h"  // IWYU pragma: keep

namespace xla::gpu {
namespace se = ::stream_executor;

template <typename T>
absl::Status select_k_exec(int device_ordinal,
                           se::DeviceAddressAllocator* allocator,
                           se::Stream* stream, se::DeviceAddressBase data_in,
                           se::DeviceAddressBase data_out,
                           se::DeviceAddressBase indices_out,
                           std::uint32_t batch, std::uint32_t n,
                           std::uint32_t k) {
  (void)device_ordinal;
  (void)allocator;
  if (k > n) {
    return absl::InvalidArgumentError("select_k_exec requires k <= n");
  }

  const uint64_t input_bytes = sizeof(T) * batch * n;
  const uint64_t output_bytes = sizeof(T) * batch * k;
  const uint64_t indices_bytes = sizeof(std::uint32_t) * batch * k;

  auto input_alloc = stream->parent()->HostMemoryAllocate(input_bytes);
  if (!input_alloc.ok()) {
    return input_alloc.status();
  }
  auto output_alloc = stream->parent()->HostMemoryAllocate(output_bytes);
  if (!output_alloc.ok()) {
    return output_alloc.status();
  }
  auto indices_alloc = stream->parent()->HostMemoryAllocate(indices_bytes);
  if (!indices_alloc.ok()) {
    return indices_alloc.status();
  }

  T* input = reinterpret_cast<T*>((*input_alloc)->address().opaque());
  T* output = reinterpret_cast<T*>((*output_alloc)->address().opaque());
  std::uint32_t* indices =
      reinterpret_cast<std::uint32_t*>((*indices_alloc)->address().opaque());

  absl::Status status = stream->Memcpy(input, data_in, input_bytes);
  if (!status.ok()) {
    return status;
  }
  status = stream->BlockHostUntilDone();
  if (!status.ok()) {
    return status;
  }

  std::vector<std::uint32_t> order(n);
  for (std::uint32_t b = 0; b < batch; ++b) {
    std::iota(order.begin(), order.end(), 0);
    const T* batch_input = input + static_cast<uint64_t>(b) * n;
    auto better = [&](std::uint32_t lhs, std::uint32_t rhs) {
      float lhs_value = static_cast<float>(batch_input[lhs]);
      float rhs_value = static_cast<float>(batch_input[rhs]);
      if (lhs_value == rhs_value) {
        return lhs < rhs;
      }
      return lhs_value > rhs_value;
    };
    std::partial_sort(order.begin(), order.begin() + k, order.end(), better);

    for (std::uint32_t i = 0; i < k; ++i) {
      const uint64_t out_index = static_cast<uint64_t>(b) * k + i;
      output[out_index] = batch_input[order[i]];
      indices[out_index] = order[i];
    }
  }

  status = stream->Memcpy(&data_out, output, output_bytes);
  if (!status.ok()) {
    return status;
  }
  status = stream->Memcpy(&indices_out, indices, indices_bytes);
  if (!status.ok()) {
    return status;
  }
  return stream->BlockHostUntilDone();
}

// Explicit instantiations for supported dtypes.
template absl::Status select_k_exec<float>(int, se::DeviceAddressAllocator*,
                                           se::Stream*, se::DeviceAddressBase,
                                           se::DeviceAddressBase,
                                           se::DeviceAddressBase, std::uint32_t,
                                           std::uint32_t, std::uint32_t);

template absl::Status select_k_exec<::xla::bfloat16>(
    int, se::DeviceAddressAllocator*, se::Stream*, se::DeviceAddressBase,
    se::DeviceAddressBase, se::DeviceAddressBase, std::uint32_t, std::uint32_t,
    std::uint32_t);

template <typename T>
absl::Status select_k_payload_exec(
    int device_ordinal, se::DeviceAddressAllocator* allocator,
    se::Stream* stream, se::DeviceAddressBase data_in,
    se::DeviceAddressBase indices_in, se::DeviceAddressBase data_out,
    se::DeviceAddressBase indices_out, std::uint32_t batch, std::uint32_t n,
    std::uint32_t k) {
  (void)device_ordinal;
  (void)allocator;
  (void)stream;
  (void)data_in;
  (void)indices_in;
  (void)data_out;
  (void)indices_out;
  (void)batch;
  (void)n;
  (void)k;
  return absl::UnimplementedError(
      "select_k_payload_exec is only implemented for oneAPI/SYCL");
}

template absl::Status select_k_payload_exec<float>(
    int, se::DeviceAddressAllocator*, se::Stream*, se::DeviceAddressBase,
    se::DeviceAddressBase, se::DeviceAddressBase, se::DeviceAddressBase,
    std::uint32_t, std::uint32_t, std::uint32_t);

template absl::Status select_k_payload_exec<::xla::bfloat16>(
    int, se::DeviceAddressAllocator*, se::Stream*, se::DeviceAddressBase,
    se::DeviceAddressBase, se::DeviceAddressBase, se::DeviceAddressBase,
    std::uint32_t, std::uint32_t, std::uint32_t);

}  // namespace xla::gpu
