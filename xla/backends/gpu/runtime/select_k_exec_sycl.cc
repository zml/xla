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

#include <algorithm>
#include <cstdint>
#include <exception>
#include <limits>

#include <sycl/sycl.hpp>
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "xla/backends/gpu/runtime/select_k_exec.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/device_address_allocator.h"
#include "xla/stream_executor/stream.h"
#include "xla/types.h"  // IWYU pragma: keep

namespace xla::gpu {
namespace se = ::stream_executor;
namespace {

constexpr std::uint32_t kMaxK = 16;
constexpr std::uint32_t kDefaultWorkGroupSize = 256;
constexpr std::uint32_t kInvalidIndex =
    std::numeric_limits<std::uint32_t>::max();

template <typename T>
class SelectKExecSyclKernel;

inline bool Better(float lhs_value, std::uint32_t lhs_index, float rhs_value,
                   std::uint32_t rhs_index) {
  if (rhs_index == kInvalidIndex) {
    return true;
  }
  if (lhs_index == kInvalidIndex) {
    return false;
  }
  if (lhs_value == rhs_value) {
    return lhs_index < rhs_index;
  }
  return lhs_value > rhs_value;
}

inline void InsertCandidate(float value, std::uint32_t index,
                            float (&values)[kMaxK],
                            std::uint32_t (&indices)[kMaxK],
                            std::uint32_t k) {
  if (!Better(value, index, values[k - 1], indices[k - 1])) {
    return;
  }

  std::uint32_t pos = k - 1;
  while (pos > 0 && Better(value, index, values[pos - 1], indices[pos - 1])) {
    values[pos] = values[pos - 1];
    indices[pos] = indices[pos - 1];
    --pos;
  }
  values[pos] = value;
  indices[pos] = index;
}

template <typename T>
absl::Status LaunchSelectK(::sycl::queue* queue, const T* data_in,
                           const std::uint32_t* payload_indices_in,
                           T* data_out, std::uint32_t* indices_out,
                           std::uint32_t batch, std::uint32_t n,
                           std::uint32_t k) {
  if (queue == nullptr) {
    return absl::InternalError("SYCL select_k_exec stream has null queue");
  }

  const std::uint32_t max_work_group_size =
      static_cast<std::uint32_t>(queue->get_device().get_info<
                                 ::sycl::info::device::max_work_group_size>());
  const std::uint32_t work_group_size =
      std::min(kDefaultWorkGroupSize, max_work_group_size);
  if (work_group_size == 0) {
    return absl::InternalError("SYCL select_k_exec work-group size is zero");
  }

  try {
    queue->submit([&](::sycl::handler& cgh) {
      ::sycl::local_accessor<float, 1> local_values(
          ::sycl::range<1>(work_group_size * k), cgh);
      ::sycl::local_accessor<std::uint32_t, 1> local_indices(
          ::sycl::range<1>(work_group_size * k), cgh);

      cgh.parallel_for<SelectKExecSyclKernel<T>>(
          ::sycl::nd_range<1>(::sycl::range<1>(batch * work_group_size),
                              ::sycl::range<1>(work_group_size)),
          [=](::sycl::nd_item<1> item) {
            const std::uint32_t row =
                static_cast<std::uint32_t>(item.get_group(0));
            const std::uint32_t lid =
                static_cast<std::uint32_t>(item.get_local_id(0));
            const std::uint64_t row_offset =
                static_cast<std::uint64_t>(row) * n;

            float best_values[kMaxK];
            std::uint32_t best_indices[kMaxK];
            for (std::uint32_t i = 0; i < k; ++i) {
              best_values[i] = -std::numeric_limits<float>::infinity();
              best_indices[i] = kInvalidIndex;
            }

            for (std::uint32_t col = lid; col < n; col += work_group_size) {
              const float value =
                  static_cast<float>(data_in[row_offset + col]);
              const std::uint32_t payload =
                  payload_indices_in == nullptr
                      ? col
                      : payload_indices_in[row_offset + col];
              InsertCandidate(value, payload, best_values, best_indices, k);
            }

            const std::uint32_t local_offset = lid * k;
            for (std::uint32_t i = 0; i < k; ++i) {
              local_values[local_offset + i] = best_values[i];
              local_indices[local_offset + i] = best_indices[i];
            }
            item.barrier(::sycl::access::fence_space::local_space);

            if (lid == 0) {
              float row_values[kMaxK];
              std::uint32_t row_indices[kMaxK];
              for (std::uint32_t i = 0; i < k; ++i) {
                row_values[i] = -std::numeric_limits<float>::infinity();
                row_indices[i] = kInvalidIndex;
              }

              for (std::uint32_t worker = 0; worker < work_group_size;
                   ++worker) {
                const std::uint32_t worker_offset = worker * k;
                for (std::uint32_t i = 0; i < k; ++i) {
                  InsertCandidate(local_values[worker_offset + i],
                                  local_indices[worker_offset + i], row_values,
                                  row_indices, k);
                }
              }

              const std::uint64_t out_offset =
                  static_cast<std::uint64_t>(row) * k;
              for (std::uint32_t i = 0; i < k; ++i) {
                indices_out[out_offset + i] = row_indices[i];
                data_out[out_offset + i] = static_cast<T>(row_values[i]);
              }
            }
          });
    });
  } catch (const ::sycl::exception& e) {
    return absl::InternalError(
        absl::StrCat("SYCL select_k_exec launch failed: ", e.what()));
  } catch (const std::exception& e) {
    return absl::InternalError(
        absl::StrCat("SYCL select_k_exec launch failed: ", e.what()));
  }

  return absl::OkStatus();
}

}  // namespace

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

  if (k == 0 || batch == 0) {
    return absl::OkStatus();
  }
  if (k > n) {
    return absl::InvalidArgumentError("select_k_exec requires k <= n");
  }
  if (k > kMaxK) {
    return absl::UnimplementedError(
        absl::StrCat("SYCL select_k_exec supports k <= ", kMaxK, ", got ", k));
  }

  if (stream == nullptr) {
    return absl::InternalError("SYCL select_k_exec received a null stream");
  }
  auto* queue =
      static_cast<::sycl::queue*>(stream->platform_specific_handle().stream);
  return LaunchSelectK(
      queue, static_cast<const T*>(data_in.opaque()), nullptr,
      static_cast<T*>(data_out.opaque()),
      static_cast<std::uint32_t*>(indices_out.opaque()), batch, n, k);
}

template <typename T>
absl::Status select_k_payload_exec(
    int device_ordinal, se::DeviceAddressAllocator* allocator,
    se::Stream* stream, se::DeviceAddressBase data_in,
    se::DeviceAddressBase indices_in, se::DeviceAddressBase data_out,
    se::DeviceAddressBase indices_out, std::uint32_t batch, std::uint32_t n,
    std::uint32_t k) {
  (void)device_ordinal;
  (void)allocator;

  if (k == 0 || batch == 0) {
    return absl::OkStatus();
  }
  if (k > n) {
    return absl::InvalidArgumentError(
        "select_k_payload_exec requires k <= n");
  }
  if (k > kMaxK) {
    return absl::UnimplementedError(absl::StrCat(
        "SYCL select_k_payload_exec supports k <= ", kMaxK, ", got ", k));
  }

  if (stream == nullptr) {
    return absl::InternalError(
        "SYCL select_k_payload_exec received a null stream");
  }
  auto* queue =
      static_cast<::sycl::queue*>(stream->platform_specific_handle().stream);
  return LaunchSelectK(
      queue, static_cast<const T*>(data_in.opaque()),
      static_cast<const std::uint32_t*>(indices_in.opaque()),
      static_cast<T*>(data_out.opaque()),
      static_cast<std::uint32_t*>(indices_out.opaque()), batch, n, k);
}

template absl::Status select_k_exec<float>(int, se::DeviceAddressAllocator*,
                                           se::Stream*, se::DeviceAddressBase,
                                           se::DeviceAddressBase,
                                           se::DeviceAddressBase, std::uint32_t,
                                           std::uint32_t, std::uint32_t);

template absl::Status select_k_exec<::xla::bfloat16>(
    int, se::DeviceAddressAllocator*, se::Stream*, se::DeviceAddressBase,
    se::DeviceAddressBase, se::DeviceAddressBase, std::uint32_t, std::uint32_t,
    std::uint32_t);

template absl::Status select_k_payload_exec<float>(
    int, se::DeviceAddressAllocator*, se::Stream*, se::DeviceAddressBase,
    se::DeviceAddressBase, se::DeviceAddressBase, se::DeviceAddressBase,
    std::uint32_t, std::uint32_t, std::uint32_t);

template absl::Status select_k_payload_exec<::xla::bfloat16>(
    int, se::DeviceAddressAllocator*, se::Stream*, se::DeviceAddressBase,
    se::DeviceAddressBase, se::DeviceAddressBase, se::DeviceAddressBase,
    std::uint32_t, std::uint32_t, std::uint32_t);

}  // namespace xla::gpu
