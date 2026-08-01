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

#include "xla/backends/gpu/runtime/metal_gemm_thunk.h"

#include <cstdint>
#include <utility>

#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
#include "xla/runtime/buffer_use.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/service/gpu/metalblas_gemm.h"
#include "xla/shape.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/kernel_args.h"
#include "xla/stream_executor/kernel_spec.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace gpu {

namespace se = ::stream_executor;

MetalGemmThunk::MetalGemmThunk(ThunkInfo thunk_info, MetalGemmLaunch launch,
                               BufferAllocation::Slice lhs, Shape lhs_shape,
                               BufferAllocation::Slice rhs, Shape rhs_shape,
                               BufferAllocation::Slice out, Shape out_shape,
                               BufferAllocation::Slice num_tokens,
                               Shape num_tokens_shape)
    : Thunk(Kind::kGemm, std::move(thunk_info)),
      launch_(std::move(launch)),
      lhs_(lhs),
      lhs_shape_(std::move(lhs_shape)),
      rhs_(rhs),
      rhs_shape_(std::move(rhs_shape)),
      out_(out),
      out_shape_(std::move(out_shape)),
      num_tokens_(num_tokens),
      num_tokens_shape_(std::move(num_tokens_shape)),
      has_num_tokens_(num_tokens.allocation() != nullptr) {}

MetalGemmThunk::~MetalGemmThunk() {
  absl::MutexLock lock(&mu_);
  if (executor_ != nullptr) {
    if (!params_mem_.is_null()) executor_->Deallocate(&params_mem_);
    if (!staging_mem_.is_null()) executor_->Deallocate(&staging_mem_);
    if (!accum_params_mem_.is_null()) executor_->Deallocate(&accum_params_mem_);
  }
}

absl::Status MetalGemmThunk::ExecuteOnStream(const ExecuteParams& params) {
  se::Stream* stream = params.stream;
  se::StreamExecutor* executor = stream->parent();
  const BufferAllocations& allocs = *params.buffer_allocations;
  se::DeviceAddressBase lhs = allocs.GetDeviceAddress(lhs_);
  se::DeviceAddressBase rhs = allocs.GetDeviceAddress(rhs_);
  se::DeviceAddressBase out = allocs.GetDeviceAddress(out_);

  const bool splitk = launch_.splitk_partitions > 0;

  absl::MutexLock lock(&mu_);
  if (kernel_ == nullptr || executor_ != executor) {
    if (executor_ != nullptr) {
      if (!params_mem_.is_null()) executor_->Deallocate(&params_mem_);
      if (!staging_mem_.is_null()) executor_->Deallocate(&staging_mem_);
      if (!accum_params_mem_.is_null())
        executor_->Deallocate(&accum_params_mem_);
      params_mem_ = se::DeviceAddressBase();
      staging_mem_ = se::DeviceAddressBase();
      accum_params_mem_ = se::DeviceAddressBase();
      accum_kernel_ = nullptr;
    }
    auto spec = se::KernelLoaderSpec::CreateOwningMetalLibraryInMemorySpec(
        launch_.metallib, launch_.kernel_name,
        /*arity=*/has_num_tokens_ ? 5 : 4);
    TF_ASSIGN_OR_RETURN(kernel_, executor->LoadKernel(spec));

    const void* params_src = launch_.params.data();
    uint64_t params_bytes =
        static_cast<uint64_t>(launch_.params.size()) * sizeof(uint32_t);
    if (splitk) {
      params_src = launch_.splitk_params.data();
      params_bytes = launch_.splitk_params.size() * sizeof(uint32_t);
      auto aspec = se::KernelLoaderSpec::CreateOwningMetalLibraryInMemorySpec(
          launch_.metallib, launch_.accum_kernel_name, /*arity=*/5);
      TF_ASSIGN_OR_RETURN(accum_kernel_, executor->LoadKernel(aspec));
      staging_mem_ = executor->Allocate(launch_.staging_bytes);
      accum_params_mem_ =
          executor->Allocate(launch_.accum_params.size() * sizeof(uint32_t));
      if (staging_mem_.is_null() || accum_params_mem_.is_null()) {
        kernel_ = nullptr;
        return absl::ResourceExhaustedError(
            "Failed to allocate Metal split-K staging buffers.");
      }
      TF_RETURN_IF_ERROR(
          stream->Memcpy(&accum_params_mem_, launch_.accum_params.data(),
                         launch_.accum_params.size() * sizeof(uint32_t)));
    }
    params_mem_ = executor->Allocate(params_bytes);
    if (params_mem_.is_null()) {
      kernel_ = nullptr;
      return absl::ResourceExhaustedError(
          "Failed to allocate Metal GEMM params buffer.");
    }
    TF_RETURN_IF_ERROR(stream->Memcpy(&params_mem_, params_src, params_bytes));
    executor_ = executor;
  }

  if (splitk) {
    se::KernelArgsPackedArray sk_args(/*num_args=*/4);
    sk_args.add_argument(lhs);           // A = x [M,K]
    sk_args.add_argument(rhs);           // B = W [N,K] (TRANS_B)
    sk_args.add_argument(staging_mem_);  // C_split f32 [parts, M, N]
    sk_args.add_argument(params_mem_);   // GEMMSpiltKParams
    TF_RETURN_IF_ERROR(kernel_->Launch(launch_.thread_dim, launch_.block_dim,
                                       stream, sk_args));
    const char* p2 = static_cast<const char*>(accum_params_mem_.opaque());
    se::KernelArgsPackedArray acc_args(/*num_args=*/5);
    acc_args.add_argument(staging_mem_);
    acc_args.add_argument(out);
    acc_args.add_argument(se::DeviceAddressBase(const_cast<char*>(p2 + 0), 4));
    acc_args.add_argument(se::DeviceAddressBase(const_cast<char*>(p2 + 4), 4));
    acc_args.add_argument(se::DeviceAddressBase(const_cast<char*>(p2 + 8), 4));
    return accum_kernel_->Launch(launch_.accum_thread_dim,
                                 launch_.accum_block_dim, stream, acc_args);
  }

  se::KernelArgsPackedArray kernel_args(has_num_tokens_ ? 5 : 4);
  if (launch_.swap_ab) {
    kernel_args.add_argument(rhs);
    kernel_args.add_argument(lhs);
  } else {
    kernel_args.add_argument(lhs);
    kernel_args.add_argument(rhs);
  }
  kernel_args.add_argument(out);
  kernel_args.add_argument(params_mem_);

  if (has_num_tokens_) {
    kernel_args.add_argument(allocs.GetDeviceAddress(num_tokens_));  // buffer(4)
  }

  return kernel_->Launch(launch_.thread_dim, launch_.block_dim, stream,
                         kernel_args);
}

Thunk::BufferUses MetalGemmThunk::buffer_uses() const {
  Thunk::BufferUses uses = {
      BufferUse::Read(lhs_, lhs_shape_),
      BufferUse::Read(rhs_, rhs_shape_),
      BufferUse::Write(out_, out_shape_),
  };
  if (has_num_tokens_) {
    uses.push_back(BufferUse::Read(num_tokens_, num_tokens_shape_));
  }
  return uses;
}

absl::StatusOr<ThunkProto> MetalGemmThunk::ToProto() const {
  return absl::UnimplementedError(
      "MetalGemmThunk does not support serialization.");
}

}  // namespace gpu
}  // namespace xla
