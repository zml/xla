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

#include "xla/stream_executor/metal/metal_kernel.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/stream_executor/kernel_args.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/stream_executor/metal/metal_runtime.h"
#include "xla/stream_executor/metal/metal_stream.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/platform/statusor.h"

namespace stream_executor::metal {

MetalKernel::~MetalKernel() {
  if (executor_ != nullptr) {
    executor_->UnloadKernel(this);
  }
  if (pipeline_ != nullptr) {
    ReleaseObject(pipeline_);
  }
  if (function_ != nullptr) {
    ReleaseObject(function_);
  }
  if (library_ != nullptr) {
    ReleaseObject(library_);
  }
}

absl::StatusOr<int32_t> MetalKernel::GetMaxOccupiedBlocksPerCore(
    ThreadDim threads, size_t dynamic_shared_memory_bytes) const {
  return absl::UnimplementedError(
      "GetMaxOccupiedBlocksPerCore is not implemented for Metal.");
}

absl::Status MetalKernel::Launch(
    const ThreadDim& thread_dims, const BlockDim& block_dims,
    const std::optional<ClusterDim>& cluster_dims, Stream* stream,
    const KernelArgs& args) {
  if (cluster_dims.has_value()) {
    return absl::UnimplementedError(
        "Cluster launches are not implemented for Metal.");
  }
  if (pipeline_ == nullptr) {
    return absl::InternalError("Metal kernel pipeline is not set.");
  }

  auto launch = [&](const KernelArgsPackedArrayBase& packed) {
    int32_t expected_number_of_arguments =
        Arity() + (packed.number_of_shared_bytes() > 0 ? 1 : 0);
    CHECK_EQ(expected_number_of_arguments, packed.number_of_arguments())
        << "Kernel " << name() << " has " << packed.number_of_arguments()
        << " arguments, but expected " << expected_number_of_arguments
        << "; arity=" << Arity()
        << "; number_of_shared_bytes=" << packed.number_of_shared_bytes();

    std::vector<void*> kernel_arg_addresses;
    kernel_arg_addresses.reserve(packed.argument_addresses().size());
    for (const void* arg : packed.argument_addresses()) {
      kernel_arg_addresses.push_back(const_cast<void*>(arg));
    }
    absl::Span<const KernelArgumentMetadata> metadata =
        packed.argument_metadata();
    if (metadata.size() != kernel_arg_addresses.size()) {
      return absl::InternalError(
          "Metal kernel arguments expose inconsistent binding metadata.");
    }
    size_t kernel_args_size = kernel_arg_addresses.size();
    std::array<void*, 2> config = {kernel_arg_addresses.data(),
                                   &kernel_args_size};
    auto* metal_stream = dynamic_cast<MetalStream*>(stream);
    if (metal_stream == nullptr) {
      return absl::InvalidArgumentError("Expected a MetalStream.");
    }
    return metal_stream->LaunchMetalKernel(
        thread_dims, block_dims, cluster_dims, pipeline_, function_,
        uses_argument_buffer_, name(), reinterpret_cast<void**>(config.data()),
        metadata, packed.number_of_shared_bytes(), use_pdl());
  };

  if (auto* packed = DynCast<KernelArgsPackedArrayBase>(&args)) {
    return launch(*packed);
  }

  if (auto* device_mem = DynCast<KernelArgsDeviceAddressArray>(&args)) {
    auto& pack = args_packing();
    if (!pack) {
      return absl::InternalError(
          "Metal kernel is missing a custom arguments packing function.");
    }
    TF_ASSIGN_OR_RETURN(auto packed, pack(*this, *device_mem));
    return launch(*packed);
  }

  return absl::InternalError("Unsupported Metal kernel arguments type.");
}

}  // namespace stream_executor::metal
