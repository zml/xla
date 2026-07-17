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

#ifndef XLA_STREAM_EXECUTOR_METAL_METAL_KERNEL_H_
#define XLA_STREAM_EXECUTOR_METAL_METAL_KERNEL_H_

#include <cstddef>
#include <cstdint>
#include <optional>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"

namespace stream_executor::metal {

class MetalKernel : public Kernel {
 public:
  explicit MetalKernel(StreamExecutor* executor) : executor_(executor) {}
  ~MetalKernel() override;

  void set_arity(unsigned arity) { arity_ = arity; }
  unsigned Arity() const override { return arity_; }

  void set_library(void* library) { library_ = library; }
  void set_function(void* function) { function_ = function; }
  void set_pipeline(void* pipeline) { pipeline_ = pipeline; }

  void* pipeline() const { return pipeline_; }
  void set_uses_argument_buffer(bool uses_argument_buffer) {
    uses_argument_buffer_ = uses_argument_buffer;
  }

  absl::StatusOr<int32_t> GetMaxOccupiedBlocksPerCore(
      ThreadDim threads, size_t dynamic_shared_memory_bytes) const override;

 private:
  absl::Status Launch(const ThreadDim& thread_dims, const BlockDim& block_dims,
                      const std::optional<ClusterDim>& cluster_dims,
                      Stream* stream, const KernelArgs& args) override;

  StreamExecutor* executor_ = nullptr;
  void* library_ = nullptr;
  void* function_ = nullptr;
  void* pipeline_ = nullptr;
  unsigned arity_ = 0;
  bool uses_argument_buffer_ = false;
};

}  // namespace stream_executor::metal

#endif  // XLA_STREAM_EXECUTOR_METAL_METAL_KERNEL_H_
