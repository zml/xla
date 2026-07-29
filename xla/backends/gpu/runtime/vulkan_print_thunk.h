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

#ifndef XLA_BACKENDS_GPU_RUNTIME_VULKAN_PRINT_THUNK_H_
#define XLA_BACKENDS_GPU_RUNTIME_VULKAN_PRINT_THUNK_H_

#include <string>

#include "absl/status/status.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/service/buffer_assignment.h"
#include "xla/shape.h"

namespace xla::gpu {

// Implements zml.Tensor.print on Vulkan. The thunk waits for prior work, reads
// the operand on the host, and writes a labeled value summary to stderr.
class VulkanPrintThunk : public Thunk {
 public:
  VulkanPrintThunk(ThunkInfo thunk_info, std::string label,
                   BufferAllocation::Slice operand, Shape operand_shape);

  VulkanPrintThunk(const VulkanPrintThunk&) = delete;
  VulkanPrintThunk& operator=(const VulkanPrintThunk&) = delete;

  absl::Status ExecuteOnStream(const ExecuteParams& params) override;
  absl::StatusOr<ThunkProto> ToProto() const override;
  BufferUses buffer_uses() const override;

 private:
  const std::string label_;
  const BufferAllocation::Slice operand_;
  const Shape operand_shape_;
};

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_RUNTIME_VULKAN_PRINT_THUNK_H_
