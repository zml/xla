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

#ifndef XLA_SERVICE_GPU_MUSA_MUSA_CUSTOM_KERNEL_SOURCE_H_
#define XLA_SERVICE_GPU_MUSA_MUSA_CUSTOM_KERNEL_SOURCE_H_

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/codegen/llvm_kernel_source.h"

namespace xla::gpu::musa {

// Parses and validates textual current-LLVM IR for the versioned MUSA custom
// kernel ABI. The source must contain exactly one marked kernel entry and its
// name must match `kernel_name`. Validation uses the same LLVM 14 compatibility
// boundary as production compilation, before the vendor bridge is invoked.
absl::StatusOr<LlvmKernelSource> ParseMusaCustomKernelSource(
    absl::string_view kernel_name, absl::string_view llvm_ir);

}  // namespace xla::gpu::musa

#endif  // XLA_SERVICE_GPU_MUSA_MUSA_CUSTOM_KERNEL_SOURCE_H_
