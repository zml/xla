/* Copyright 2025 The TensorFlow Authors. All Rights Reserved.

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

#ifndef XLA_SERVICE_GPU_KERNEL_CALL_H_
#define XLA_SERVICE_GPU_KERNEL_CALL_H_

#include <cstdint>
#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/stream_executor/launch_dim.h"

namespace xla::gpu {

inline constexpr absl::string_view kPtxCustomCallTarget =
    "__gpu$xla.gpu.ptx";
inline constexpr absl::string_view kMusaLlvmCustomCallTarget =
    "__gpu$xla.gpu.musa_llvm";

struct KernelCall {
  std::string name;
  std::string kernel_data;
  enum class KernelType {
    kPtxSource,
    kCudaBinary,
    kMusaLlvmSource,
  } kernel_type;

  stream_executor::BlockDim block_dim;
  stream_executor::ThreadDim thread_dim;
  size_t shared_mem;
  std::vector<int32_t> output_indices;

  // Parse the metadata of a structured GPU kernel custom call.
  static absl::StatusOr<KernelCall> Parse(absl::string_view backend_config,
                                          mlir::MLIRContext* mlir_context);
};

}  // namespace xla::gpu

#endif  // XLA_SERVICE_GPU_KERNEL_CALL_H_
