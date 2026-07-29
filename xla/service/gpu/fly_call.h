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

#ifndef XLA_SERVICE_GPU_FLY_CALL_H_
#define XLA_SERVICE_GPU_FLY_CALL_H_

#include <cstdint>
#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "mlir/IR/MLIRContext.h"

namespace xla::gpu {

// Metadata for a __gpu$xla.gpu.fly custom call. The embedded module contains
// native Fly/FlyROCDL MLIR. It can expose either a top-level func.func using
// XLA's tensor kernel ABI, or a kernel gpu.func using Fly's native bare-pointer
// ABI. The function is selected by `name`.
struct FlyCall {
  std::string name;
  std::string ir;
  int64_t num_warps;
  int32_t grid_x;
  int32_t grid_y;
  int32_t grid_z;
  int32_t waves_per_eu = 0;
  int64_t shared_mem_bytes = 0;
  std::vector<int64_t> zeroed_outputs;

  static absl::StatusOr<FlyCall> Parse(absl::string_view backend_config,
                                       mlir::MLIRContext* mlir_context);
};

}  // namespace xla::gpu

#endif  // XLA_SERVICE_GPU_FLY_CALL_H_
