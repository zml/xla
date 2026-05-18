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

#ifndef XLA_SERVICE_GPU_METAL_MSL_EMITTER_H_
#define XLA_SERVICE_GPU_METAL_MSL_EMITTER_H_

#include <string>

#include "absl/status/statusor.h"

namespace llvm {
class Module;
}  // namespace llvm

namespace xla {
namespace gpu {

// Emits Metal Shading Language source from the LLVM GPU IR produced by XLA.
//
// This is the initial direct-MSL bridge for the Metal backend. It supports
// simple data-parallel kernels and the bounds-check control-flow shape emitted
// for common fusions; unsupported IR returns a descriptive error instead of
// falling back to MPS, PTX, SPIR-V, or another GPU backend.
absl::StatusOr<std::string> EmitMslFromLlvmModule(const llvm::Module& module);

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_METAL_MSL_EMITTER_H_
