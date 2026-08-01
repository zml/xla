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

#ifndef XLA_SERVICE_GPU_METAL_AIR_METADATA_H_
#define XLA_SERVICE_GPU_METAL_AIR_METADATA_H_

#include <string>

#include "absl/types/span.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"

namespace xla {
namespace gpu {
namespace metal {

struct AirArg {
  enum class Kind {
    kDeviceBuffer,
    kConstantBuffer,
    kThreadPositionInGrid,
    kThreadPositionInThreadgroup,
    kThreadgroupPositionInGrid,
  };

  Kind kind;
  std::string name;         // air.arg_name, e.g. "a", "out", "n", "gid"
  std::string type_name;    // air.arg_type_name, e.g. "float", "uint"
  unsigned type_size = 4;   // air.arg_type_size (bytes)
  unsigned type_align = 4;  // air.arg_type_align_size (bytes)
  bool read_only = true;    // device buffers only: air.read vs air.read_write
};

void StampAirModuleEnvelope(llvm::Module& module);

void AttachAirKernelMetadata(llvm::Function* f, absl::Span<const AirArg> args);

}  // namespace metal
}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_METAL_AIR_METADATA_H_
