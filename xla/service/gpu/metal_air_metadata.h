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

// Apple AIR module-level metadata helpers for XLA's Metal backend.
//
// The Metal backend lowers HLO fusions to LLVM via XLA's standard MLIR->LLVM
// (GpuToAIR) path, then prints textual `.ll` for Apple's `air-as`. `air-as`
// requires an AIR "envelope" on the module (the air64 target triple + data
// layout, the air.max_* module flags, and the air.version / air.language_version
// / air.compile_options named metadata) plus, per kernel, an `!air.kernel`
// argument descriptor table. These two free functions attach exactly that;
// AirArg describes one kernel argument in AIR ABI terms.
//
// Apple's `air-as` is an ~LLVM-15-era parser, so callers must also strip
// function attributes down to the air-as-safe set before printing (see
// MetalMlirKernelFusion::CreateLLVMModule). That scrubbing is the caller's job;
// this file only attaches the metadata.

#include <string>

#include "absl/types/span.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"

namespace xla {
namespace gpu {
namespace metal {

// One kernel argument, in AIR ABI terms. Drives the per-argument `!air.kernel`
// descriptor metadata.
struct AirArg {
  enum class Kind {
    // A device (global) buffer: `ptr addrspace(1)`, descriptor "air.buffer",
    // address space 1, read or read_write.
    kDeviceBuffer,
    // A constant buffer holding a scalar param: `ptr addrspace(2)`, descriptor
    // "air.buffer" + "air.buffer_size", address space 2, read.
    kConstantBuffer,
    // The dispatch thread index: a scalar `i32` value, descriptor
    // "air.thread_position_in_grid" (no location_index / address space).
    kThreadPositionInGrid,
    // The thread's index within its threadgroup: scalar `i32`, descriptor
    // "air.thread_position_in_threadgroup". Auto-filled by the runtime.
    kThreadPositionInThreadgroup,
    // The threadgroup's index in the grid: scalar `i32`, descriptor
    // "air.threadgroup_position_in_grid". Auto-filled by the runtime.
    kThreadgroupPositionInGrid,
  };

  Kind kind;
  std::string name;         // air.arg_name, e.g. "a", "out", "n", "gid"
  std::string type_name;    // air.arg_type_name, e.g. "float", "uint"
  unsigned type_size = 4;   // air.arg_type_size (bytes)
  unsigned type_align = 4;  // air.arg_type_align_size (bytes)
  bool read_only = true;    // device buffers only: air.read vs air.read_write
};

// Stamps the Apple AIR module envelope onto `module`: the air64 target triple +
// data layout, the air.max_* module flags, and the air.version /
// air.language_version / air.compile_options named metadata. Idempotent for the
// triple/datalayout and each flag; called once per fusion kernel and again on
// the linked whole module in CompileTargetBinary (whose merged-in constants
// module carries only triple+datalayout).
void StampAirModuleEnvelope(llvm::Module& module);

// Attaches the `!air.kernel` named-metadata entry + per-argument AIR descriptors
// for `f` onto `f->getParent()`; `f`'s parameters must correspond to `args` in
// order. Free-standing (operates on any llvm::Function, using f->getContext())
// so a natively MLIR->LLVM-lowered kernel module (the GpuToAIR path) can be
// stamped with the metadata air-as requires.
void AttachAirKernelMetadata(llvm::Function* f, absl::Span<const AirArg> args);

}  // namespace metal
}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_METAL_AIR_METADATA_H_
