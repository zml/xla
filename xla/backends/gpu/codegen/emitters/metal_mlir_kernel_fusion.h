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

#ifndef XLA_BACKENDS_GPU_CODEGEN_EMITTERS_METAL_MLIR_KERNEL_FUSION_H_
#define XLA_BACKENDS_GPU_CODEGEN_EMITTERS_METAL_MLIR_KERNEL_FUSION_H_

#include <memory>
#include <string>

#include "absl/status/statusor.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/backends/gpu/codegen/emitters/mlir_kernel_emitter.h"
#include "xla/backends/gpu/codegen/fusion_emitter.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/ir_emitter_context.h"
#include "xla/stream_executor/device_description.h"

namespace xla::gpu {

// MetalMlirKernelFusion overrides ONLY the MLIR->LLVM step of MlirKernelFusion
// to emit Apple AIR directly (via the Metal AIR emitter) instead of lowering the
// MLIR module through the NVVM/ROCDL pipeline. Everything else is inherited
// unchanged — KernelArguments, the gpu::KernelThunk, launch_dimensions(),
// ComputeThreadIdTo*Indexing (so the cost model's dynamic_cast<MlirKernelFusion*>
// still resolves) — so the stock fusion pipeline drives our AIR kernel. It wraps
// the SAME MlirKernelEmitter (e.g. LoopFusion) the stock path would use, only for
// its launch dimensions / indexing. Selected in GetFusionEmitter's kLoop arm on
// the Metal platform.
class MetalMlirKernelFusion : public MlirKernelFusion {
 public:
  explicit MetalMlirKernelFusion(std::unique_ptr<MlirKernelEmitter> emitter)
      : MlirKernelFusion(std::move(emitter)) {}

  // Overrides the base MLIR->LLVM step to post-process the lowered module into
  // Apple AIR: attach the `!air.kernel` metadata air-as requires and scrub the
  // post-LLVM-15 function attributes air-as cannot lex. The base's async Emit()
  // reaches this via virtual dispatch (EmitLlvmModule). StampAirModuleEnvelope
  // runs later on the linked whole module in MetalGpuCompiler::CompileTargetBinary.
  xla::Future<LlvmKernelSource> CreateLLVMModule(
      const se::DeviceDescription& device, const HloFusionInstruction& fusion,
      const std::string& entry_function_name,
      const BufferAssignment* buffer_assignment, KernelCompiler* kernel_compiler,
      BorrowedMlirContext borrowed_context) const override;
};

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_CODEGEN_EMITTERS_METAL_MLIR_KERNEL_FUSION_H_
