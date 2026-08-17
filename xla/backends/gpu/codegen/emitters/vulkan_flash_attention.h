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

#ifndef XLA_BACKENDS_GPU_CODEGEN_EMITTERS_VULKAN_FLASH_ATTENTION_H_
#define XLA_BACKENDS_GPU_CODEGEN_EMITTERS_VULKAN_FLASH_ATTENTION_H_

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "xla/backends/gpu/codegen/emitters/mlir_kernel_emitter.h"
#include "xla/codegen/emitters/computation_partitioner.h"
#include "xla/hlo/analysis/indexing_map.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/stream_executor/device_description.h"

namespace xla::gpu {

namespace internal {

mlir::Value SelectTensorElementOrZero(mlir::ImplicitLocOpBuilder& builder,
                                      mlir::Value condition, mlir::Value tensor,
                                      mlir::ValueRange indices);

}  // namespace internal

// Emits portable Vulkan subgroup flash-attention kernels.
class VulkanFlashAttentionEmitter final : public MlirKernelEmitter {
 public:
  VulkanFlashAttentionEmitter(const HloFusionInstruction& fusion,
                              const se::DeviceDescription& device);

  LaunchDimensions launch_dimensions() const override;

  absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> CreateMLIRModule(
      mlir::MLIRContext& mlir_context, const HloFusionInstruction& fusion,
      const std::string& entry_function_name,
      const BufferAssignment* buffer_assignment) const override;

  std::optional<IndexingMap> ComputeThreadIdToOutputIndexing(
      int64_t root_index, mlir::MLIRContext* context) const override;

  std::optional<std::vector<IndexingMap>> ComputeThreadIdToInputIndexing(
      int64_t root_index, mlir::MLIRContext* context) const override;

 protected:
  absl::Status EmitEntryFunction(
      const emitters::PartitionedComputations& computations,
      const emitters::CallTargetProvider& call_targets,
      mlir::func::FuncOp entry_function,
      const HloFusionInstruction& fusion) const override;

 private:
  absl::Status EmitKernel(mlir::func::FuncOp entry_function,
                          const HloFusionInstruction& fusion) const;

  int64_t query_heads_;
  int64_t query_length_;
  int64_t kv_heads_;
  int64_t kv_length_;
  int64_t head_dim_;
  int64_t subgroup_size_;
};

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_CODEGEN_EMITTERS_VULKAN_FLASH_ATTENTION_H_
