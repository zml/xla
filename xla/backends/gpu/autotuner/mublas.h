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

#ifndef XLA_BACKENDS_GPU_AUTOTUNER_MUBLAS_H_
#define XLA_BACKENDS_GPU_AUTOTUNER_MUBLAS_H_

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/backends/autotuner/codegen_backend.h"
#include "xla/backends/gpu/autotuner/gpu_codegen_backend.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/compiler.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/xla.pb.h"

namespace xla::gpu {

// Stable algorithm identifiers exposed by the qualified muBLAS adapter.
// These identifiers are part of the serialized autotune-cache contract and
// deliberately do not expose vendor enum values.
inline constexpr int64_t kMublasDefaultAlgorithm = 0;
inline constexpr int64_t kMublasTensorOpAlgorithm = 1;

// Autotuner backend for the dedicated __mublas$gemm custom call. The MUSA
// 4.0.1 contract has two normalized algorithms and requires zero workspace.
class MublasBackend final : public GpuCodegenBackend {
 public:
  MublasBackend(stream_executor::StreamExecutor* stream_executor,
                const DebugOptions* debug_options, Compiler* compiler,
                const Compiler::GpuTargetConfig* target_config)
      : GpuCodegenBackend(autotuner::Backend::MUBLAS, debug_options, compiler,
                          target_config, stream_executor),
        stream_executor_(stream_executor) {}

  absl::StatusOr<std::vector<std::unique_ptr<BackendConfig>>>
  GetSupportedConfigs(const HloInstruction& instr) override;

  absl::StatusOr<std::unique_ptr<BackendConfig>> GetDefaultConfig(
      const HloInstruction& instr) override;

  absl::Status ApplyConfig(HloInstruction& instr,
                           const BackendConfig& config) override;

  std::string version() const override;

 private:
  bool IsSupported(const HloInstruction& instr) override;

  stream_executor::StreamExecutor* stream_executor_;
};

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_AUTOTUNER_MUBLAS_H_
