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

#include <algorithm>
#include <memory>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/types/span.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/backends/autotuner/backends.pb.h"
#include "xla/backends/autotuner/codegen_backend.h"
#include "xla/backends/gpu/autotuner/factory.h"
#include "xla/backends/gpu/autotuner/mublas.h"
#include "xla/backends/gpu/autotuner/mudnn.h"
#include "xla/hlo/analysis/alias_info.h"
#include "xla/service/compiler.h"
#include "xla/service/hlo_cost_analysis.h"
#include "xla/stream_executor/musa/musa_platform_id.h"
#include "xla/stream_executor/platform/platform_object_registry.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/xla.pb.h"

namespace xla::gpu {

std::vector<std::unique_ptr<CodegenBackend>> GetCodegenBackendsForMusa(
    stream_executor::StreamExecutor* stream_executor,
    stream_executor::DeviceAddressAllocator* device_allocator,
    const DebugOptions* debug_options, Compiler* compiler,
    const Compiler::GpuTargetConfig* target_config, const AliasInfo* alias_info,
    mlir::MLIRContext* mlir_context,
    HloCostAnalysis::ShapeSizeFunction shape_size_fn,
    absl::Span<const autotuner::Backend> backend_allowlist) {
  (void)alias_info;
  (void)mlir_context;
  (void)shape_size_fn;

  std::vector<std::unique_ptr<CodegenBackend>> backends;
  backends.push_back(std::make_unique<MublasBackend>(
      stream_executor, debug_options, compiler, target_config));
  backends.push_back(
      std::make_unique<MudnnBackend>(stream_executor, debug_options, compiler,
                                     target_config, device_allocator));

  if (!backend_allowlist.empty()) {
    backends.erase(
        std::remove_if(backends.begin(), backends.end(),
                       [&](const std::unique_ptr<CodegenBackend>& backend) {
                         return !absl::c_linear_search(backend_allowlist,
                                                       backend->backend());
                       }),
        backends.end());
  }
  return backends;
}

STREAM_EXECUTOR_REGISTER_OBJECT_STATICALLY(GetCodegenBackendsMusaRegistration,
                                           GetCodegenBackends,
                                           se::musa::kMusaPlatformId,
                                           GetCodegenBackendsForMusa);

}  // namespace xla::gpu
