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

#include <memory>
#include <vector>

#include "absl/types/span.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/backends/autotuner/backends.pb.h"
#include "xla/backends/autotuner/codegen_backend.h"
#include "xla/backends/gpu/autotuner/factory.h"
#include "xla/backends/gpu/autotuner/triton.h"
#include "xla/hlo/analysis/alias_info.h"
#include "xla/service/compiler.h"
#include "xla/service/hlo_cost_analysis.h"
#include "xla/stream_executor/platform/platform_object_registry.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/stream_executor/sycl/sycl_platform_id.h"
#include "xla/xla.pb.h"

namespace xla {
namespace gpu {
namespace {

bool AllowsTriton(absl::Span<const autotuner::Backend> backend_allowlist) {
  if (backend_allowlist.empty()) {
    return true;
  }
  for (autotuner::Backend backend : backend_allowlist) {
    if (backend == autotuner::Backend::TRITON) {
      return true;
    }
  }
  return false;
}

}  // namespace

std::vector<std::unique_ptr<CodegenBackend>> GetCodegenBackendsForSycl(
    stream_executor::StreamExecutor*,
    stream_executor::DeviceAddressAllocator*, const DebugOptions* debug_options,
    Compiler* compiler, const Compiler::GpuTargetConfig* target_config,
    const AliasInfo* alias_info, mlir::MLIRContext* mlir_context,
    HloCostAnalysis::ShapeSizeFunction,
    absl::Span<const autotuner::Backend> backend_allowlist) {
  std::vector<std::unique_ptr<CodegenBackend>> backends;
  if (AllowsTriton(backend_allowlist)) {
    backends.push_back(std::make_unique<TritonBackend>(
        debug_options, compiler, target_config, alias_info, mlir_context));
  }
  return backends;
}

STREAM_EXECUTOR_REGISTER_OBJECT_STATICALLY(GetCodegenBackendsSyclRegistration,
                                           GetCodegenBackends,
                                           se::sycl::kSyclPlatformId,
                                           GetCodegenBackendsForSycl);

}  // namespace gpu
}  // namespace xla
