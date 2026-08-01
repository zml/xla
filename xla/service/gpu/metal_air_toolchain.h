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

#ifndef XLA_SERVICE_GPU_METAL_AIR_TOOLCHAIN_H_
#define XLA_SERVICE_GPU_METAL_AIR_TOOLCHAIN_H_

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"

// Shared Apple-Metal AIR toolchain entry points (air-as / air-opt / metallib),
// invoked by MetalGpuCompiler::CompileTargetBinary and the metalBLAS gemm path
// to run the identical assemble-to-metallib tail. A leaf library: no LLVM/MLIR
// or emitter dependency, just subprocess + filesystem.

namespace xla {
namespace gpu {

// Runs `argv` as a subprocess. On success returns captured stdout when
// `capture_stdout` is true, otherwise captured stderr. On non-zero exit returns
// an InternalError including the command and its stderr.
absl::StatusOr<std::string> RunCommand(std::vector<std::string> argv,
                                       bool capture_stdout);

// Locates a Metal toolchain tool under `$METAL_TOOLCHAIN/<tool_name>`.
absl::StatusOr<std::string> FindMetalTool(const char* tool_name);

// Assembles textual AIR (`source`, LLVM .ll text) into a `.metallib` via
// air-as -> air-opt --O3 -> metallib. `temp_name` labels the temp artifacts.
absl::StatusOr<std::vector<uint8_t>> CompileMetalAirToMetallib(
    absl::string_view source, absl::string_view temp_name);

// Compiles Metal Shading Language `source` to a `.metallib` via the Metal
// compiler (`metal -std=metal4.0 -c`) followed by `metallib`. Optional `subs`
// are applied as a literal StrReplaceAll on the source first (kernel
// specialization). Used by the runtime thunks that ship a `.metal` shader
// (flash-attn, paged-attn, kv-write, topk) — distinct from
// CompileMetalAirToMetallib's AIR-text path.
absl::StatusOr<std::vector<uint8_t>> CompileMetalSourceToMetallib(
    absl::string_view source,
    const std::vector<std::pair<std::string, std::string>>& subs = {});

// As CompileMetalSourceToMetallib, but memoizes on the substituted source so the
// (expensive) compilation runs once per distinct specialization per process.
absl::StatusOr<std::vector<uint8_t>> CompileMetalSourceToMetallibCached(
    absl::string_view source,
    const std::vector<std::pair<std::string, std::string>>& subs = {});

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_METAL_AIR_TOOLCHAIN_H_
