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

#ifndef XLA_SERVICE_GPU_MUSA_MUSA_COMPILER_BUNDLE_H_
#define XLA_SERVICE_GPU_MUSA_MUSA_COMPILER_BUNDLE_H_

#include <memory>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/service/gpu/musa/musa_compilation_provider.h"

namespace xla::gpu::musa {

inline constexpr char kMusaCompilerBundleSchema[] =
    "xla-musa-compiler-bundle-v1";
inline constexpr char kMusaCompilerBundleEnvironment[] =
    "XLA_MUSA_COMPILER_BUNDLE";

// Loads the isolated compiler closure described by a bounded textual manifest.
// An empty path first checks XLA_MUSA_COMPILER_BUNDLE, then resolves
// musa_compiler_bundle.conf next to the shared object containing this function.
// With no configured or adjacent manifest, discovers the S4000 package layout.
absl::StatusOr<std::unique_ptr<MusaCompilationProvider>>
LoadMusaCompilationProviderFromBundle(absl::string_view manifest_path = {});

// Discovers the isolated compiler from a fixed, relocatable package layout.
// An empty plugin_path locates the shared object containing this function.
// An empty plugin_path uses the bridge and guard embedded in PJRT. An explicit
// plugin_path selects the legacy adjacent musa-compiler layout for diagnostics.
// An empty toolkit_root checks MUSA_PATH, then defaults to the adjacent musa-sdk
// directory. Explicit paths must be absolute; invalid overrides never fall back
// to host tools. Identities are derived from actual packaged files (after RPATH
// patching), using the bridge's own fingerprint implementation. No manifest is
// required.
absl::StatusOr<std::unique_ptr<MusaCompilationProvider>>
LoadMusaCompilationProviderFromLayout(absl::string_view plugin_path = {},
                                      absl::string_view toolkit_root = {});

}  // namespace xla::gpu::musa

#endif  // XLA_SERVICE_GPU_MUSA_MUSA_COMPILER_BUNDLE_H_
