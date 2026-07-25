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

#ifndef XLA_SERVICE_GPU_MUSA_MUSA_LLVM14_COMPATIBILITY_H_
#define XLA_SERVICE_GPU_MUSA_MUSA_LLVM14_COMPATIBILITY_H_

#include <string>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/service/gpu/musa/musa_bridge_ir_validator.h"

namespace llvm {
class Module;
}  // namespace llvm

namespace xla::gpu::musa {

// Bump this token whenever the accepted current-LLVM surface, a
// normalization rule, or an LLVM-14 spelling changes. The compiler provider
// and executable envelope include it in their cache and compatibility identity.
inline constexpr char kMusaLlvm14CompatibilityRevision[] =
    "musa-llvm14-compat-v5";

struct MusaLlvm14CompatibilityResult {
  std::string normalized_llvm;
  std::string normalized_llvm_sha256;
  MusaBridgeIrMetadata metadata;
};

// Clones and normalizes a verified current-LLVM module into the finite textual
// interchange accepted by the pinned vendor LLVM 14 bridge. The input module
// is never mutated. Diagnostics are bounded and never include LLVM IR or an
// untrusted source path.
absl::StatusOr<MusaLlvm14CompatibilityResult> NormalizeMusaLlvmForLlvm14(
    const llvm::Module& module, absl::string_view module_name);

// Text entry point for compatibility corpora and external reproducers. The
// production compiler uses the Module entry point immediately before
// serialization.
absl::StatusOr<MusaLlvm14CompatibilityResult> NormalizeMusaLlvmTextForLlvm14(
    absl::string_view current_llvm, absl::string_view module_name);

}  // namespace xla::gpu::musa

#endif  // XLA_SERVICE_GPU_MUSA_MUSA_LLVM14_COMPATIBILITY_H_
