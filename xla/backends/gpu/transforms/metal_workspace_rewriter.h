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

#ifndef XLA_BACKENDS_GPU_TRANSFORMS_METAL_WORKSPACE_REWRITER_H_
#define XLA_BACKENDS_GPU_TRANSFORMS_METAL_WORKSPACE_REWRITER_H_

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/backends/gpu/runtime/metal_nvfp4_dispatch.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"

namespace xla::gpu {

class MetalWorkspaceRewriter : public HloModulePass {
 public:
  MetalWorkspaceRewriter() = default;
  MetalWorkspaceRewriter(char arch_size, int arch_gen)
      : arch_size_(arch_size), arch_gen_(arch_gen) {}

  absl::string_view name() const override { return "metal-workspace-rewriter"; }

 protected:
  absl::StatusOr<bool> RunImpl(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 private:
  char arch_size_ = kNvfp4DefaultArchSize;
  int arch_gen_ = kNvfp4DefaultArchGen;
};

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_TRANSFORMS_METAL_WORKSPACE_REWRITER_H_
