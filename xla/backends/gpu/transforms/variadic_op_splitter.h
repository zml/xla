/* Copyright 2018 The OpenXLA Authors.

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

#ifndef XLA_BACKENDS_GPU_TRANSFORMS_VARIADIC_OP_SPLITTER_H_
#define XLA_BACKENDS_GPU_TRANSFORMS_VARIADIC_OP_SPLITTER_H_

#include <cstdint>

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"

namespace xla {
namespace gpu {

// Splits variadic ops with many operands into pieces such that we don't exceed
// the parameter space on the GPU. Currently only concatenate ops are split up.
//
// `max_parameters` is both the threshold above which an op is split and the
// maximum fan-in of each resulting piece. The default (128) bounds the generic
// GPU parameter space; Metal needs a much smaller value because its buffer
// argument table has only 31 slots (a wide concatenate otherwise fails in
// air-opt with "buffer argument in31 has invalid location").
class VariadicOpSplitter : public HloModulePass {
 public:
  explicit VariadicOpSplitter(int64_t max_parameters = 128)
      : max_parameters_(max_parameters) {}

  absl::string_view name() const override { return "variadic-op-splitter"; }

 protected:
  absl::StatusOr<bool> RunImpl(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 private:
  int64_t max_parameters_;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_BACKENDS_GPU_TRANSFORMS_VARIADIC_OP_SPLITTER_H_
