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

#ifndef XLA_BACKENDS_GPU_CODEGEN_COLLECTIVE_EPILOGUE_H_
#define XLA_BACKENDS_GPU_CODEGEN_COLLECTIVE_EPILOGUE_H_

#include <cstdint>
#include <vector>

#include "xla/hlo/ir/hlo_opcode.h"

namespace xla::gpu {

// One operation in a linear FP32 epilogue rooted at a reduced BF16 value. A
// step combines the accumulator with either a same-shaped BF16 input buffer or
// a scalar constant. `accumulator_is_lhs` preserves subtract/divide order; it
// is immaterial for commutative operations.
struct CollectiveEpilogueStep {
  HloOpcode opcode;
  bool accumulator_is_lhs = true;
  int64_t buffer_index = -1;
  double scalar_value = 0.0;

  bool uses_buffer() const { return buffer_index >= 0; }
};

struct CollectiveEpilogue {
  int64_t buffer_count = 0;
  std::vector<CollectiveEpilogueStep> steps;

  bool empty() const { return steps.empty(); }
};

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_CODEGEN_COLLECTIVE_EPILOGUE_H_
