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

#ifndef XLA_BACKENDS_GPU_CODEGEN_TRITON_NVFP4_DECODE_DOT_H_
#define XLA_BACKENDS_GPU_CODEGEN_TRITON_NVFP4_DECODE_DOT_H_

#include <cstdint>
#include <optional>

#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/stream_executor/device_description.h"

namespace xla::gpu {

struct Nvfp4DecodeLimits {
  bool claim;
  bool swap;
  int64_t max_decode_rows;
  int64_t min_weight_rows;
  int64_t min_weight_tile;
  int64_t min_batch_tile;
  struct Seed {
    int64_t weight_tile;
    int64_t block_k;
    int num_warps;
    int num_stages;
  } seed;
  // A ceiling on the arm's split-K; the block_k ladder is what usually binds.
  int64_t max_split_k;
};

const Nvfp4DecodeLimits& Nvfp4DecodeLimitsFor(
    const se::GpuComputeCapability& gpu_version);

// The widest contracting tile the autotuner enumerates for `k`, or 0. One answer for the matcher,
// the seed, the search and the split chooser.
int64_t WidestNvfp4BlockK(int64_t k);

bool HasNvfp4BlockK(int64_t k);

int64_t Nvfp4BlockKAtMost(int64_t k, int64_t preferred);

struct Nvfp4DecodeDotSpec {
  int64_t batch;
  int64_t weight_rows;
  int64_t k;
  bool weight_on_lhs;
};

inline constexpr absl::string_view kNvfp4DecodeDotComputationPrefix =
    "nvfp4_decode_dot_";

std::optional<Nvfp4DecodeDotSpec> MatchNvfp4DecodeDot(
    const HloScaledDotInstruction& dot,
    const se::GpuComputeCapability& gpu_version);

std::optional<Nvfp4DecodeDotSpec> MatchNvfp4DecodeDotFusion(
    const HloFusionInstruction& fusion);

struct Nvfp4DecodeDotConfig {
  int64_t weight_tile;
  int64_t batch_tile;
  int64_t block_k;
  int num_warps;
  int num_stages;
  bool swap;
};

Nvfp4DecodeDotConfig Nvfp4DecodeDotSeed(const Nvfp4DecodeDotSpec& spec,
                                        const Nvfp4DecodeLimits& limits);

std::optional<Nvfp4DecodeDotConfig> Nvfp4DecodeDotConfigFor(
    const HloScaledDotInstruction& dot,
    const se::GpuComputeCapability& gpu_version);

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_CODEGEN_TRITON_NVFP4_DECODE_DOT_H_
