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

#include "xla/backends/gpu/codegen/triton/nvfp4_decode_dot.h"

#include <algorithm>
#include <cstdint>
#include <optional>

#include "absl/algorithm/container.h"
#include "absl/log/log.h"
#include "absl/strings/match.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/shape.h"
#include "xla/stream_executor/device_description.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {
namespace {

constexpr int64_t kNvfp4ScaleBlock = 16;

constexpr Nvfp4DecodeLimits kTcgen05Limits = {
    /*claim=*/true,
    /*swap=*/true,
    /*max_decode_rows=*/64,
    /*min_weight_rows=*/128,
    /*min_weight_tile=*/128,
    /*min_batch_tile=*/16,
    /*seed=*/{/*weight_tile=*/128, /*block_k=*/256, /*num_warps=*/8,
              /*num_stages=*/4},
    /*max_split_k=*/4,
};

constexpr Nvfp4DecodeLimits kSm120Limits = {
    /*claim=*/true,
    /*swap=*/true,
    /*max_decode_rows=*/64,
    /*min_weight_rows=*/128,
    /*min_weight_tile=*/16,
    /*min_batch_tile=*/16,
    /*seed=*/{/*weight_tile=*/128, /*block_k=*/256, /*num_warps=*/4,
              /*num_stages=*/3},
    /*max_split_k=*/4,
};

constexpr Nvfp4DecodeLimits kNoClaim = {
    /*claim=*/false, /*swap=*/false, 0, 0, 0, 0, {0, 0, 0, 0},
    /*max_split_k=*/1,
};

bool IsNvfp4Operand(const HloInstruction& values, const HloInstruction& scales) {
  return values.shape().element_type() == F4E2M1FN &&
         scales.shape().element_type() == F8E4M3FN;
}

// The contracting tiles the autotuner enumerates; nothing outside is a legal configuration.
constexpr int64_t kMinBlockK = 128;
constexpr int64_t kMaxBlockK = 512;

int64_t RoundUpToPowerOfTwo(int64_t v) {
  int64_t p = 1;
  while (p < v) p *= 2;
  return p;
}

}  // namespace

int64_t WidestNvfp4BlockK(int64_t k) { return Nvfp4BlockKAtMost(k, kMaxBlockK); }

bool HasNvfp4BlockK(int64_t k) { return WidestNvfp4BlockK(k) != 0; }

int64_t Nvfp4BlockKAtMost(int64_t k, int64_t preferred) {
  int64_t best = 0;
  for (int64_t block_k = kMinBlockK; block_k <= kMaxBlockK; block_k *= 2) {
    if (block_k <= preferred && k % block_k == 0) best = block_k;
  }
  return best;
}

const Nvfp4DecodeLimits& Nvfp4DecodeLimitsFor(
    const se::GpuComputeCapability& gpu_version) {
  const se::CudaComputeCapability* cc = gpu_version.cuda_compute_capability();
  if (cc == nullptr) return kNoClaim;
  if (cc->major == se::CudaComputeCapability::kBlackwell) return kTcgen05Limits;
  if (cc->major == se::CudaComputeCapability::kBlackwell_12) return kSm120Limits;
  return kNoClaim;
}

std::optional<Nvfp4DecodeDotSpec> MatchNvfp4DecodeDot(
    const HloScaledDotInstruction& dot,
    const se::GpuComputeCapability& gpu_version) {
  const Nvfp4DecodeLimits& limits = Nvfp4DecodeLimitsFor(gpu_version);
  if (!limits.claim) return std::nullopt;

  if (dot.operand_count() != 4) return std::nullopt;
  if (!IsNvfp4Operand(*dot.operand(0), *dot.operand(2)) ||
      !IsNvfp4Operand(*dot.operand(1), *dot.operand(3))) {
    return std::nullopt;
  }

  const DotDimensionNumbers& dnums = dot.dot_dimension_numbers();
  if (dnums.lhs_batch_dimensions_size() != dnums.rhs_batch_dimensions_size() ||
      dnums.lhs_batch_dimensions_size() > 1) {
    return std::nullopt;
  }
  if (dnums.lhs_contracting_dimensions_size() != 1 ||
      dnums.rhs_contracting_dimensions_size() != 1) {
    return std::nullopt;
  }
  const Shape& lhs = dot.operand(0)->shape();
  const Shape& rhs = dot.operand(1)->shape();
  const int64_t rank = 2 + dnums.lhs_batch_dimensions_size();
  if (lhs.dimensions().size() != rank || rhs.dimensions().size() != rank) {
    return std::nullopt;
  }

  const int64_t lhs_k_dim = dnums.lhs_contracting_dimensions(0);
  const int64_t rhs_k_dim = dnums.rhs_contracting_dimensions(0);
  auto free_dim = [&](const Shape& s, int64_t k_dim,
                      const auto& batch) -> std::optional<int64_t> {
    for (int64_t d = 0; d < s.dimensions().size(); ++d) {
      if (d == k_dim || absl::c_linear_search(batch, d)) continue;
      return d;
    }
    return std::nullopt;
  };
  std::optional<int64_t> lhs_free =
      free_dim(lhs, lhs_k_dim, dnums.lhs_batch_dimensions());
  std::optional<int64_t> rhs_free =
      free_dim(rhs, rhs_k_dim, dnums.rhs_batch_dimensions());
  if (!lhs_free.has_value() || !rhs_free.has_value()) return std::nullopt;

  const int64_t m = lhs.dimensions(*lhs_free);
  const int64_t k = lhs.dimensions(lhs_k_dim);
  const int64_t n = rhs.dimensions(*rhs_free);
  if (rhs.dimensions(rhs_k_dim) != k) return std::nullopt;

  if (m > limits.max_decode_rows) return std::nullopt;
  if (n < limits.min_weight_rows) return std::nullopt;

  if (k % kNvfp4ScaleBlock != 0) return std::nullopt;
  if (!HasNvfp4BlockK(k)) return std::nullopt;
  const Shape& lhs_scale = dot.operand(2)->shape();
  const Shape& rhs_scale = dot.operand(3)->shape();
  if (lhs_scale.dimensions().size() != rank ||
      rhs_scale.dimensions().size() != rank) {
    return std::nullopt;
  }
  const int64_t groups = k / kNvfp4ScaleBlock;
  auto has_group_extent = [&](const Shape& s) {
    return absl::c_linear_search(s.dimensions(), groups);
  };
  if (!has_group_extent(lhs_scale) || !has_group_extent(rhs_scale)) {
    return std::nullopt;
  }

  return Nvfp4DecodeDotSpec{/*batch=*/m, /*weight_rows=*/n, k,
                            /*weight_on_lhs=*/false};
}

std::optional<Nvfp4DecodeDotSpec> MatchNvfp4DecodeDotFusion(
    const HloFusionInstruction& fusion) {
  const HloComputation* computation = fusion.fused_instructions_computation();
  if (computation == nullptr ||
      !absl::StartsWith(computation->name(),
                        kNvfp4DecodeDotComputationPrefix)) {
    return std::nullopt;
  }
  const HloInstruction* root = nullptr;
  for (const HloInstruction* instr : computation->instructions()) {
    if (instr->opcode() == HloOpcode::kScaledDot) {
      if (root != nullptr) return std::nullopt;  // one per claimed fusion
      root = instr;
    }
  }
  if (root == nullptr) return std::nullopt;
  const DotDimensionNumbers& dnums = root->dot_dimension_numbers();
  if (dnums.lhs_contracting_dimensions_size() != 1 ||
      dnums.rhs_contracting_dimensions_size() != 1) {
    return std::nullopt;
  }
  const Shape& lhs = root->operand(0)->shape();
  const Shape& rhs = root->operand(1)->shape();
  if (lhs.dimensions().size() != rhs.dimensions().size() ||
      lhs.dimensions().size() < 2) {
    return std::nullopt;
  }
  const int64_t lhs_k_dim = dnums.lhs_contracting_dimensions(0);
  const int64_t rhs_k_dim = dnums.rhs_contracting_dimensions(0);
  auto free_extent = [](const Shape& s, int64_t k_dim, const auto& batch) {
    for (int64_t d = 0; d < s.dimensions().size(); ++d) {
      if (d == k_dim || absl::c_linear_search(batch, d)) continue;
      return s.dimensions(d);
    }
    return int64_t{0};
  };
  const int64_t m = free_extent(lhs, lhs_k_dim, dnums.lhs_batch_dimensions());
  const int64_t n = free_extent(rhs, rhs_k_dim, dnums.rhs_batch_dimensions());
  if (m == 0 || n == 0) return std::nullopt;
  const bool weight_on_lhs = m > n;
  return Nvfp4DecodeDotSpec{/*batch=*/weight_on_lhs ? n : m,
                            /*weight_rows=*/weight_on_lhs ? m : n,
                            /*k=*/lhs.dimensions(lhs_k_dim), weight_on_lhs};
}

Nvfp4DecodeDotConfig Nvfp4DecodeDotSeed(const Nvfp4DecodeDotSpec& spec,
                                        const Nvfp4DecodeLimits& limits) {
  return Nvfp4DecodeDotConfig{
      /*weight_tile=*/limits.seed.weight_tile,
      /*batch_tile=*/std::max(limits.min_batch_tile,
                              RoundUpToPowerOfTwo(spec.batch)),
      /*block_k=*/Nvfp4BlockKAtMost(spec.k, limits.seed.block_k),
      limits.seed.num_warps, limits.seed.num_stages, limits.swap};
}

std::optional<Nvfp4DecodeDotConfig> Nvfp4DecodeDotConfigFor(
    const HloScaledDotInstruction& dot,
    const se::GpuComputeCapability& gpu_version) {
  std::optional<Nvfp4DecodeDotSpec> spec =
      MatchNvfp4DecodeDot(dot, gpu_version);
  if (!spec.has_value()) return std::nullopt;
  const Nvfp4DecodeDotConfig config =
      Nvfp4DecodeDotSeed(*spec, Nvfp4DecodeLimitsFor(gpu_version));
  VLOG(1) << "nvfp4 decode dot seed for " << dot.name() << ": batch "
          << spec->batch << " x weight " << spec->weight_rows << " x k "
          << spec->k << " -> weight tile " << config.weight_tile
          << ", batch tile " << config.batch_tile << ", block_k "
          << config.block_k << (config.swap ? ", swapped" : ", unswapped");
  return config;
}

}  // namespace xla::gpu
