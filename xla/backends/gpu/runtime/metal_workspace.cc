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

#include "xla/backends/gpu/runtime/metal_workspace.h"

#include <cstdint>
#include <limits>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "xla/tsl/platform/status_macros.h"
#include "xla/backends/gpu/runtime/metal_nvfp4_dispatch.h"

namespace xla::gpu {
namespace {

absl::StatusOr<int64_t> CheckedMultiply(int64_t lhs, int64_t rhs,
                                        const char* description) {
  if (lhs < 0 || rhs < 0) {
    return absl::InvalidArgumentError(
        absl::StrCat(description, " has a negative dimension"));
  }
  if (lhs != 0 && rhs > std::numeric_limits<int64_t>::max() / lhs) {
    return absl::ResourceExhaustedError(
        absl::StrCat(description, " byte size overflows int64"));
  }
  return lhs * rhs;
}

absl::StatusOr<int64_t> CheckedAdd(int64_t lhs, int64_t rhs,
                                   const char* description) {
  if (lhs < 0 || rhs < 0 || rhs > std::numeric_limits<int64_t>::max() - lhs) {
    return absl::ResourceExhaustedError(
        absl::StrCat(description, " byte offset overflows int64"));
  }
  return lhs + rhs;
}

absl::StatusOr<int64_t> AlignUp(int64_t value) {
  if (value < 0) {
    return absl::InvalidArgumentError("Metal workspace offset is negative");
  }
  const int64_t remainder = value % kMetalWorkspaceAlignment;
  if (remainder == 0) return value;
  return CheckedAdd(value, kMetalWorkspaceAlignment - remainder,
                    "Metal workspace alignment");
}

absl::StatusOr<MetalWorkspaceRegion> AppendRegion(int64_t size, int64_t* cursor,
                                                  const char* description) {
  TF_ASSIGN_OR_RETURN(const int64_t offset, AlignUp(*cursor));
  TF_ASSIGN_OR_RETURN(*cursor, CheckedAdd(offset, size, description));
  return MetalWorkspaceRegion{offset, size};
}

absl::Status ValidatePositiveDimensions(int64_t a, int64_t b, int64_t c,
                                        const char* description) {
  if (a <= 0 || b <= 0 || c <= 0) {
    return absl::InvalidArgumentError(
        absl::StrCat(description, " dimensions must be positive; got ", a, ", ",
                     b, ", ", c));
  }
  return absl::OkStatus();
}

absl::Status ValidateMetalInt32Dimension(int64_t value,
                                         const char* description) {
  if (value <= 0 || value > std::numeric_limits<int32_t>::max()) {
    return absl::InvalidArgumentError(absl::StrCat(
        description, " must fit a positive Metal int32; got ", value));
  }
  return absl::OkStatus();
}

}  // namespace

absl::StatusOr<MetalNvfp4WorkspaceLayout> GetMetalNvfp4SplitKWorkspaceLayout(
    int64_t split_k, int64_t m, int64_t n) {
  RETURN_IF_ERROR(
      ValidatePositiveDimensions(split_k, m, n, "NVFP4 split-K workspace"));
  TF_ASSIGN_OR_RETURN(const int64_t plane_elements,
                      CheckedMultiply(m, n, "NVFP4 split-K M*N stride"));
  if (plane_elements > std::numeric_limits<int32_t>::max()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "NVFP4 split-K M*N stride must fit Metal int32; got ", plane_elements));
  }
  TF_ASSIGN_OR_RETURN(
      int64_t elements,
      CheckedMultiply(split_k, plane_elements, "NVFP4 split-K workspace"));
  TF_ASSIGN_OR_RETURN(
      const int64_t bytes,
      CheckedMultiply(elements, sizeof(uint16_t), "NVFP4 split-K workspace"));
  MetalNvfp4WorkspaceLayout layout;
  layout.staging = MetalWorkspaceRegion{/*offset=*/0, /*size=*/bytes};
  layout.total_bytes = bytes;
  return layout;
}

absl::StatusOr<MetalNvfp4WorkspaceLayout> GetMetalNvfp4WorkspaceLayout(
    int64_t m, int64_t k, int64_t n, char arch_size, int arch_gen) {
  RETURN_IF_ERROR(ValidateMetalInt32Dimension(m, "NVFP4 M"));
  RETURN_IF_ERROR(ValidateMetalInt32Dimension(k, "NVFP4 K"));
  RETURN_IF_ERROR(ValidateMetalInt32Dimension(n, "NVFP4 N"));

  if (SelectNvfp4DensePath(m, k, n, arch_size, arch_gen) !=
      Nvfp4DensePath::kQmmSplitK) {
    return MetalNvfp4WorkspaceLayout{};
  }
  const int split_k = ComputeNvfp4QmmSplitK(
      static_cast<int>(m), static_cast<int>(n), static_cast<int>(k));
  if (split_k <= 1) return MetalNvfp4WorkspaceLayout{};
  return GetMetalNvfp4SplitKWorkspaceLayout(split_k, m, n);
}

absl::StatusOr<MetalMoeWorkspaceLayout> GetMetalMoeWorkspaceLayout(
    int64_t rows, int64_t k, int64_t n) {
  RETURN_IF_ERROR(
      ValidatePositiveDimensions(rows, k, n, "Metal MoE workspace"));

  TF_ASSIGN_OR_RETURN(
      const int64_t route_bytes,
      CheckedMultiply(rows, sizeof(int32_t), "Metal MoE route workspace"));
  TF_ASSIGN_OR_RETURN(int64_t x_elements,
                      CheckedMultiply(rows, k, "Metal MoE x workspace"));
  TF_ASSIGN_OR_RETURN(
      const int64_t x_bytes,
      CheckedMultiply(x_elements, sizeof(uint16_t), "Metal MoE x workspace"));
  TF_ASSIGN_OR_RETURN(int64_t out_elements,
                      CheckedMultiply(rows, n, "Metal MoE output workspace"));
  TF_ASSIGN_OR_RETURN(const int64_t out_bytes,
                      CheckedMultiply(out_elements, sizeof(uint16_t),
                                      "Metal MoE output workspace"));

  int64_t cursor = 0;
  MetalMoeWorkspaceLayout layout;
  TF_ASSIGN_OR_RETURN(layout.order,
                      AppendRegion(route_bytes, &cursor, "MoE order"));
  TF_ASSIGN_OR_RETURN(layout.expert_ids,
                      AppendRegion(route_bytes, &cursor, "MoE expert ids"));
  TF_ASSIGN_OR_RETURN(layout.x_sorted,
                      AppendRegion(x_bytes, &cursor, "MoE sorted x"));
  TF_ASSIGN_OR_RETURN(layout.out_sorted,
                      AppendRegion(out_bytes, &cursor, "MoE sorted output"));
  layout.total_bytes = cursor;
  return layout;
}

absl::StatusOr<int64_t> GetMetalNvfp4WorkspaceBytes(int64_t m, int64_t k,
                                                    int64_t n, char arch_size,
                                                    int arch_gen) {
  TF_ASSIGN_OR_RETURN(
      MetalNvfp4WorkspaceLayout layout,
      GetMetalNvfp4WorkspaceLayout(m, k, n, arch_size, arch_gen));
  return layout.total_bytes;
}

absl::StatusOr<int64_t> GetMetalMoeWorkspaceBytes(int64_t r, int64_t e,
                                                  int64_t k, int64_t n,
                                                  bool is_nvfp4) {
  RETURN_IF_ERROR(ValidateMetalInt32Dimension(r, "Metal MoE R"));
  RETURN_IF_ERROR(ValidateMetalInt32Dimension(e, "Metal MoE E"));
  RETURN_IF_ERROR(ValidateMetalInt32Dimension(k, "Metal MoE K"));
  RETURN_IF_ERROR(ValidateMetalInt32Dimension(n, "Metal MoE N"));
  // Only the sorted-prefill pipeline owns scratch; the per-row GEMV path reads
  // and writes the public buffers directly.
  if (!ShouldUseMetalMoeSortedPath(r, e, k, n, is_nvfp4)) return 0;
  TF_ASSIGN_OR_RETURN(MetalMoeWorkspaceLayout layout,
                      GetMetalMoeWorkspaceLayout(r, k, n));
  return layout.total_bytes;
}

}  // namespace xla::gpu
