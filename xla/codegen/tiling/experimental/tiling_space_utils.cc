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

#include "xla/codegen/tiling/experimental/tiling_space_utils.h"

#include <cstdint>
#include <utility>
#include <vector>

#include "absl/functional/function_ref.h"
#include "absl/numeric/bits.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"

namespace xla {

namespace {

// The possible tiles sizes for one dimension.
absl::StatusOr<std::vector<int64_t>> PossibleTileSizesForOneDimension(
    int64_t dim_size) {
  if (dim_size < 0) {
    return absl::InvalidArgumentError("Dimension size must be non-negative.");
  }
  std::vector<int64_t> result;
  if (dim_size == 0) {
    result.push_back(0);
    return result;
  }

  result.reserve(absl::bit_width(static_cast<uint64_t>(dim_size)));
  for (int64_t tile_size = 1; tile_size < dim_size; tile_size *= 2) {
    result.push_back(tile_size);
  }
  result.push_back(dim_size);
  return result;
}

absl::Status EnumerateFlatTilings(
    absl::Span<const std::vector<int64_t>> possible_tile_sizes,
    int64_t dimension, FlatTiling& current,
    absl::FunctionRef<absl::Status(absl::Span<const int64_t>)> callback) {
  if (dimension == possible_tile_sizes.size()) {
    return callback(current);
  }

  for (int64_t tile_size : possible_tile_sizes[dimension]) {
    current[dimension] = tile_size;
    ABSL_RETURN_IF_ERROR(EnumerateFlatTilings(
        possible_tile_sizes, dimension + 1, current, callback));
  }
  return absl::OkStatus();
}

}  // namespace

absl::Status ForEachFlatTilingForInputSpace(
    absl::Span<const int64_t> input_space,
    absl::FunctionRef<absl::Status(absl::Span<const int64_t>)> callback) {
  std::vector<std::vector<int64_t>> possible_tile_sizes;
  possible_tile_sizes.reserve(input_space.size());
  for (int64_t parameter_size : input_space) {
    ABSL_ASSIGN_OR_RETURN(auto sizes,
                          PossibleTileSizesForOneDimension(parameter_size));
    possible_tile_sizes.push_back(std::move(sizes));
  }

  FlatTiling current(input_space.size(), 0);
  return EnumerateFlatTilings(possible_tile_sizes, /*dimension=*/0, current,
                              callback);
}

absl::StatusOr<std::vector<FlatTiling>> GetFlatTilingsForInputSpace(
    absl::Span<const int64_t> input_space) {
  std::vector<FlatTiling> flat_tilings;
  ABSL_RETURN_IF_ERROR(ForEachFlatTilingForInputSpace(
      input_space, [&](absl::Span<const int64_t> flat_tiling) {
        flat_tilings.emplace_back(flat_tiling.begin(), flat_tiling.end());
        return absl::OkStatus();
      }));

  return flat_tilings;
}

}  // namespace xla
