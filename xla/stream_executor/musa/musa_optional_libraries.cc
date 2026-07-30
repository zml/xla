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

#include "xla/stream_executor/musa/musa_optional_libraries.h"

#include <algorithm>
#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "xla/tsl/platform/status_macros.h"
#include "xla/stream_executor/musa/musa_mublas_api.h"
#include "xla/stream_executor/musa/musa_mudnn_api.h"
#include "xla/stream_executor/musa/musa_mufft_api.h"
#include "xla/stream_executor/musa/musa_optional_library_abi.h"

namespace stream_executor::musa {
namespace internal {

absl::StatusOr<std::vector<MusaOptionalLibraryAbi>>
MergeMusaOptionalLibraryAbis(
    absl::Span<const std::vector<MusaOptionalLibraryAbi>> providers) {
  std::vector<MusaOptionalLibraryAbi> result;
  for (const std::vector<MusaOptionalLibraryAbi>& provider : providers) {
    result.insert(result.end(), provider.begin(), provider.end());
  }
  std::sort(
      result.begin(), result.end(),
      [](const MusaOptionalLibraryAbi& lhs, const MusaOptionalLibraryAbi& rhs) {
        return lhs.name < rhs.name;
      });
  for (size_t i = 0; i < result.size(); ++i) {
    if (result[i].name.empty() || result[i].abi_version.empty()) {
      return absl::FailedPreconditionError(
          "MUSA optional-library ABI has an empty name or version");
    }
    if (i != 0 && result[i - 1].name == result[i].name) {
      return absl::FailedPreconditionError(absl::StrCat(
          "duplicate MUSA optional-library ABI provider: ", result[i].name));
    }
  }
  return result;
}

}  // namespace internal

absl::StatusOr<std::vector<MusaOptionalLibraryAbi>>
GetAvailableMusaOptionalLibraryAbis() {
  TF_ASSIGN_OR_RETURN(std::vector<MusaOptionalLibraryAbi> mublas,
                      GetAvailableMusaMuBlasOptionalLibraryAbis());
  TF_ASSIGN_OR_RETURN(std::vector<MusaOptionalLibraryAbi> mufft,
                      GetAvailableMusaMuFftOptionalLibraryAbis());
  TF_ASSIGN_OR_RETURN(std::vector<MusaOptionalLibraryAbi> mudnn,
                      GetAvailableMusaMuDnnOptionalLibraryAbis());
  const std::vector<std::vector<MusaOptionalLibraryAbi>> providers = {
      std::move(mublas), std::move(mudnn), std::move(mufft)};
  return internal::MergeMusaOptionalLibraryAbis(providers);
}

}  // namespace stream_executor::musa
