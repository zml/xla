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

#include "xla/backends/gpu/collectives/oneccl_errors.h"

#include <string>

#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "oneapi/ccl.h"

namespace xla::gpu {

absl::Status ToStatus(onecclResult_t result, absl::string_view expr,
                      absl::string_view file, int line, onecclComm_t comm) {
  if (result == onecclSuccess) {
    return absl::OkStatus();
  }

  const char* error = onecclGetErrorString(result);
  const char* last_error = comm != nullptr ? onecclGetLastError(comm) : nullptr;
  return absl::InternalError(absl::StrFormat(
      "oneCCL call failed: %s at %s:%d returned %d (%s). Last oneCCL error: %s",
      expr, file, line, static_cast<int>(result),
      error != nullptr ? error : "<unknown>",
      last_error != nullptr ? last_error : "<unavailable>"));
}

}  // namespace xla::gpu
