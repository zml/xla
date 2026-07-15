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

#include "xla/stream_executor/musa/musa_status.h"

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"

namespace stream_executor::musa {

absl::Status ToStatus(int result, absl::string_view expr,
                      absl::string_view error_string) {
  if (result == 0) {
    return absl::OkStatus();
  }
  if (error_string.empty()) {
    return absl::InternalError(
        absl::StrCat(expr, " failed with MUSA error ", result));
  }
  return absl::InternalError(absl::StrCat(
      expr, " failed with MUSA error ", result, ": ", error_string));
}

}  // namespace stream_executor::musa
