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

#ifndef XLA_BACKENDS_GPU_COLLECTIVES_ONECCL_ERRORS_H_
#define XLA_BACKENDS_GPU_COLLECTIVES_ONECCL_ERRORS_H_

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "oneapi/ccl.h"

namespace xla::gpu {

absl::Status ToStatus(onecclResult_t result, absl::string_view expr,
                      absl::string_view file, int line,
                      onecclComm_t comm = nullptr);

}  // namespace xla::gpu

#define XLA_ONECCL_STATUS(expr) \
  ::xla::gpu::ToStatus((expr), #expr, __FILE__, __LINE__)

#define XLA_ONECCL_STATUS_WITH_COMM(expr, comm) \
  ::xla::gpu::ToStatus((expr), #expr, __FILE__, __LINE__, (comm))

#define XLA_ONECCL_RETURN_IF_ERROR(expr)          \
  do {                                            \
    absl::Status status = XLA_ONECCL_STATUS(expr); \
    if (!status.ok()) {                           \
      return status;                              \
    }                                             \
  } while (false)

#define XLA_ONECCL_RETURN_IF_ERROR_WITH_COMM(expr, comm)         \
  do {                                                           \
    absl::Status status = XLA_ONECCL_STATUS_WITH_COMM(expr, comm); \
    if (!status.ok()) {                                          \
      return status;                                             \
    }                                                            \
  } while (false)

#define XLA_ONECCL_LOG_IF_ERROR(expr)                    \
  do {                                                   \
    absl::Status status = XLA_ONECCL_STATUS(expr);       \
    if (!status.ok()) {                                  \
      LOG(ERROR) << status;                              \
    }                                                    \
  } while (false)

#define XLA_ONECCL_LOG_IF_ERROR_WITH_COMM(expr, comm)          \
  do {                                                         \
    absl::Status status = XLA_ONECCL_STATUS_WITH_COMM(expr, comm); \
    if (!status.ok()) {                                        \
      LOG(ERROR) << status;                                    \
    }                                                          \
  } while (false)

#endif  // XLA_BACKENDS_GPU_COLLECTIVES_ONECCL_ERRORS_H_
