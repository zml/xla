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

#ifndef XLA_PJRT_STREAM_EXECUTOR_HOST_TO_DEVICE_H_
#define XLA_PJRT_STREAM_EXECUTOR_HOST_TO_DEVICE_H_

#include <cstdint>

#include "absl/status/status.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"

namespace xla {

bool IsSyclStreamExecutor(stream_executor::StreamExecutor* executor);

// PJRT upload paths that bypass TransferManager::TransferBufferToDevice() use
// this helper to share the same conservative SYCL H2D policy.
absl::Status SubmitStreamExecutorHostToDeviceCopy(
    stream_executor::Stream* stream, stream_executor::DeviceAddressBase* dst,
    const void* src, int64_t size);

}  // namespace xla

#endif  // XLA_PJRT_STREAM_EXECUTOR_HOST_TO_DEVICE_H_
