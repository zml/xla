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

#include "xla/pjrt/stream_executor_host_to_device.h"

#include "xla/stream_executor/sycl/sycl_platform_id.h"
#include "xla/tsl/platform/status_macros.h"

namespace xla {

bool IsSyclStreamExecutor(stream_executor::StreamExecutor* executor) {
  return executor->GetPlatform()->id() ==
         stream_executor::sycl::kSyclPlatformId;
}

absl::Status SubmitStreamExecutorHostToDeviceCopy(
    stream_executor::Stream* stream, stream_executor::DeviceAddressBase* dst,
    const void* src, int64_t size) {
  if (!IsSyclStreamExecutor(stream->parent())) {
    return stream->Memcpy(dst, src, size);
  }

  RETURN_IF_ERROR(stream->BlockHostUntilDone());
  return stream->parent()->SynchronousMemcpyH2D(src, size, dst);
}

}  // namespace xla
