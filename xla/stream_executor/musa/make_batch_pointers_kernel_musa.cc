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

#include <cstddef>
#include <cstdint>

#include "absl/types/span.h"
#include "xla/stream_executor/gpu/gpu_kernel_registry.h"
#include "xla/stream_executor/gpu/make_batch_pointers_kernel.h"
#include "xla/stream_executor/kernel_spec.h"
#include "xla/stream_executor/musa/make_batch_pointers_kernel_musa_mubin.h"
#include "xla/stream_executor/musa/musa_platform_id.h"

namespace stream_executor::musa {
namespace {

KernelLoaderSpec MakeBatchPointersKernelSpec(size_t arity) {
  const FileToc* toc = make_batch_pointers_kernel_musa_mubin_create();
  const absl::Span<const uint8_t> mubin(
      reinterpret_cast<const uint8_t*>(toc[0].data), toc[0].size);
  return KernelLoaderSpec::CreateMusaMubinInMemorySpec(
      mubin, "make_batch_pointers", arity);
}

}  // namespace
}  // namespace stream_executor::musa

GPU_KERNEL_REGISTRY_REGISTER_KERNEL_STATICALLY(
    MakeBatchPointersKernelMusa,
    stream_executor::gpu::MakeBatchPointersKernel,
    stream_executor::musa::kMusaPlatformId,
    stream_executor::musa::MakeBatchPointersKernelSpec);
