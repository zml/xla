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

#ifndef XLA_STREAM_EXECUTOR_MUSA_MUSA_MODULE_USE_TRACKER_H_
#define XLA_STREAM_EXECUTOR_MUSA_MUSA_MODULE_USE_TRACKER_H_

#include <memory>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/stream_executor/kernel_args.h"

namespace stream_executor::musa {

class MusaModule;

// A stream-side lifetime boundary used by MusaKernel without depending on the
// concrete MusaStream implementation. A successful launch is retired by an
// ordered stream callback. A launch whose completion is ambiguous is orphaned
// until the owning executor successfully synchronizes its context.
class MusaModuleUseTracker {
 public:
  virtual ~MusaModuleUseTracker() = default;

  virtual absl::Status RecordModuleUse(std::shared_ptr<MusaModule> module) = 0;
  virtual void OrphanModuleUse(std::shared_ptr<MusaModule> module) = 0;

  virtual absl::StatusOr<const KernelArgsPackedArrayBase*>
  RetainGraphCaptureKernelArguments(
      const KernelArgsPackedArrayBase& arguments) {
    return &arguments;
  }
};

}  // namespace stream_executor::musa

#endif  // XLA_STREAM_EXECUTOR_MUSA_MUSA_MODULE_USE_TRACKER_H_
