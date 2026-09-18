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

#ifndef XLA_STREAM_EXECUTOR_MUSA_MUSA_CONTEXT_H_
#define XLA_STREAM_EXECUTOR_MUSA_MUSA_CONTEXT_H_

#include <memory>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/stream_executor/gpu/context.h"
#include "xla/stream_executor/musa/musa_driver.h"

namespace stream_executor::musa {

// Owns one reference to a device's MUSA primary context.
//
// MusaDriver outlives MusaContext. Production contexts use the process-wide
// driver singleton; accepting a driver pointer also permits hermetic lifecycle
// and activation tests.
class MusaContext final : public gpu::Context {
 public:
  static absl::StatusOr<std::unique_ptr<MusaContext>> Create(
      int device_ordinal, MusaDriver* driver);

  ~MusaContext() override;

  void SetActive() override;
  bool IsActive() const override;
  int device_ordinal() const override { return device_ordinal_; }
  absl::Status Synchronize() override;

  MUcontext context() const { return context_; }
  MUdevice device() const { return device_; }

  MusaContext(const MusaContext&) = delete;
  MusaContext& operator=(const MusaContext&) = delete;
  MusaContext(MusaContext&&) = delete;
  MusaContext& operator=(MusaContext&&) = delete;

 private:
  MusaContext(int device_ordinal, MUdevice device, MUcontext context,
              MusaDriver* driver)
      : device_ordinal_(device_ordinal),
        device_(device),
        context_(context),
        driver_(driver) {}

  const int device_ordinal_;
  const MUdevice device_;
  const MUcontext context_;
  MusaDriver* const driver_;
};

}  // namespace stream_executor::musa

#endif  // XLA_STREAM_EXECUTOR_MUSA_MUSA_CONTEXT_H_
