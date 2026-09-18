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

#ifndef XLA_STREAM_EXECUTOR_MUSA_MUSA_FFT_H_
#define XLA_STREAM_EXECUTOR_MUSA_MUSA_FFT_H_

#include <cstddef>
#include <cstdint>
#include <memory>

#include "absl/status/status.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/fft.h"
#include "xla/stream_executor/musa/musa_mufft_api.h"
#include "xla/stream_executor/scratch_allocator.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"

namespace stream_executor::musa {

// A muFFT plan whose vendor handle is owned behind the SDK-free shim API. The
// plan is context-bound, and the shared GPU FftThunk serializes host access to
// it while allowing the native stream to be rebound for every execution.
class MusaFftPlan final : public fft::Plan {
 public:
  MusaFftPlan(StreamExecutor* parent, MusaMuFftApi* api);
  MusaFftPlan(const MusaFftPlan&) = delete;
  MusaFftPlan& operator=(const MusaFftPlan&) = delete;
  ~MusaFftPlan() override;

  absl::Status Initialize(Stream* stream, int rank,
                          const uint64_t* element_count,
                          const uint64_t* input_embed, uint64_t input_stride,
                          uint64_t input_distance, const uint64_t* output_embed,
                          uint64_t output_stride, uint64_t output_distance,
                          fft::Type type, int batch_count,
                          ScratchAllocator* scratch_allocator);

  absl::Status UpdateScratchAllocator(Stream* stream,
                                      ScratchAllocator* scratch_allocator);

  void* handle() const { return handle_; }
  fft::Type type() const { return type_; }
  ScratchAllocator* scratch_allocator() const { return scratch_allocator_; }
  const absl::Status& status() const { return status_; }

 private:
  StreamExecutor* parent_;
  MusaMuFftApi* api_;
  void* handle_ = nullptr;
  fft::Type type_ = fft::Type::kInvalid;
  DeviceAddress<uint8_t> scratch_;
  uint64_t scratch_size_bytes_ = 0;
  ScratchAllocator* scratch_allocator_ = nullptr;
  bool initialized_ = false;
  absl::Status status_ =
      absl::FailedPreconditionError("muFFT plan is not initialized");
};

// muFFT implementation of the platform-neutral StreamExecutor FFT contract.
// The vendor DSO is optional and is resolved only when a plan is first used.
class MusaFft final : public fft::FftSupport {
 public:
  explicit MusaFft(StreamExecutor* parent,
                   MusaMuFftApi* api = GetMusaMuFftApi())
      : parent_(parent), api_(api) {}
  MusaFft(const MusaFft&) = delete;
  MusaFft& operator=(const MusaFft&) = delete;
  ~MusaFft() override = default;

  TENSORFLOW_STREAM_EXECUTOR_GPU_FFT_SUPPORT_OVERRIDES

 private:
  template <typename InputT, typename OutputT>
  bool Execute(Stream* stream, fft::Plan* plan,
               const DeviceAddress<InputT>& input,
               DeviceAddress<OutputT>* output, fft::Type expected_type,
               bool preserve_input);

  bool PrepareExecution(Stream* stream, MusaFftPlan* plan,
                        const DeviceAddressBase& input,
                        DeviceAddressBase* output);

  StreamExecutor* parent_;
  MusaMuFftApi* api_;
};

void InitializeMusaFft();

}  // namespace stream_executor::musa

#endif  // XLA_STREAM_EXECUTOR_MUSA_MUSA_FFT_H_
