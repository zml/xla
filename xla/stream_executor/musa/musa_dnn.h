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

#ifndef XLA_STREAM_EXECUTOR_MUSA_MUSA_DNN_H_
#define XLA_STREAM_EXECUTOR_MUSA_MUSA_DNN_H_

#include <memory>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/dnn.h"
#include "xla/stream_executor/engine_options.h"
#include "xla/stream_executor/musa/musa_mudnn_api.h"
#include "xla/stream_executor/scratch_allocator.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"

namespace stream_executor::musa {

// StreamExecutor DNN implementation for the qualified muDNN shim contract.
// Unfused convolution uses XLA's shared GPU custom calls, thunks, descriptors,
// lazy runners, workspace buffers, and AOT serialization. Vendor C++ types are
// contained entirely in the separately linked shim.
class MusaDnn final : public dnn::DnnSupport {
 public:
  explicit MusaDnn(StreamExecutor* parent,
                   MusaMuDnnApi* api = GetMusaMuDnnApi());
  ~MusaDnn() override;

  MusaDnn(const MusaDnn&) = delete;
  MusaDnn& operator=(const MusaDnn&) = delete;

  absl::Status Init() override;
  absl::StatusOr<dnn::VersionInfo> GetVersion() override;

  absl::Status GetConvolveRunners(
      dnn::ConvolutionKind kind, dnn::DataType input_type,
      dnn::DataType output_type, Stream* stream,
      const dnn::BatchDescriptor& input_descriptor,
      DeviceAddressBase input_data,
      const dnn::FilterDescriptor& filter_descriptor,
      DeviceAddressBase filter_data,
      const dnn::BatchDescriptor& output_descriptor,
      DeviceAddressBase output_data,
      const dnn::ConvolutionDescriptor& convolution_descriptor,
      bool use_fallback, ScratchAllocator* scratch_allocator,
      const EngineOptions& engine_options,
      std::vector<std::unique_ptr<const dnn::ConvRunner>>* out_runners)
      override;

  absl::StatusOr<std::unique_ptr<const dnn::ConvRunner>> ConvolveRunnerFromDesc(
      Stream* stream, const dnn::AlgorithmDesc& algorithm_desc,
      dnn::ConvolutionKind kind, dnn::DataType input_type,
      dnn::DataType output_type, const dnn::BatchDescriptor& input_descriptor,
      const dnn::FilterDescriptor& filter_descriptor,
      const dnn::BatchDescriptor& output_descriptor,
      const dnn::ConvolutionDescriptor& convolution_descriptor) override;

  // Pooling remains on XLA's generic reduce-window path in C19. These pure
  // virtual hooks fail explicitly if a caller attempts direct DNN pooling.
  absl::Status DoPoolForward(dnn::DataType element_type, Stream* stream,
                             const dnn::PoolingDescriptor& pooling_dimensions,
                             const dnn::BatchDescriptor& input_dimensions,
                             DeviceAddressBase input_data,
                             const dnn::BatchDescriptor& output_dimensions,
                             DeviceAddressBase output_data,
                             ScratchAllocator* workspace_allocator) override;

  absl::Status DoPoolBackward(dnn::DataType element_type, Stream* stream,
                              const dnn::PoolingDescriptor& pooling_dimensions,
                              const dnn::BatchDescriptor& input_dimensions,
                              DeviceAddressBase input_data,
                              const dnn::BatchDescriptor& output_dimensions,
                              DeviceAddressBase output_data,
                              DeviceAddressBase input_diff_data,
                              DeviceAddressBase output_diff_data,
                              ScratchAllocator* workspace_allocator) override;

 private:
  StreamExecutor* parent_;
  MusaMuDnnApi* api_;
  absl::Mutex mutex_;
  void* handle_ ABSL_GUARDED_BY(mutex_) = nullptr;
};

}  // namespace stream_executor::musa

#endif  // XLA_STREAM_EXECUTOR_MUSA_MUSA_DNN_H_
