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

#include "xla/stream_executor/musa/musa_dnn.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "xla/tsl/platform/status_macros.h"
#include "xla/stream_executor/activate_context.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/dnn.h"
#include "xla/stream_executor/engine_options.h"
#include "xla/stream_executor/event_based_timer.h"
#include "xla/stream_executor/musa/mudnn_shim/mudnn_shim_abi.h"
#include "xla/stream_executor/musa/musa_mudnn_api.h"
#include "xla/stream_executor/musa/musa_platform_id.h"
#include "xla/stream_executor/platform/initialize.h"
#include "xla/stream_executor/plugin_registry.h"
#include "xla/stream_executor/scratch_allocator.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"

namespace stream_executor::musa {
namespace {

struct TensorConfig {
  XlaMusaMuDnnTensorFormat format;
  std::vector<int64_t> dimensions;
  std::vector<int64_t> strides;
};

absl::StatusOr<XlaMusaMuDnnDataType> ToMuDnnDataType(dnn::DataType type) {
  switch (type) {
    case dnn::DataType::kHalf:
      return XLA_MUSA_MUDNN_DATA_TYPE_F16;
    case dnn::DataType::kFloat:
      return XLA_MUSA_MUDNN_DATA_TYPE_F32;
    default:
      return absl::UnimplementedError(absl::StrCat(
          "muDNN 2.8.0 convolution supports only f16 and f32; got DNN type ",
          static_cast<int>(type)));
  }
}

absl::StatusOr<XlaMusaMuDnnConvolutionKind> ToMuDnnConvolutionKind(
    dnn::ConvolutionKind kind) {
  switch (kind) {
    case dnn::ConvolutionKind::FORWARD:
      return XLA_MUSA_MUDNN_CONVOLUTION_FORWARD;
    case dnn::ConvolutionKind::BACKWARD_DATA:
      return XLA_MUSA_MUDNN_CONVOLUTION_BACKWARD_DATA;
    case dnn::ConvolutionKind::BACKWARD_FILTER:
      return XLA_MUSA_MUDNN_CONVOLUTION_BACKWARD_FILTER;
    default:
      return absl::UnimplementedError(
          absl::StrCat("muDNN C19 supports only unfused convolution; got kind ",
                       static_cast<int>(kind)));
  }
}

absl::StatusOr<XlaMusaMuDnnAlgorithm> ToMuDnnAlgorithm(int64_t algorithm) {
  switch (algorithm) {
    case XLA_MUSA_MUDNN_ALGORITHM_IMPLICIT_GEMM:
    case XLA_MUSA_MUDNN_ALGORITHM_WINOGRAD_NONFUSED:
    case XLA_MUSA_MUDNN_ALGORITHM_GEMM:
      return static_cast<XlaMusaMuDnnAlgorithm>(algorithm);
    default:
      return absl::InvalidArgumentError(
          absl::StrCat("unknown normalized muDNN convolution algorithm ",
                       algorithm, "; expected one of 0, 2, or 3"));
  }
}

XlaMusaMuDnnTensorFormat ChannelsFirstFormat(int spatial_rank) {
  switch (spatial_rank) {
    case 1:
      return XLA_MUSA_MUDNN_TENSOR_FORMAT_NCW;
    case 2:
      return XLA_MUSA_MUDNN_TENSOR_FORMAT_NCHW;
    case 3:
      return XLA_MUSA_MUDNN_TENSOR_FORMAT_NCDHW;
    default:
      return XLA_MUSA_MUDNN_TENSOR_FORMAT_UNKNOWN;
  }
}

XlaMusaMuDnnTensorFormat ChannelsLastFormat(int spatial_rank) {
  switch (spatial_rank) {
    case 1:
      return XLA_MUSA_MUDNN_TENSOR_FORMAT_NWC;
    case 2:
      return XLA_MUSA_MUDNN_TENSOR_FORMAT_NHWC;
    case 3:
      return XLA_MUSA_MUDNN_TENSOR_FORMAT_NDHWC;
    default:
      return XLA_MUSA_MUDNN_TENSOR_FORMAT_UNKNOWN;
  }
}

absl::Status ValidateTensorConfig(const TensorConfig& config,
                                  const char* description) {
  if (config.dimensions.size() < 3 || config.dimensions.size() > 5 ||
      config.dimensions.size() != config.strides.size()) {
    return absl::InvalidArgumentError(
        absl::StrCat(description,
                     " must have matching rank-3 through rank-5 dimensions "
                     "and strides"));
  }
  for (size_t index = 0; index < config.dimensions.size(); ++index) {
    if (config.dimensions[index] <= 0 || config.strides[index] <= 0) {
      return absl::InvalidArgumentError(absl::StrCat(
          description, " has a non-positive dimension or stride at index ",
          index));
    }
  }
  return absl::OkStatus();
}

absl::StatusOr<TensorConfig> BatchTensorConfig(
    const dnn::BatchDescriptor& descriptor) {
  TensorConfig config;
  const dnn::DataLayout layout = descriptor.layout();
  config.dimensions = descriptor.full_dims(layout);
  config.strides = descriptor.full_strides(layout);
  switch (layout) {
    case dnn::DataLayout::kBatchDepthYX:
      config.format = ChannelsFirstFormat(descriptor.ndims());
      break;
    case dnn::DataLayout::kBatchYXDepth:
      config.format = ChannelsLastFormat(descriptor.ndims());
      break;
    default:
      return absl::UnimplementedError(absl::StrCat(
          "muDNN does not support batch layout ", static_cast<int>(layout)));
  }
  RETURN_IF_ERROR(ValidateTensorConfig(config, "muDNN batch tensor"));
  return config;
}

absl::StatusOr<TensorConfig> FilterTensorConfig(
    const dnn::FilterDescriptor& descriptor) {
  TensorConfig config;
  const dnn::FilterLayout layout = descriptor.layout();
  switch (layout) {
    case dnn::FilterLayout::kOutputInputYX:
      config.dimensions = descriptor.full_dims(layout);
      config.strides = descriptor.full_strides(layout);
      config.format = ChannelsFirstFormat(descriptor.ndims());
      break;
    case dnn::FilterLayout::kOutputYXInput:
      return absl::UnimplementedError(
          "muDNN 2.8.0 requires contiguous filters: XLA's channel-last "
          "convolution ABI stores OHWI, while muDNN requires HWIO; the MUSA "
          "layout assignment must select OIHW");
    default:
      return absl::UnimplementedError(absl::StrCat(
          "muDNN does not support filter layout ", static_cast<int>(layout)));
  }
  RETURN_IF_ERROR(ValidateTensorConfig(config, "muDNN filter tensor"));
  return config;
}

class ScopedMuDnnTensor {
 public:
  static absl::StatusOr<std::unique_ptr<ScopedMuDnnTensor>> Create(
      MusaMuDnnApi* api, XlaMusaMuDnnDataType type,
      const TensorConfig& config) {
    void* handle = nullptr;
    RETURN_IF_ERROR(api->CreateTensor(&handle));
    auto tensor =
        std::unique_ptr<ScopedMuDnnTensor>(new ScopedMuDnnTensor(api, handle));
    absl::Status status =
        api->ConfigureTensor(handle, type, config.format,
                             static_cast<int32_t>(config.dimensions.size()),
                             config.dimensions.data(), config.strides.data());
    if (!status.ok()) return status;
    return tensor;
  }

  ~ScopedMuDnnTensor() {
    if (handle_ != nullptr) {
      absl::Status status = api_->DestroyTensor(handle_);
      if (!status.ok()) VLOG(1) << "Failed to destroy muDNN tensor: " << status;
    }
  }

  void* handle() const { return handle_; }
  absl::Status SetAddress(DeviceAddressBase address) const {
    if (address.is_null()) {
      return absl::InvalidArgumentError("muDNN tensor address is null");
    }
    return api_->SetTensorAddress(handle_, address.opaque());
  }

 private:
  ScopedMuDnnTensor(MusaMuDnnApi* api, void* handle)
      : api_(api), handle_(handle) {}

  MusaMuDnnApi* api_;
  void* handle_;
};

class ScopedMuDnnConvolution {
 public:
  static absl::StatusOr<std::unique_ptr<ScopedMuDnnConvolution>> Create(
      MusaMuDnnApi* api, const dnn::ConvolutionDescriptor& descriptor) {
    if (descriptor.convolution_not_crosscorr()) {
      return absl::UnimplementedError(
          "muDNN C19 does not qualify reversed-window convolution");
    }
    if (descriptor.ndims() < 1 || descriptor.ndims() > 3) {
      return absl::UnimplementedError(
          "muDNN convolution supports one through three spatial dimensions");
    }
    void* handle = nullptr;
    RETURN_IF_ERROR(api->CreateConvolution(&handle));
    auto convolution = std::unique_ptr<ScopedMuDnnConvolution>(
        new ScopedMuDnnConvolution(api, handle));
    RETURN_IF_ERROR(api->ConfigureConvolution(
        handle, descriptor.ndims(), descriptor.padding().data(),
        descriptor.strides().data(), descriptor.dilations().data(),
        descriptor.group_count()));
    return convolution;
  }

  ~ScopedMuDnnConvolution() {
    if (handle_ != nullptr) {
      absl::Status status = api_->DestroyConvolution(handle_);
      if (!status.ok()) {
        VLOG(1) << "Failed to destroy muDNN convolution: " << status;
      }
    }
  }

  void* handle() const { return handle_; }

 private:
  ScopedMuDnnConvolution(MusaMuDnnApi* api, void* handle)
      : api_(api), handle_(handle) {}

  MusaMuDnnApi* api_;
  void* handle_;
};

struct WorkspaceState {
  uintptr_t base;
  uint64_t capacity;
  uint64_t offset;
};

XlaMusaMuDnnStatus AllocateWorkspace(void* user_data, uint64_t size_bytes,
                                     void** address) {
  if (user_data == nullptr || address == nullptr) {
    return XLA_MUSA_MUDNN_STATUS_INVALID_ARGUMENT;
  }
  auto* state = static_cast<WorkspaceState*>(user_data);
  if (size_bytes == 0) {
    *address = nullptr;
    return XLA_MUSA_MUDNN_STATUS_SUCCESS;
  }
  constexpr uint64_t kAlignment = 256;
  if (state->offset > std::numeric_limits<uint64_t>::max() - kAlignment + 1) {
    return XLA_MUSA_MUDNN_STATUS_OUT_OF_RANGE;
  }
  const uint64_t aligned = (state->offset + kAlignment - 1) & ~(kAlignment - 1);
  if (aligned > state->capacity || size_bytes > state->capacity - aligned ||
      state->base == 0) {
    return XLA_MUSA_MUDNN_STATUS_RESOURCE_EXHAUSTED;
  }
  if (aligned > std::numeric_limits<uintptr_t>::max() ||
      state->base > std::numeric_limits<uintptr_t>::max() -
                        static_cast<uintptr_t>(aligned)) {
    return XLA_MUSA_MUDNN_STATUS_OUT_OF_RANGE;
  }
  *address =
      reinterpret_cast<void*>(state->base + static_cast<uintptr_t>(aligned));
  state->offset = aligned + size_bytes;
  return XLA_MUSA_MUDNN_STATUS_SUCCESS;
}

void ReleaseWorkspace(void*, void*, uint64_t) {
  // The XLA thunk owns one stream-ordered scratch buffer for the full launch.
}

class MusaConvRunner final : public dnn::ConvRunner {
 public:
  static absl::StatusOr<std::unique_ptr<MusaConvRunner>> Create(
      StreamExecutor* parent, MusaMuDnnApi* api, absl::Mutex* mutex,
      void* handle, dnn::ConvolutionKind kind, dnn::DataType input_type,
      dnn::DataType output_type, const dnn::BatchDescriptor& input_descriptor,
      const dnn::FilterDescriptor& filter_descriptor,
      const dnn::BatchDescriptor& output_descriptor,
      const dnn::ConvolutionDescriptor& convolution_descriptor,
      XlaMusaMuDnnAlgorithm algorithm, bool allow_tf32,
      uint64_t workspace_size) {
    if (parent == nullptr || api == nullptr || mutex == nullptr ||
        handle == nullptr) {
      return absl::InvalidArgumentError(
          "muDNN runner requires a parent, API, mutex, and handle");
    }
    std::unique_ptr<ActivateContext> activation = parent->Activate();
    if (input_type != output_type) {
      return absl::UnimplementedError(
          "muDNN C19 requires identical convolution input and output types");
    }
    TF_ASSIGN_OR_RETURN(XlaMusaMuDnnDataType type, ToMuDnnDataType(input_type));
    TF_ASSIGN_OR_RETURN(TensorConfig input_config,
                        BatchTensorConfig(input_descriptor));
    TF_ASSIGN_OR_RETURN(TensorConfig filter_config,
                        FilterTensorConfig(filter_descriptor));
    TF_ASSIGN_OR_RETURN(TensorConfig output_config,
                        BatchTensorConfig(output_descriptor));
    TF_ASSIGN_OR_RETURN(std::unique_ptr<ScopedMuDnnTensor> input,
                        ScopedMuDnnTensor::Create(api, type, input_config));
    TF_ASSIGN_OR_RETURN(std::unique_ptr<ScopedMuDnnTensor> filter,
                        ScopedMuDnnTensor::Create(api, type, filter_config));
    TF_ASSIGN_OR_RETURN(std::unique_ptr<ScopedMuDnnTensor> output,
                        ScopedMuDnnTensor::Create(api, type, output_config));
    TF_ASSIGN_OR_RETURN(
        std::unique_ptr<ScopedMuDnnConvolution> convolution,
        ScopedMuDnnConvolution::Create(api, convolution_descriptor));
    return std::unique_ptr<MusaConvRunner>(new MusaConvRunner(
        parent, api, mutex, handle, kind, algorithm, allow_tf32, workspace_size,
        std::move(input), std::move(filter), std::move(output),
        std::move(convolution)));
  }

  std::string ToString() const override {
    return absl::StrCat("muDNN kind=", static_cast<int>(kind_), "; ",
                        MakeAlgorithmDesc().ToString());
  }

  size_t GetWorkspaceSize() const override {
    return static_cast<size_t>(workspace_size_);
  }

  absl::StatusOr<dnn::AlgorithmDesc> ToAlgorithmDesc() const override {
    return MakeAlgorithmDesc();
  }

  absl::Status QueryWorkspace(Stream* stream, DeviceAddressBase input_data,
                              DeviceAddressBase filter_data,
                              DeviceAddressBase output_data) {
    std::unique_ptr<ActivateContext> activation = parent_->Activate();
    absl::MutexLock lock(mutex_);
    RETURN_IF_ERROR(PrepareHandle(stream));
    RETURN_IF_ERROR(SetAddresses(input_data, filter_data, output_data));
    TensorRoles roles = Roles();
    uint64_t workspace_size = 0;
    RETURN_IF_ERROR(api_->GetWorkspaceSize(
        handle_, convolution_->handle(), MuDnnKind(), roles.output, roles.data,
        roles.filter, algorithm_, &workspace_size));
    if (workspace_size > std::numeric_limits<size_t>::max()) {
      return absl::OutOfRangeError(
          "muDNN convolution workspace exceeds host size_t");
    }
    workspace_size_ = workspace_size;
    return absl::OkStatus();
  }

  absl::Status operator()(Stream* stream,
                          dnn::ProfileResult* output_profile_result,
                          DeviceAddressBase scratch_memory,
                          DeviceAddressBase input_data,
                          DeviceAddressBase filter_data,
                          DeviceAddressBase output_data) const override {
    if (workspace_size_ > scratch_memory.size() ||
        (workspace_size_ != 0 && scratch_memory.is_null())) {
      return absl::ResourceExhaustedError(
          absl::StrCat("muDNN convolution requires ", workspace_size_,
                       " scratch bytes but received ", scratch_memory.size()));
    }
    std::unique_ptr<ActivateContext> activation = parent_->Activate();
    absl::MutexLock lock(mutex_);
    RETURN_IF_ERROR(PrepareHandle(stream));
    RETURN_IF_ERROR(SetAddresses(input_data, filter_data, output_data));
    std::unique_ptr<EventBasedTimer> timer;
    if (output_profile_result != nullptr) {
      TF_ASSIGN_OR_RETURN(timer,
                          stream->CreateEventBasedTimer(
                              output_profile_result->warmup_run_executed()));
    }
    TensorRoles roles = Roles();
    WorkspaceState workspace{
        reinterpret_cast<uintptr_t>(scratch_memory.opaque()),
        scratch_memory.size(), 0};
    const XlaMusaMuDnnWorkspaceAllocator allocator = {
        sizeof(XlaMusaMuDnnWorkspaceAllocator), 0, &workspace,
        AllocateWorkspace, ReleaseWorkspace};
    RETURN_IF_ERROR(api_->Convolve(handle_, convolution_->handle(), MuDnnKind(),
                                   roles.output, roles.data, roles.filter,
                                   algorithm_, &allocator));
    if (output_profile_result != nullptr) {
      TF_ASSIGN_OR_RETURN(absl::Duration duration, timer->GetElapsedDuration());
      output_profile_result->set_algorithm(MakeAlgorithmDesc());
      output_profile_result->set_elapsed_time_in_ms(
          absl::ToDoubleMilliseconds(duration));
      output_profile_result->set_scratch_size(workspace_size_);
    }
    return absl::OkStatus();
  }

 private:
  struct TensorRoles {
    void* output;
    const void* data;
    const void* filter;
  };

  MusaConvRunner(StreamExecutor* parent, MusaMuDnnApi* api, absl::Mutex* mutex,
                 void* handle, dnn::ConvolutionKind kind,
                 XlaMusaMuDnnAlgorithm algorithm, bool allow_tf32,
                 uint64_t workspace_size,
                 std::unique_ptr<ScopedMuDnnTensor> input,
                 std::unique_ptr<ScopedMuDnnTensor> filter,
                 std::unique_ptr<ScopedMuDnnTensor> output,
                 std::unique_ptr<ScopedMuDnnConvolution> convolution)
      : parent_(parent),
        api_(api),
        mutex_(mutex),
        handle_(handle),
        kind_(kind),
        algorithm_(algorithm),
        allow_tf32_(allow_tf32),
        workspace_size_(workspace_size),
        input_(std::move(input)),
        filter_(std::move(filter)),
        output_(std::move(output)),
        convolution_(std::move(convolution)) {}

  dnn::AlgorithmDesc MakeAlgorithmDesc() const {
    return dnn::AlgorithmDesc(static_cast<int64_t>(algorithm_), allow_tf32_,
                              workspace_size_);
  }

  XlaMusaMuDnnConvolutionKind MuDnnKind() const {
    switch (kind_) {
      case dnn::ConvolutionKind::FORWARD:
        return XLA_MUSA_MUDNN_CONVOLUTION_FORWARD;
      case dnn::ConvolutionKind::BACKWARD_DATA:
        return XLA_MUSA_MUDNN_CONVOLUTION_BACKWARD_DATA;
      case dnn::ConvolutionKind::BACKWARD_FILTER:
        return XLA_MUSA_MUDNN_CONVOLUTION_BACKWARD_FILTER;
      default:
        return 0;
    }
  }

  absl::Status PrepareHandle(Stream* stream) const {
    if (stream == nullptr || stream->parent() != parent_) {
      return absl::InvalidArgumentError(
          "muDNN convolution requires a stream from its parent executor");
    }
    RETURN_IF_ERROR(
        api_->SetStream(handle_, stream->platform_specific_handle().stream));
    return api_->SetAllowTf32(handle_, allow_tf32_);
  }

  absl::Status SetAddresses(DeviceAddressBase input_data,
                            DeviceAddressBase filter_data,
                            DeviceAddressBase output_data) const {
    RETURN_IF_ERROR(input_->SetAddress(input_data));
    RETURN_IF_ERROR(filter_->SetAddress(filter_data));
    return output_->SetAddress(output_data);
  }

  TensorRoles Roles() const {
    switch (kind_) {
      case dnn::ConvolutionKind::FORWARD:
        return {output_->handle(), input_->handle(), filter_->handle()};
      case dnn::ConvolutionKind::BACKWARD_DATA:
        return {input_->handle(), output_->handle(), filter_->handle()};
      case dnn::ConvolutionKind::BACKWARD_FILTER:
        return {filter_->handle(), input_->handle(), output_->handle()};
      default:
        return {nullptr, nullptr, nullptr};
    }
  }

  StreamExecutor* parent_;
  MusaMuDnnApi* api_;
  absl::Mutex* mutex_;
  void* handle_;
  dnn::ConvolutionKind kind_;
  XlaMusaMuDnnAlgorithm algorithm_;
  bool allow_tf32_;
  uint64_t workspace_size_;
  std::unique_ptr<ScopedMuDnnTensor> input_;
  std::unique_ptr<ScopedMuDnnTensor> filter_;
  std::unique_ptr<ScopedMuDnnTensor> output_;
  std::unique_ptr<ScopedMuDnnConvolution> convolution_;
};

// The vendor accepts workspace queries for DIRECT on configurations that
// subsequently fail execution. Do not advertise it to the shared autotuner.
constexpr std::array<XlaMusaMuDnnAlgorithm, 3> kQualifiedAlgorithms = {
    XLA_MUSA_MUDNN_ALGORITHM_IMPLICIT_GEMM,
    XLA_MUSA_MUDNN_ALGORITHM_WINOGRAD_NONFUSED,
    XLA_MUSA_MUDNN_ALGORITHM_GEMM};

}  // namespace

MusaDnn::MusaDnn(StreamExecutor* parent, MusaMuDnnApi* api)
    : parent_(parent), api_(api) {}

MusaDnn::~MusaDnn() {
  absl::MutexLock lock(&mutex_);
  if (handle_ != nullptr && api_ != nullptr) {
    std::unique_ptr<ActivateContext> activation =
        parent_ == nullptr ? nullptr : parent_->Activate();
    absl::Status status = api_->DestroyHandle(handle_);
    if (!status.ok()) VLOG(1) << "Failed to destroy muDNN handle: " << status;
    handle_ = nullptr;
  }
}

absl::Status MusaDnn::Init() {
  if (parent_ == nullptr || api_ == nullptr) {
    return absl::InvalidArgumentError(
        "muDNN support requires a parent executor and API");
  }
  RETURN_IF_ERROR(api_->Init());
  absl::MutexLock lock(&mutex_);
  if (handle_ != nullptr) return absl::OkStatus();
  std::unique_ptr<ActivateContext> activation = parent_->Activate();
  return api_->CreateHandle(parent_->device_ordinal(), &handle_);
}

absl::StatusOr<dnn::VersionInfo> MusaDnn::GetVersion() {
  XlaMusaMuDnnVersion version = {};
  RETURN_IF_ERROR(api_->GetVersion(&version));
  return dnn::VersionInfo(version.major, version.minor, version.patch);
}

absl::Status MusaDnn::GetConvolveRunners(
    dnn::ConvolutionKind kind, dnn::DataType input_type,
    dnn::DataType output_type, Stream* stream,
    const dnn::BatchDescriptor& input_descriptor, DeviceAddressBase input_data,
    const dnn::FilterDescriptor& filter_descriptor,
    DeviceAddressBase filter_data,
    const dnn::BatchDescriptor& output_descriptor,
    DeviceAddressBase output_data,
    const dnn::ConvolutionDescriptor& convolution_descriptor, bool use_fallback,
    ScratchAllocator* scratch_allocator, const EngineOptions& engine_options,
    std::vector<std::unique_ptr<const dnn::ConvRunner>>* out_runners) {
  (void)scratch_allocator;
  TF_ASSIGN_OR_RETURN(XlaMusaMuDnnConvolutionKind mudnn_kind,
                      ToMuDnnConvolutionKind(kind));
  (void)mudnn_kind;
  if (out_runners == nullptr) {
    return absl::InvalidArgumentError("muDNN runner output is null");
  }
  if (engine_options.require_command_buffer) {
    return absl::UnimplementedError(
        "muDNN C19 runners do not support explicit command buffers");
  }
  if (engine_options.require_determinism) {
    return absl::UnimplementedError(
        "muDNN 2.8.0 exposes no algorithm determinism metadata; "
        "deterministic convolution fails closed");
  }
  if (input_data.is_null() || filter_data.is_null() || output_data.is_null()) {
    return absl::InvalidArgumentError(
        "muDNN algorithm discovery requires non-null convolution buffers");
  }

  void* handle = nullptr;
  {
    absl::MutexLock lock(&mutex_);
    handle = handle_;
  }
  if (handle == nullptr) {
    return absl::FailedPreconditionError("muDNN is not initialized");
  }

  const size_t initial_size = out_runners->size();
  const size_t algorithm_count = use_fallback ? 1 : kQualifiedAlgorithms.size();
  for (size_t index = 0; index < algorithm_count; ++index) {
    const XlaMusaMuDnnAlgorithm algorithm = kQualifiedAlgorithms[index];
    TF_ASSIGN_OR_RETURN(
        std::unique_ptr<MusaConvRunner> runner,
        MusaConvRunner::Create(parent_, api_, &mutex_, handle, kind, input_type,
                               output_type, input_descriptor, filter_descriptor,
                               output_descriptor, convolution_descriptor,
                               algorithm, engine_options.allow_tf32,
                               /*workspace_size=*/0));
    absl::Status workspace_status =
        runner->QueryWorkspace(stream, input_data, filter_data, output_data);
    if (absl::IsUnimplemented(workspace_status)) continue;
    RETURN_IF_ERROR(workspace_status);
    out_runners->push_back(std::move(runner));
  }
  if (out_runners->size() == initial_size) {
    return absl::UnimplementedError(
        "muDNN found no supported algorithm for the convolution");
  }
  return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<const dnn::ConvRunner>>
MusaDnn::ConvolveRunnerFromDesc(
    Stream* stream, const dnn::AlgorithmDesc& algorithm_desc,
    dnn::ConvolutionKind kind, dnn::DataType input_type,
    dnn::DataType output_type, const dnn::BatchDescriptor& input_descriptor,
    const dnn::FilterDescriptor& filter_descriptor,
    const dnn::BatchDescriptor& output_descriptor,
    const dnn::ConvolutionDescriptor& convolution_descriptor) {
  (void)stream;
  TF_ASSIGN_OR_RETURN(XlaMusaMuDnnConvolutionKind mudnn_kind,
                      ToMuDnnConvolutionKind(kind));
  (void)mudnn_kind;
  TF_ASSIGN_OR_RETURN(XlaMusaMuDnnAlgorithm algorithm,
                      ToMuDnnAlgorithm(algorithm_desc.algo_id()));
  const std::optional<uint64_t> workspace_size =
      algorithm_desc.workspace_size();
  if (!workspace_size.has_value()) {
    return absl::InvalidArgumentError(
        "muDNN runner reconstruction requires an explicit workspace size");
  }
  void* handle = nullptr;
  {
    absl::MutexLock lock(&mutex_);
    handle = handle_;
  }
  if (handle == nullptr) {
    return absl::FailedPreconditionError("muDNN is not initialized");
  }
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<MusaConvRunner> runner,
      MusaConvRunner::Create(parent_, api_, &mutex_, handle, kind, input_type,
                             output_type, input_descriptor, filter_descriptor,
                             output_descriptor, convolution_descriptor,
                             algorithm, algorithm_desc.tensor_ops_enabled(),
                             *workspace_size));
  return std::unique_ptr<const dnn::ConvRunner>(std::move(runner));
}

absl::Status MusaDnn::DoPoolForward(dnn::DataType, Stream*,
                                    const dnn::PoolingDescriptor&,
                                    const dnn::BatchDescriptor&,
                                    DeviceAddressBase,
                                    const dnn::BatchDescriptor&,
                                    DeviceAddressBase, ScratchAllocator*) {
  return absl::UnimplementedError(
      "muDNN pooling is not routed in C19; XLA uses generic reduce-window");
}

absl::Status MusaDnn::DoPoolBackward(dnn::DataType, Stream*,
                                     const dnn::PoolingDescriptor&,
                                     const dnn::BatchDescriptor&,
                                     DeviceAddressBase,
                                     const dnn::BatchDescriptor&,
                                     DeviceAddressBase, DeviceAddressBase,
                                     DeviceAddressBase, ScratchAllocator*) {
  return absl::UnimplementedError(
      "muDNN pooling is not routed in C19; XLA uses generic reduce-window");
}

void InitializeMusaDnn() {
  PluginRegistry* registry = PluginRegistry::Instance();
  if (registry->HasFactory(kMusaPlatformId, PluginKind::kDnn)) return;
  absl::Status status = registry->RegisterFactory<PluginRegistry::DnnFactory>(
      kMusaPlatformId, "muDNN", [](StreamExecutor* parent) -> dnn::DnnSupport* {
        auto dnn = std::make_unique<MusaDnn>(parent);
        absl::Status status = dnn->Init();
        if (!status.ok()) {
          if (absl::IsNotFound(status)) {
            VLOG(1) << "Optional muDNN shim is unavailable: " << status;
          } else {
            LOG(ERROR) << "Unable to initialize optional muDNN support: "
                       << status;
          }
          return nullptr;
        }
        return dnn.release();
      });
  if (!status.ok()) {
    LOG(ERROR) << "Unable to register optional muDNN factory: " << status;
  }
}

}  // namespace stream_executor::musa

STREAM_EXECUTOR_REGISTER_MODULE_INITIALIZER(register_musa_mudnn, {
  stream_executor::musa::InitializeMusaDnn();
});
