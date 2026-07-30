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
#include <functional>
#include <limits>
#include <memory>
#include <new>
#include <utility>

#include "mudnn.h"
#include "xla/stream_executor/musa/mudnn_shim/mudnn_shim_abi.h"

namespace {

namespace mudnn = ::musa::dnn;

static_assert(MUDNN_VERSION_MAJOR == 2);
static_assert(MUDNN_VERSION_MINOR == 8);
static_assert(MUDNN_VERSION_PATCH == 0);
static_assert(sizeof(size_t) <= sizeof(uint64_t));

XlaMusaMuDnnStatus ToShimStatus(mudnn::Status status) {
  switch (status) {
    case mudnn::Status::SUCCESS:
      return XLA_MUSA_MUDNN_STATUS_SUCCESS;
    case mudnn::Status::INVALID_PARAMETER:
      return XLA_MUSA_MUDNN_STATUS_INVALID_ARGUMENT;
    case mudnn::Status::NOT_INITIALIZED:
      return XLA_MUSA_MUDNN_STATUS_FAILED_PRECONDITION;
    case mudnn::Status::ALLOC_FAILED:
      return XLA_MUSA_MUDNN_STATUS_RESOURCE_EXHAUSTED;
    case mudnn::Status::NOT_SUPPORTED:
    case mudnn::Status::ARCH_MISMATCH:
      return XLA_MUSA_MUDNN_STATUS_NOT_SUPPORTED;
    case mudnn::Status::INTERNAL_ERROR:
    case mudnn::Status::EXECUTION_FAILED:
    default:
      return XLA_MUSA_MUDNN_STATUS_VENDOR_ERROR;
  }
}

template <typename Function>
XlaMusaMuDnnStatus GuardVendorCall(Function&& function) noexcept {
  try {
    return std::forward<Function>(function)();
  } catch (const std::bad_alloc&) {
    return XLA_MUSA_MUDNN_STATUS_RESOURCE_EXHAUSTED;
  } catch (...) {
    return XLA_MUSA_MUDNN_STATUS_VENDOR_ERROR;
  }
}

bool IsKnownShimStatus(XlaMusaMuDnnStatus status) {
  return status >= XLA_MUSA_MUDNN_STATUS_SUCCESS &&
         status <= XLA_MUSA_MUDNN_STATUS_VENDOR_ERROR;
}

mudnn::Handle* AsHandle(void* handle) {
  return static_cast<mudnn::Handle*>(handle);
}

mudnn::Tensor* AsTensor(void* tensor) {
  return static_cast<mudnn::Tensor*>(tensor);
}

const mudnn::Tensor* AsTensor(const void* tensor) {
  return static_cast<const mudnn::Tensor*>(tensor);
}

mudnn::Convolution* AsConvolution(void* convolution) {
  return static_cast<mudnn::Convolution*>(convolution);
}

XlaMusaMuDnnStatus GetVersion(XlaMusaMuDnnVersion* version) {
  if (version == nullptr) {
    return XLA_MUSA_MUDNN_STATUS_INVALID_ARGUMENT;
  }
  return GuardVendorCall([&]() -> XlaMusaMuDnnStatus {
    if (mudnn::GetVersion() != static_cast<size_t>(MUDNN_VERSION)) {
      return XLA_MUSA_MUDNN_STATUS_FAILED_PRECONDITION;
    }
    *version = XlaMusaMuDnnVersion{
        MUDNN_VERSION_MAJOR,
        MUDNN_VERSION_MINOR,
        MUDNN_VERSION_PATCH,
    };
    return XLA_MUSA_MUDNN_STATUS_SUCCESS;
  });
}

XlaMusaMuDnnStatus CreateHandle(int32_t device_ordinal, void** handle) {
  if (handle == nullptr || device_ordinal < 0) {
    return XLA_MUSA_MUDNN_STATUS_INVALID_ARGUMENT;
  }
  *handle = nullptr;
  return GuardVendorCall([&]() -> XlaMusaMuDnnStatus {
    std::unique_ptr<mudnn::Handle> result =
        std::make_unique<mudnn::Handle>(device_ordinal);
    *handle = result.release();
    return XLA_MUSA_MUDNN_STATUS_SUCCESS;
  });
}

XlaMusaMuDnnStatus DestroyHandle(void* handle) {
  if (handle == nullptr) {
    return XLA_MUSA_MUDNN_STATUS_INVALID_ARGUMENT;
  }
  return GuardVendorCall([&]() -> XlaMusaMuDnnStatus {
    delete AsHandle(handle);
    return XLA_MUSA_MUDNN_STATUS_SUCCESS;
  });
}

XlaMusaMuDnnStatus SetStream(void* handle, void* stream) {
  if (handle == nullptr) {
    return XLA_MUSA_MUDNN_STATUS_INVALID_ARGUMENT;
  }
  return GuardVendorCall([&]() -> XlaMusaMuDnnStatus {
    return ToShimStatus(
        AsHandle(handle)->SetStream(reinterpret_cast<musaStream_t>(stream)));
  });
}

XlaMusaMuDnnStatus SetAllowTf32(void* handle, uint32_t allow_tf32) {
  if (handle == nullptr || allow_tf32 > 1) {
    return XLA_MUSA_MUDNN_STATUS_INVALID_ARGUMENT;
  }
  return GuardVendorCall([&]() -> XlaMusaMuDnnStatus {
    return ToShimStatus(AsHandle(handle)->SetAllowTF32(allow_tf32 != 0));
  });
}

XlaMusaMuDnnStatus CreateTensor(void** tensor) {
  if (tensor == nullptr) {
    return XLA_MUSA_MUDNN_STATUS_INVALID_ARGUMENT;
  }
  *tensor = nullptr;
  return GuardVendorCall([&]() -> XlaMusaMuDnnStatus {
    std::unique_ptr<mudnn::Tensor> result = std::make_unique<mudnn::Tensor>();
    *tensor = result.release();
    return XLA_MUSA_MUDNN_STATUS_SUCCESS;
  });
}

XlaMusaMuDnnStatus DestroyTensor(void* tensor) {
  if (tensor == nullptr) {
    return XLA_MUSA_MUDNN_STATUS_INVALID_ARGUMENT;
  }
  return GuardVendorCall([&]() -> XlaMusaMuDnnStatus {
    delete AsTensor(tensor);
    return XLA_MUSA_MUDNN_STATUS_SUCCESS;
  });
}

XlaMusaMuDnnStatus ToNativeDataType(XlaMusaMuDnnDataType data_type,
                                    mudnn::Tensor::Type* native_type) {
  if (native_type == nullptr) {
    return XLA_MUSA_MUDNN_STATUS_INVALID_ARGUMENT;
  }
  switch (data_type) {
    case XLA_MUSA_MUDNN_DATA_TYPE_F16:
      *native_type = mudnn::Tensor::Type::HALF;
      return XLA_MUSA_MUDNN_STATUS_SUCCESS;
    case XLA_MUSA_MUDNN_DATA_TYPE_BF16:
      *native_type = mudnn::Tensor::Type::BFLOAT16;
      return XLA_MUSA_MUDNN_STATUS_SUCCESS;
    case XLA_MUSA_MUDNN_DATA_TYPE_F32:
      *native_type = mudnn::Tensor::Type::FLOAT;
      return XLA_MUSA_MUDNN_STATUS_SUCCESS;
    default:
      return XLA_MUSA_MUDNN_STATUS_NOT_SUPPORTED;
  }
}

XlaMusaMuDnnStatus ToNativeTensorFormat(XlaMusaMuDnnTensorFormat format,
                                        int32_t rank,
                                        mudnn::Tensor::Format* native_format) {
  if (native_format == nullptr) {
    return XLA_MUSA_MUDNN_STATUS_INVALID_ARGUMENT;
  }
  switch (format) {
    case XLA_MUSA_MUDNN_TENSOR_FORMAT_UNKNOWN:
      *native_format = mudnn::Tensor::Format::UNKNOWN;
      return XLA_MUSA_MUDNN_STATUS_SUCCESS;
    case XLA_MUSA_MUDNN_TENSOR_FORMAT_NCW:
      if (rank != 3) return XLA_MUSA_MUDNN_STATUS_INVALID_ARGUMENT;
      *native_format = mudnn::Tensor::Format::NCW;
      return XLA_MUSA_MUDNN_STATUS_SUCCESS;
    case XLA_MUSA_MUDNN_TENSOR_FORMAT_NWC:
      if (rank != 3) return XLA_MUSA_MUDNN_STATUS_INVALID_ARGUMENT;
      *native_format = mudnn::Tensor::Format::NWC;
      return XLA_MUSA_MUDNN_STATUS_SUCCESS;
    case XLA_MUSA_MUDNN_TENSOR_FORMAT_NCHW:
      if (rank != 4) return XLA_MUSA_MUDNN_STATUS_INVALID_ARGUMENT;
      *native_format = mudnn::Tensor::Format::NCHW;
      return XLA_MUSA_MUDNN_STATUS_SUCCESS;
    case XLA_MUSA_MUDNN_TENSOR_FORMAT_NHWC:
      if (rank != 4) return XLA_MUSA_MUDNN_STATUS_INVALID_ARGUMENT;
      *native_format = mudnn::Tensor::Format::NHWC;
      return XLA_MUSA_MUDNN_STATUS_SUCCESS;
    case XLA_MUSA_MUDNN_TENSOR_FORMAT_HWCN:
      if (rank != 4) return XLA_MUSA_MUDNN_STATUS_INVALID_ARGUMENT;
      *native_format = mudnn::Tensor::Format::HWCN;
      return XLA_MUSA_MUDNN_STATUS_SUCCESS;
    case XLA_MUSA_MUDNN_TENSOR_FORMAT_NCDHW:
      if (rank != 5) return XLA_MUSA_MUDNN_STATUS_INVALID_ARGUMENT;
      *native_format = mudnn::Tensor::Format::NCDHW;
      return XLA_MUSA_MUDNN_STATUS_SUCCESS;
    case XLA_MUSA_MUDNN_TENSOR_FORMAT_NDHWC:
      if (rank != 5) return XLA_MUSA_MUDNN_STATUS_INVALID_ARGUMENT;
      *native_format = mudnn::Tensor::Format::NDHWC;
      return XLA_MUSA_MUDNN_STATUS_SUCCESS;
    case XLA_MUSA_MUDNN_TENSOR_FORMAT_DHWCN:
      if (rank != 5) return XLA_MUSA_MUDNN_STATUS_INVALID_ARGUMENT;
      *native_format = mudnn::Tensor::Format::DHWCN;
      return XLA_MUSA_MUDNN_STATUS_SUCCESS;
    default:
      return XLA_MUSA_MUDNN_STATUS_NOT_SUPPORTED;
  }
}

XlaMusaMuDnnStatus ConfigureTensor(void* tensor, XlaMusaMuDnnDataType data_type,
                                   XlaMusaMuDnnTensorFormat format,
                                   int32_t rank, const int64_t* dimensions,
                                   const int64_t* strides) {
  if (tensor == nullptr || dimensions == nullptr) {
    return XLA_MUSA_MUDNN_STATUS_INVALID_ARGUMENT;
  }
  if (rank < 3 || rank > 5) {
    return XLA_MUSA_MUDNN_STATUS_NOT_SUPPORTED;
  }
  for (int32_t dimension = 0; dimension < rank; ++dimension) {
    if (dimensions[dimension] <= 0 ||
        (strides != nullptr && strides[dimension] <= 0)) {
      return XLA_MUSA_MUDNN_STATUS_INVALID_ARGUMENT;
    }
  }

  mudnn::Tensor::Type native_type;
  XlaMusaMuDnnStatus status = ToNativeDataType(data_type, &native_type);
  if (status != XLA_MUSA_MUDNN_STATUS_SUCCESS) return status;
  mudnn::Tensor::Format native_format;
  status = ToNativeTensorFormat(format, rank, &native_format);
  if (status != XLA_MUSA_MUDNN_STATUS_SUCCESS) return status;

  return GuardVendorCall([&]() -> XlaMusaMuDnnStatus {
    mudnn::Tensor replacement;
    mudnn::Status vendor_status = replacement.SetType(native_type);
    if (vendor_status != mudnn::Status::SUCCESS) {
      return ToShimStatus(vendor_status);
    }
    vendor_status = replacement.SetFormat(native_format);
    if (vendor_status != mudnn::Status::SUCCESS) {
      return ToShimStatus(vendor_status);
    }
    vendor_status =
        strides == nullptr
            ? replacement.SetNdInfo(static_cast<int>(rank), dimensions)
            : replacement.SetNdInfo(static_cast<int>(rank), dimensions,
                                    strides);
    if (vendor_status != mudnn::Status::SUCCESS) {
      return ToShimStatus(vendor_status);
    }
    *AsTensor(tensor) = std::move(replacement);
    return XLA_MUSA_MUDNN_STATUS_SUCCESS;
  });
}

XlaMusaMuDnnStatus SetTensorAddress(void* tensor, void* address) {
  if (tensor == nullptr) {
    return XLA_MUSA_MUDNN_STATUS_INVALID_ARGUMENT;
  }
  return GuardVendorCall(
      [&] { return ToShimStatus(AsTensor(tensor)->SetAddr(address)); });
}

XlaMusaMuDnnStatus CreateConvolution(void** convolution) {
  if (convolution == nullptr) {
    return XLA_MUSA_MUDNN_STATUS_INVALID_ARGUMENT;
  }
  *convolution = nullptr;
  return GuardVendorCall([&]() -> XlaMusaMuDnnStatus {
    std::unique_ptr<mudnn::Convolution> result =
        std::make_unique<mudnn::Convolution>();
    *convolution = result.release();
    return XLA_MUSA_MUDNN_STATUS_SUCCESS;
  });
}

XlaMusaMuDnnStatus DestroyConvolution(void* convolution) {
  if (convolution == nullptr) {
    return XLA_MUSA_MUDNN_STATUS_INVALID_ARGUMENT;
  }
  return GuardVendorCall([&]() -> XlaMusaMuDnnStatus {
    delete AsConvolution(convolution);
    return XLA_MUSA_MUDNN_STATUS_SUCCESS;
  });
}

bool FitsNonNegativeInt(int64_t value) {
  return value >= 0 &&
         value <= static_cast<int64_t>(std::numeric_limits<int>::max());
}

bool FitsPositiveInt(int64_t value) {
  return value > 0 &&
         value <= static_cast<int64_t>(std::numeric_limits<int>::max());
}

XlaMusaMuDnnStatus ConfigureConvolution(void* convolution, int32_t spatial_rank,
                                        const int64_t* padding,
                                        const int64_t* strides,
                                        const int64_t* dilations,
                                        int64_t group_count) {
  if (convolution == nullptr || padding == nullptr || strides == nullptr ||
      dilations == nullptr) {
    return XLA_MUSA_MUDNN_STATUS_INVALID_ARGUMENT;
  }
  if (spatial_rank < 1 || spatial_rank > 3) {
    return XLA_MUSA_MUDNN_STATUS_NOT_SUPPORTED;
  }
  if (!FitsPositiveInt(group_count)) {
    return group_count <= 0 ? XLA_MUSA_MUDNN_STATUS_INVALID_ARGUMENT
                            : XLA_MUSA_MUDNN_STATUS_OUT_OF_RANGE;
  }

  int native_padding[3] = {};
  int native_strides[3] = {};
  int native_dilations[3] = {};
  for (int32_t dimension = 0; dimension < spatial_rank; ++dimension) {
    if (!FitsNonNegativeInt(padding[dimension]) ||
        !FitsPositiveInt(strides[dimension]) ||
        !FitsPositiveInt(dilations[dimension])) {
      if (padding[dimension] < 0 || strides[dimension] <= 0 ||
          dilations[dimension] <= 0) {
        return XLA_MUSA_MUDNN_STATUS_INVALID_ARGUMENT;
      }
      return XLA_MUSA_MUDNN_STATUS_OUT_OF_RANGE;
    }
    native_padding[dimension] = static_cast<int>(padding[dimension]);
    native_strides[dimension] = static_cast<int>(strides[dimension]);
    native_dilations[dimension] = static_cast<int>(dilations[dimension]);
  }

  return GuardVendorCall([&]() -> XlaMusaMuDnnStatus {
    mudnn::Status vendor_status =
        AsConvolution(convolution)->SetGroups(static_cast<int>(group_count));
    if (vendor_status != mudnn::Status::SUCCESS) {
      return ToShimStatus(vendor_status);
    }
    return ToShimStatus(AsConvolution(convolution)
                            ->SetNdInfo(static_cast<int>(spatial_rank),
                                        native_padding, native_strides,
                                        native_dilations));
  });
}

XlaMusaMuDnnStatus FromNativeAlgorithm(
    mudnn::Convolution::Algorithm native_algorithm,
    XlaMusaMuDnnAlgorithm* algorithm) {
  if (algorithm == nullptr) {
    return XLA_MUSA_MUDNN_STATUS_INVALID_ARGUMENT;
  }
  switch (native_algorithm) {
    case mudnn::Convolution::Algorithm::IMPLICIT_GEMM:
      *algorithm = XLA_MUSA_MUDNN_ALGORITHM_IMPLICIT_GEMM;
      return XLA_MUSA_MUDNN_STATUS_SUCCESS;
    case mudnn::Convolution::Algorithm::DIRECT:
      *algorithm = XLA_MUSA_MUDNN_ALGORITHM_DIRECT;
      return XLA_MUSA_MUDNN_STATUS_SUCCESS;
    case mudnn::Convolution::Algorithm::WINOGRAD_NONFUSED:
      *algorithm = XLA_MUSA_MUDNN_ALGORITHM_WINOGRAD_NONFUSED;
      return XLA_MUSA_MUDNN_STATUS_SUCCESS;
    case mudnn::Convolution::Algorithm::GEMM:
      *algorithm = XLA_MUSA_MUDNN_ALGORITHM_GEMM;
      return XLA_MUSA_MUDNN_STATUS_SUCCESS;
    default:
      return XLA_MUSA_MUDNN_STATUS_VENDOR_ERROR;
  }
}

XlaMusaMuDnnStatus FromNativeAlgorithm(
    mudnn::Convolution::AlgorithmBwdData native_algorithm,
    XlaMusaMuDnnAlgorithm* algorithm) {
  if (algorithm == nullptr) {
    return XLA_MUSA_MUDNN_STATUS_INVALID_ARGUMENT;
  }
  switch (native_algorithm) {
    case mudnn::Convolution::AlgorithmBwdData::IMPLICIT_GEMM:
      *algorithm = XLA_MUSA_MUDNN_ALGORITHM_IMPLICIT_GEMM;
      return XLA_MUSA_MUDNN_STATUS_SUCCESS;
    case mudnn::Convolution::AlgorithmBwdData::DIRECT:
      *algorithm = XLA_MUSA_MUDNN_ALGORITHM_DIRECT;
      return XLA_MUSA_MUDNN_STATUS_SUCCESS;
    case mudnn::Convolution::AlgorithmBwdData::WINOGRAD_NONFUSED:
      *algorithm = XLA_MUSA_MUDNN_ALGORITHM_WINOGRAD_NONFUSED;
      return XLA_MUSA_MUDNN_STATUS_SUCCESS;
    case mudnn::Convolution::AlgorithmBwdData::GEMM:
      *algorithm = XLA_MUSA_MUDNN_ALGORITHM_GEMM;
      return XLA_MUSA_MUDNN_STATUS_SUCCESS;
    default:
      return XLA_MUSA_MUDNN_STATUS_VENDOR_ERROR;
  }
}

XlaMusaMuDnnStatus FromNativeAlgorithm(
    mudnn::Convolution::AlgorithmBwdFilter native_algorithm,
    XlaMusaMuDnnAlgorithm* algorithm) {
  if (algorithm == nullptr) {
    return XLA_MUSA_MUDNN_STATUS_INVALID_ARGUMENT;
  }
  switch (native_algorithm) {
    case mudnn::Convolution::AlgorithmBwdFilter::IMPLICIT_GEMM:
      *algorithm = XLA_MUSA_MUDNN_ALGORITHM_IMPLICIT_GEMM;
      return XLA_MUSA_MUDNN_STATUS_SUCCESS;
    case mudnn::Convolution::AlgorithmBwdFilter::DIRECT:
      *algorithm = XLA_MUSA_MUDNN_ALGORITHM_DIRECT;
      return XLA_MUSA_MUDNN_STATUS_SUCCESS;
    case mudnn::Convolution::AlgorithmBwdFilter::WINOGRAD_NONFUSED:
      *algorithm = XLA_MUSA_MUDNN_ALGORITHM_WINOGRAD_NONFUSED;
      return XLA_MUSA_MUDNN_STATUS_SUCCESS;
    case mudnn::Convolution::AlgorithmBwdFilter::GEMM:
      *algorithm = XLA_MUSA_MUDNN_ALGORITHM_GEMM;
      return XLA_MUSA_MUDNN_STATUS_SUCCESS;
    default:
      return XLA_MUSA_MUDNN_STATUS_VENDOR_ERROR;
  }
}

XlaMusaMuDnnStatus ToNativeAlgorithm(
    XlaMusaMuDnnAlgorithm algorithm,
    mudnn::Convolution::Algorithm* native_algorithm) {
  if (native_algorithm == nullptr) {
    return XLA_MUSA_MUDNN_STATUS_INVALID_ARGUMENT;
  }
  switch (algorithm) {
    case XLA_MUSA_MUDNN_ALGORITHM_IMPLICIT_GEMM:
      *native_algorithm = mudnn::Convolution::Algorithm::IMPLICIT_GEMM;
      return XLA_MUSA_MUDNN_STATUS_SUCCESS;
    case XLA_MUSA_MUDNN_ALGORITHM_DIRECT:
      *native_algorithm = mudnn::Convolution::Algorithm::DIRECT;
      return XLA_MUSA_MUDNN_STATUS_SUCCESS;
    case XLA_MUSA_MUDNN_ALGORITHM_WINOGRAD_NONFUSED:
      *native_algorithm = mudnn::Convolution::Algorithm::WINOGRAD_NONFUSED;
      return XLA_MUSA_MUDNN_STATUS_SUCCESS;
    case XLA_MUSA_MUDNN_ALGORITHM_GEMM:
      *native_algorithm = mudnn::Convolution::Algorithm::GEMM;
      return XLA_MUSA_MUDNN_STATUS_SUCCESS;
    default:
      return XLA_MUSA_MUDNN_STATUS_INVALID_ARGUMENT;
  }
}

XlaMusaMuDnnStatus ToNativeAlgorithm(
    XlaMusaMuDnnAlgorithm algorithm,
    mudnn::Convolution::AlgorithmBwdData* native_algorithm) {
  if (native_algorithm == nullptr) {
    return XLA_MUSA_MUDNN_STATUS_INVALID_ARGUMENT;
  }
  switch (algorithm) {
    case XLA_MUSA_MUDNN_ALGORITHM_IMPLICIT_GEMM:
      *native_algorithm = mudnn::Convolution::AlgorithmBwdData::IMPLICIT_GEMM;
      return XLA_MUSA_MUDNN_STATUS_SUCCESS;
    case XLA_MUSA_MUDNN_ALGORITHM_DIRECT:
      *native_algorithm = mudnn::Convolution::AlgorithmBwdData::DIRECT;
      return XLA_MUSA_MUDNN_STATUS_SUCCESS;
    case XLA_MUSA_MUDNN_ALGORITHM_WINOGRAD_NONFUSED:
      *native_algorithm =
          mudnn::Convolution::AlgorithmBwdData::WINOGRAD_NONFUSED;
      return XLA_MUSA_MUDNN_STATUS_SUCCESS;
    case XLA_MUSA_MUDNN_ALGORITHM_GEMM:
      *native_algorithm = mudnn::Convolution::AlgorithmBwdData::GEMM;
      return XLA_MUSA_MUDNN_STATUS_SUCCESS;
    default:
      return XLA_MUSA_MUDNN_STATUS_INVALID_ARGUMENT;
  }
}

XlaMusaMuDnnStatus ToNativeAlgorithm(
    XlaMusaMuDnnAlgorithm algorithm,
    mudnn::Convolution::AlgorithmBwdFilter* native_algorithm) {
  if (native_algorithm == nullptr) {
    return XLA_MUSA_MUDNN_STATUS_INVALID_ARGUMENT;
  }
  switch (algorithm) {
    case XLA_MUSA_MUDNN_ALGORITHM_IMPLICIT_GEMM:
      *native_algorithm = mudnn::Convolution::AlgorithmBwdFilter::IMPLICIT_GEMM;
      return XLA_MUSA_MUDNN_STATUS_SUCCESS;
    case XLA_MUSA_MUDNN_ALGORITHM_DIRECT:
      *native_algorithm = mudnn::Convolution::AlgorithmBwdFilter::DIRECT;
      return XLA_MUSA_MUDNN_STATUS_SUCCESS;
    case XLA_MUSA_MUDNN_ALGORITHM_WINOGRAD_NONFUSED:
      *native_algorithm =
          mudnn::Convolution::AlgorithmBwdFilter::WINOGRAD_NONFUSED;
      return XLA_MUSA_MUDNN_STATUS_SUCCESS;
    case XLA_MUSA_MUDNN_ALGORITHM_GEMM:
      *native_algorithm = mudnn::Convolution::AlgorithmBwdFilter::GEMM;
      return XLA_MUSA_MUDNN_STATUS_SUCCESS;
    default:
      return XLA_MUSA_MUDNN_STATUS_INVALID_ARGUMENT;
  }
}

XlaMusaMuDnnStatus GetRecommendedAlgorithmImpl(
    mudnn::Handle& handle, mudnn::Convolution& convolution,
    XlaMusaMuDnnConvolutionKind kind, mudnn::Tensor& output,
    const mudnn::Tensor& data, const mudnn::Tensor& filter,
    XlaMusaMuDnnAlgorithm* algorithm) {
  switch (kind) {
    case XLA_MUSA_MUDNN_CONVOLUTION_FORWARD: {
      mudnn::Convolution::Algorithm native_algorithm;
      mudnn::Status status = convolution.GetRecommendForwardAlgorithm(
          handle, native_algorithm, output, data, filter);
      if (status != mudnn::Status::SUCCESS) return ToShimStatus(status);
      return FromNativeAlgorithm(native_algorithm, algorithm);
    }
    case XLA_MUSA_MUDNN_CONVOLUTION_BACKWARD_DATA: {
      mudnn::Convolution::AlgorithmBwdData native_algorithm;
      mudnn::Status status = convolution.GetRecommendBackwardDataAlgorithm(
          handle, native_algorithm, output, data, filter);
      if (status != mudnn::Status::SUCCESS) return ToShimStatus(status);
      return FromNativeAlgorithm(native_algorithm, algorithm);
    }
    case XLA_MUSA_MUDNN_CONVOLUTION_BACKWARD_FILTER: {
      mudnn::Convolution::AlgorithmBwdFilter native_algorithm;
      mudnn::Status status = convolution.GetRecommendBackwardFilterAlgorithm(
          handle, native_algorithm, output, data, filter);
      if (status != mudnn::Status::SUCCESS) return ToShimStatus(status);
      return FromNativeAlgorithm(native_algorithm, algorithm);
    }
    default:
      return XLA_MUSA_MUDNN_STATUS_INVALID_ARGUMENT;
  }
}

bool HasConvolutionArguments(void* handle, void* convolution, void* output,
                             const void* data, const void* filter) {
  return handle != nullptr && convolution != nullptr && output != nullptr &&
         data != nullptr && filter != nullptr;
}

XlaMusaMuDnnStatus GetRecommendedAlgorithm(void* handle, void* convolution,
                                           XlaMusaMuDnnConvolutionKind kind,
                                           void* output, const void* data,
                                           const void* filter,
                                           XlaMusaMuDnnAlgorithm* algorithm) {
  if (!HasConvolutionArguments(handle, convolution, output, data, filter) ||
      algorithm == nullptr) {
    return XLA_MUSA_MUDNN_STATUS_INVALID_ARGUMENT;
  }
  return GuardVendorCall([&]() -> XlaMusaMuDnnStatus {
    return GetRecommendedAlgorithmImpl(
        *AsHandle(handle), *AsConvolution(convolution), kind, *AsTensor(output),
        *AsTensor(data), *AsTensor(filter), algorithm);
  });
}

XlaMusaMuDnnStatus ResolveAlgorithm(mudnn::Handle& handle,
                                    mudnn::Convolution& convolution,
                                    XlaMusaMuDnnConvolutionKind kind,
                                    mudnn::Tensor& output,
                                    const mudnn::Tensor& data,
                                    const mudnn::Tensor& filter,
                                    XlaMusaMuDnnAlgorithm* algorithm) {
  if (*algorithm != XLA_MUSA_MUDNN_ALGORITHM_RECOMMENDED) {
    return (*algorithm >= XLA_MUSA_MUDNN_ALGORITHM_IMPLICIT_GEMM &&
            *algorithm <= XLA_MUSA_MUDNN_ALGORITHM_GEMM)
               ? XLA_MUSA_MUDNN_STATUS_SUCCESS
               : XLA_MUSA_MUDNN_STATUS_INVALID_ARGUMENT;
  }
  return GetRecommendedAlgorithmImpl(handle, convolution, kind, output, data,
                                     filter, algorithm);
}

XlaMusaMuDnnStatus GetWorkspaceSize(void* handle, void* convolution,
                                    XlaMusaMuDnnConvolutionKind kind,
                                    void* output, const void* data,
                                    const void* filter,
                                    XlaMusaMuDnnAlgorithm algorithm,
                                    uint64_t* workspace_size_bytes) {
  if (!HasConvolutionArguments(handle, convolution, output, data, filter) ||
      workspace_size_bytes == nullptr) {
    return XLA_MUSA_MUDNN_STATUS_INVALID_ARGUMENT;
  }
  *workspace_size_bytes = 0;
  return GuardVendorCall([&]() -> XlaMusaMuDnnStatus {
    mudnn::Handle& native_handle = *AsHandle(handle);
    mudnn::Convolution& native_convolution = *AsConvolution(convolution);
    mudnn::Tensor& native_output = *AsTensor(output);
    const mudnn::Tensor& native_data = *AsTensor(data);
    const mudnn::Tensor& native_filter = *AsTensor(filter);
    XlaMusaMuDnnStatus status =
        ResolveAlgorithm(native_handle, native_convolution, kind, native_output,
                         native_data, native_filter, &algorithm);
    if (status != XLA_MUSA_MUDNN_STATUS_SUCCESS) return status;

    size_t native_workspace_size = 0;
    mudnn::Status vendor_status;
    switch (kind) {
      case XLA_MUSA_MUDNN_CONVOLUTION_FORWARD: {
        mudnn::Convolution::Algorithm native_algorithm;
        status = ToNativeAlgorithm(algorithm, &native_algorithm);
        if (status != XLA_MUSA_MUDNN_STATUS_SUCCESS) return status;
        vendor_status = native_convolution.GetForwardWorkspaceSize(
            native_handle, native_workspace_size, native_output, native_data,
            native_filter, native_algorithm);
        break;
      }
      case XLA_MUSA_MUDNN_CONVOLUTION_BACKWARD_DATA: {
        mudnn::Convolution::AlgorithmBwdData native_algorithm;
        status = ToNativeAlgorithm(algorithm, &native_algorithm);
        if (status != XLA_MUSA_MUDNN_STATUS_SUCCESS) return status;
        vendor_status = native_convolution.GetBackwardDataWorkspaceSize(
            native_handle, native_workspace_size, native_output, native_data,
            native_filter, native_algorithm);
        break;
      }
      case XLA_MUSA_MUDNN_CONVOLUTION_BACKWARD_FILTER: {
        mudnn::Convolution::AlgorithmBwdFilter native_algorithm;
        status = ToNativeAlgorithm(algorithm, &native_algorithm);
        if (status != XLA_MUSA_MUDNN_STATUS_SUCCESS) return status;
        vendor_status = native_convolution.GetBackwardFilterWorkspaceSize(
            native_handle, native_workspace_size, native_output, native_data,
            native_filter, native_algorithm);
        break;
      }
      default:
        return XLA_MUSA_MUDNN_STATUS_INVALID_ARGUMENT;
    }
    if (vendor_status != mudnn::Status::SUCCESS) {
      return ToShimStatus(vendor_status);
    }
    *workspace_size_bytes = static_cast<uint64_t>(native_workspace_size);
    return XLA_MUSA_MUDNN_STATUS_SUCCESS;
  });
}

struct WorkspaceState {
  const XlaMusaMuDnnWorkspaceAllocator* allocator;
  XlaMusaMuDnnStatus status = XLA_MUSA_MUDNN_STATUS_SUCCESS;
};

mudnn::MemoryHandler AllocateWorkspace(WorkspaceState* state,
                                       size_t size_bytes) {
  void* address = nullptr;
  XlaMusaMuDnnStatus status = state->allocator->allocate(
      state->allocator->user_data, static_cast<uint64_t>(size_bytes), &address);
  if (!IsKnownShimStatus(status)) {
    status = XLA_MUSA_MUDNN_STATUS_VENDOR_ERROR;
  }
  if (status == XLA_MUSA_MUDNN_STATUS_SUCCESS && size_bytes != 0 &&
      address == nullptr) {
    status = XLA_MUSA_MUDNN_STATUS_RESOURCE_EXHAUSTED;
  }
  if (status != XLA_MUSA_MUDNN_STATUS_SUCCESS) {
    if (state->status == XLA_MUSA_MUDNN_STATUS_SUCCESS) {
      state->status = status;
    }
    return mudnn::MemoryHandler(nullptr, [](void*) {});
  }

  const XlaMusaMuDnnWorkspaceAllocator* allocator = state->allocator;
  return mudnn::MemoryHandler(
      address, [allocator, size_bytes](void* pointer) noexcept {
        if (pointer == nullptr) return;
        try {
          allocator->release(allocator->user_data, pointer,
                             static_cast<uint64_t>(size_bytes));
        } catch (...) {
          // Exceptions must never escape a vendor-owned unique_ptr deleter.
        }
      });
}

XlaMusaMuDnnStatus Convolve(
    void* handle, void* convolution, XlaMusaMuDnnConvolutionKind kind,
    void* output, const void* data, const void* filter,
    XlaMusaMuDnnAlgorithm algorithm,
    const XlaMusaMuDnnWorkspaceAllocator* workspace_allocator) {
  if (!HasConvolutionArguments(handle, convolution, output, data, filter) ||
      workspace_allocator == nullptr ||
      workspace_allocator->struct_size <
          XLA_MUSA_MUDNN_WORKSPACE_ALLOCATOR_MIN_STRUCT_SIZE ||
      workspace_allocator->reserved != 0 ||
      workspace_allocator->allocate == nullptr ||
      workspace_allocator->release == nullptr) {
    return XLA_MUSA_MUDNN_STATUS_INVALID_ARGUMENT;
  }

  return GuardVendorCall([&]() -> XlaMusaMuDnnStatus {
    mudnn::Handle& native_handle = *AsHandle(handle);
    mudnn::Convolution& native_convolution = *AsConvolution(convolution);
    mudnn::Tensor& native_output = *AsTensor(output);
    const mudnn::Tensor& native_data = *AsTensor(data);
    const mudnn::Tensor& native_filter = *AsTensor(filter);
    XlaMusaMuDnnStatus status =
        ResolveAlgorithm(native_handle, native_convolution, kind, native_output,
                         native_data, native_filter, &algorithm);
    if (status != XLA_MUSA_MUDNN_STATUS_SUCCESS) return status;

    WorkspaceState workspace_state{workspace_allocator};
    mudnn::MemoryMaintainer maintainer = [&workspace_state](size_t size_bytes) {
      return AllocateWorkspace(&workspace_state, size_bytes);
    };

    mudnn::Status vendor_status;
    switch (kind) {
      case XLA_MUSA_MUDNN_CONVOLUTION_FORWARD: {
        mudnn::Convolution::Algorithm native_algorithm;
        status = ToNativeAlgorithm(algorithm, &native_algorithm);
        if (status != XLA_MUSA_MUDNN_STATUS_SUCCESS) return status;
        vendor_status =
            native_convolution.Run(native_handle, native_output, native_data,
                                   native_filter, native_algorithm, maintainer);
        break;
      }
      case XLA_MUSA_MUDNN_CONVOLUTION_BACKWARD_DATA: {
        mudnn::Convolution::AlgorithmBwdData native_algorithm;
        status = ToNativeAlgorithm(algorithm, &native_algorithm);
        if (status != XLA_MUSA_MUDNN_STATUS_SUCCESS) return status;
        vendor_status = native_convolution.RunBwdData(
            native_handle, native_output, native_data, native_filter,
            native_algorithm, maintainer);
        break;
      }
      case XLA_MUSA_MUDNN_CONVOLUTION_BACKWARD_FILTER: {
        mudnn::Convolution::AlgorithmBwdFilter native_algorithm;
        status = ToNativeAlgorithm(algorithm, &native_algorithm);
        if (status != XLA_MUSA_MUDNN_STATUS_SUCCESS) return status;
        vendor_status = native_convolution.RunBwdFilter(
            native_handle, native_output, native_data, native_filter,
            native_algorithm, maintainer);
        break;
      }
      default:
        return XLA_MUSA_MUDNN_STATUS_INVALID_ARGUMENT;
    }
    if (workspace_state.status != XLA_MUSA_MUDNN_STATUS_SUCCESS) {
      return workspace_state.status;
    }
    return ToShimStatus(vendor_status);
  });
}

const XlaMusaMuDnnApiV1 kApiV1 = {
    sizeof(XlaMusaMuDnnApiV1),
    XLA_MUSA_MUDNN_ABI_VERSION_1,
    XLA_MUSA_MUDNN_CAPABILITIES_V1,
    GetVersion,
    CreateHandle,
    DestroyHandle,
    SetStream,
    SetAllowTf32,
    CreateTensor,
    DestroyTensor,
    ConfigureTensor,
    SetTensorAddress,
    CreateConvolution,
    DestroyConvolution,
    ConfigureConvolution,
    GetRecommendedAlgorithm,
    GetWorkspaceSize,
    Convolve,
};

}  // namespace

extern "C" XLA_MUSA_MUDNN_SHIM_EXPORT const XlaMusaMuDnnApiV1*
xla_musa_mudnn_get_api_v1(void) {
  return &kApiV1;
}
