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

#ifndef XLA_STREAM_EXECUTOR_MUSA_MUDNN_SHIM_MUDNN_SHIM_ABI_H_
#define XLA_STREAM_EXECUTOR_MUSA_MUDNN_SHIM_MUDNN_SHIM_ABI_H_

#include <stddef.h>
#include <stdint.h>

// This header is the SDK-free boundary between the XLA PJRT plugin and the
// vendor-linked muDNN shim. Keep the ABI C-compatible, append-only, and free
// of MUSA or muDNN types. Handles, descriptors, streams, and device addresses
// are opaque.

#define XLA_MUSA_MUDNN_ABI_VERSION_1 UINT32_C(1)

typedef int32_t XlaMusaMuDnnStatus;
enum {
  XLA_MUSA_MUDNN_STATUS_SUCCESS = 0,
  XLA_MUSA_MUDNN_STATUS_INVALID_ARGUMENT = 1,
  XLA_MUSA_MUDNN_STATUS_NOT_SUPPORTED = 2,
  XLA_MUSA_MUDNN_STATUS_OUT_OF_RANGE = 3,
  XLA_MUSA_MUDNN_STATUS_RESOURCE_EXHAUSTED = 4,
  XLA_MUSA_MUDNN_STATUS_FAILED_PRECONDITION = 5,
  XLA_MUSA_MUDNN_STATUS_VENDOR_ERROR = 6,
};

typedef uint32_t XlaMusaMuDnnDataType;
enum {
  XLA_MUSA_MUDNN_DATA_TYPE_F16 = 1,
  XLA_MUSA_MUDNN_DATA_TYPE_BF16 = 2,
  XLA_MUSA_MUDNN_DATA_TYPE_F32 = 3,
};

// Formats are normalized at the SDK-free boundary. UNKNOWN is useful with
// explicit strides when no named vendor layout describes the tensor.
typedef uint32_t XlaMusaMuDnnTensorFormat;
enum {
  XLA_MUSA_MUDNN_TENSOR_FORMAT_UNKNOWN = 0,
  XLA_MUSA_MUDNN_TENSOR_FORMAT_NCW = 1,
  XLA_MUSA_MUDNN_TENSOR_FORMAT_NWC = 2,
  XLA_MUSA_MUDNN_TENSOR_FORMAT_NCHW = 3,
  XLA_MUSA_MUDNN_TENSOR_FORMAT_NHWC = 4,
  XLA_MUSA_MUDNN_TENSOR_FORMAT_HWCN = 5,
  XLA_MUSA_MUDNN_TENSOR_FORMAT_NCDHW = 6,
  XLA_MUSA_MUDNN_TENSOR_FORMAT_NDHWC = 7,
  XLA_MUSA_MUDNN_TENSOR_FORMAT_DHWCN = 8,
};

typedef uint32_t XlaMusaMuDnnConvolutionKind;
enum {
  XLA_MUSA_MUDNN_CONVOLUTION_FORWARD = 1,
  XLA_MUSA_MUDNN_CONVOLUTION_BACKWARD_DATA = 2,
  XLA_MUSA_MUDNN_CONVOLUTION_BACKWARD_FILTER = 3,
};

// These values are normalized and deliberately do not expose the differently
// ordered forward and backward vendor enums. RECOMMENDED may be supplied to
// workspace-size and execution calls; the shim resolves it for that operation.
typedef int32_t XlaMusaMuDnnAlgorithm;
enum {
  XLA_MUSA_MUDNN_ALGORITHM_RECOMMENDED = -1,
  XLA_MUSA_MUDNN_ALGORITHM_IMPLICIT_GEMM = 0,
  XLA_MUSA_MUDNN_ALGORITHM_DIRECT = 1,
  XLA_MUSA_MUDNN_ALGORITHM_WINOGRAD_NONFUSED = 2,
  XLA_MUSA_MUDNN_ALGORITHM_GEMM = 3,
};

typedef uint64_t XlaMusaMuDnnCapabilities;
enum {
  XLA_MUSA_MUDNN_CAPABILITY_F16 = UINT64_C(1) << 0,
  XLA_MUSA_MUDNN_CAPABILITY_BF16 = UINT64_C(1) << 1,
  XLA_MUSA_MUDNN_CAPABILITY_F32 = UINT64_C(1) << 2,
  XLA_MUSA_MUDNN_CAPABILITY_RANK_3_TO_5 = UINT64_C(1) << 3,
  XLA_MUSA_MUDNN_CAPABILITY_EXPLICIT_STRIDES = UINT64_C(1) << 4,
  XLA_MUSA_MUDNN_CAPABILITY_CONVOLUTION_FORWARD = UINT64_C(1) << 5,
  XLA_MUSA_MUDNN_CAPABILITY_CONVOLUTION_BACKWARD_DATA = UINT64_C(1) << 6,
  XLA_MUSA_MUDNN_CAPABILITY_CONVOLUTION_BACKWARD_FILTER = UINT64_C(1) << 7,
  XLA_MUSA_MUDNN_CAPABILITY_GROUPED_CONVOLUTION = UINT64_C(1) << 8,
  XLA_MUSA_MUDNN_CAPABILITY_PAD_STRIDE_DILATION = UINT64_C(1) << 9,
  XLA_MUSA_MUDNN_CAPABILITY_RECOMMENDED_ALGORITHM = UINT64_C(1) << 10,
  XLA_MUSA_MUDNN_CAPABILITY_EXPLICIT_ALGORITHM = UINT64_C(1) << 11,
  XLA_MUSA_MUDNN_CAPABILITY_EXTERNAL_WORKSPACE = UINT64_C(1) << 12,
  XLA_MUSA_MUDNN_CAPABILITY_STREAM_BINDING = UINT64_C(1) << 13,
  XLA_MUSA_MUDNN_CAPABILITY_ALLOW_TF32 = UINT64_C(1) << 14,
};

#define XLA_MUSA_MUDNN_CAPABILITIES_V1                                     \
  (XLA_MUSA_MUDNN_CAPABILITY_F16 | XLA_MUSA_MUDNN_CAPABILITY_F32 |         \
   XLA_MUSA_MUDNN_CAPABILITY_RANK_3_TO_5 |                                 \
   XLA_MUSA_MUDNN_CAPABILITY_EXPLICIT_STRIDES |                            \
   XLA_MUSA_MUDNN_CAPABILITY_CONVOLUTION_FORWARD |                         \
   XLA_MUSA_MUDNN_CAPABILITY_CONVOLUTION_BACKWARD_DATA |                   \
   XLA_MUSA_MUDNN_CAPABILITY_CONVOLUTION_BACKWARD_FILTER |                 \
   XLA_MUSA_MUDNN_CAPABILITY_GROUPED_CONVOLUTION |                         \
   XLA_MUSA_MUDNN_CAPABILITY_PAD_STRIDE_DILATION |                         \
   XLA_MUSA_MUDNN_CAPABILITY_RECOMMENDED_ALGORITHM |                       \
   XLA_MUSA_MUDNN_CAPABILITY_EXPLICIT_ALGORITHM |                          \
   XLA_MUSA_MUDNN_CAPABILITY_EXTERNAL_WORKSPACE |                          \
   XLA_MUSA_MUDNN_CAPABILITY_STREAM_BINDING |                              \
   XLA_MUSA_MUDNN_CAPABILITY_ALLOW_TF32)

typedef struct XlaMusaMuDnnVersion {
  int32_t major;
  int32_t minor;
  int32_t patch;
} XlaMusaMuDnnVersion;

// muDNN asks for temporary device memory through a C++ MemoryMaintainer.
// The shim converts that request into these caller-owned C callbacks. The
// release callback may be a no-op when the caller's scratch allocator owns the
// allocation for a longer stream-ordered lifetime.
typedef XlaMusaMuDnnStatus (*XlaMusaMuDnnWorkspaceAllocateFn)(
    void* user_data, uint64_t size_bytes, void** address);
typedef void (*XlaMusaMuDnnWorkspaceReleaseFn)(void* user_data, void* address,
                                               uint64_t size_bytes);

typedef struct XlaMusaMuDnnWorkspaceAllocator {
  // Callers and shims must check struct_size before reading callback fields.
  uint32_t struct_size;
  uint32_t reserved;
  void* user_data;
  XlaMusaMuDnnWorkspaceAllocateFn allocate;
  XlaMusaMuDnnWorkspaceReleaseFn release;
} XlaMusaMuDnnWorkspaceAllocator;

#define XLA_MUSA_MUDNN_WORKSPACE_ALLOCATOR_MIN_STRUCT_SIZE \
  (offsetof(XlaMusaMuDnnWorkspaceAllocator, release) +     \
   sizeof(((XlaMusaMuDnnWorkspaceAllocator*)0)->release))

typedef XlaMusaMuDnnStatus (*XlaMusaMuDnnGetVersionFn)(
    XlaMusaMuDnnVersion* version);
typedef XlaMusaMuDnnStatus (*XlaMusaMuDnnCreateHandleFn)(int32_t device_ordinal,
                                                         void** handle);
typedef XlaMusaMuDnnStatus (*XlaMusaMuDnnDestroyHandleFn)(void* handle);
typedef XlaMusaMuDnnStatus (*XlaMusaMuDnnSetStreamFn)(void* handle,
                                                      void* stream);
typedef XlaMusaMuDnnStatus (*XlaMusaMuDnnSetAllowTf32Fn)(void* handle,
                                                         uint32_t allow_tf32);

typedef XlaMusaMuDnnStatus (*XlaMusaMuDnnCreateTensorFn)(void** tensor);
typedef XlaMusaMuDnnStatus (*XlaMusaMuDnnDestroyTensorFn)(void* tensor);
typedef XlaMusaMuDnnStatus (*XlaMusaMuDnnConfigureTensorFn)(
    void* tensor, XlaMusaMuDnnDataType data_type,
    XlaMusaMuDnnTensorFormat format, int32_t rank, const int64_t* dimensions,
    const int64_t* strides);
typedef XlaMusaMuDnnStatus (*XlaMusaMuDnnSetTensorAddressFn)(void* tensor,
                                                             void* address);

typedef XlaMusaMuDnnStatus (*XlaMusaMuDnnCreateConvolutionFn)(
    void** convolution);
typedef XlaMusaMuDnnStatus (*XlaMusaMuDnnDestroyConvolutionFn)(
    void* convolution);
typedef XlaMusaMuDnnStatus (*XlaMusaMuDnnConfigureConvolutionFn)(
    void* convolution, int32_t spatial_rank, const int64_t* padding,
    const int64_t* strides, const int64_t* dilations, int64_t group_count);

// For FORWARD: output=y, data=x, filter=w.
// For BACKWARD_DATA: output=dx, data=dy, filter=w.
// For BACKWARD_FILTER: output=dw, data=x, filter=dy.
typedef XlaMusaMuDnnStatus (*XlaMusaMuDnnGetRecommendedAlgorithmFn)(
    void* handle, void* convolution, XlaMusaMuDnnConvolutionKind kind,
    void* output, const void* data, const void* filter,
    XlaMusaMuDnnAlgorithm* algorithm);
typedef XlaMusaMuDnnStatus (*XlaMusaMuDnnGetWorkspaceSizeFn)(
    void* handle, void* convolution, XlaMusaMuDnnConvolutionKind kind,
    void* output, const void* data, const void* filter,
    XlaMusaMuDnnAlgorithm algorithm, uint64_t* workspace_size_bytes);
typedef XlaMusaMuDnnStatus (*XlaMusaMuDnnConvolveFn)(
    void* handle, void* convolution, XlaMusaMuDnnConvolutionKind kind,
    void* output, const void* data, const void* filter,
    XlaMusaMuDnnAlgorithm algorithm,
    const XlaMusaMuDnnWorkspaceAllocator* workspace_allocator);

typedef struct XlaMusaMuDnnApiV1 {
  // Callers must check all three prefix fields before reading function
  // pointers. Newer compatible shims may append fields and increase
  // struct_size without changing this prefix.
  uint32_t struct_size;
  uint32_t abi_version;
  XlaMusaMuDnnCapabilities capabilities;

  XlaMusaMuDnnGetVersionFn get_version;
  XlaMusaMuDnnCreateHandleFn create_handle;
  XlaMusaMuDnnDestroyHandleFn destroy_handle;
  XlaMusaMuDnnSetStreamFn set_stream;
  XlaMusaMuDnnSetAllowTf32Fn set_allow_tf32;
  XlaMusaMuDnnCreateTensorFn create_tensor;
  XlaMusaMuDnnDestroyTensorFn destroy_tensor;
  XlaMusaMuDnnConfigureTensorFn configure_tensor;
  XlaMusaMuDnnSetTensorAddressFn set_tensor_address;
  XlaMusaMuDnnCreateConvolutionFn create_convolution;
  XlaMusaMuDnnDestroyConvolutionFn destroy_convolution;
  XlaMusaMuDnnConfigureConvolutionFn configure_convolution;
  XlaMusaMuDnnGetRecommendedAlgorithmFn get_recommended_algorithm;
  XlaMusaMuDnnGetWorkspaceSizeFn get_workspace_size;
  XlaMusaMuDnnConvolveFn convolve;
} XlaMusaMuDnnApiV1;

#define XLA_MUSA_MUDNN_API_V1_MIN_STRUCT_SIZE \
  (offsetof(XlaMusaMuDnnApiV1, convolve) +    \
   sizeof(((XlaMusaMuDnnApiV1*)0)->convolve))

typedef const XlaMusaMuDnnApiV1* (*XlaMusaMuDnnGetApiV1Fn)(void);

#if defined(_WIN32)
#define XLA_MUSA_MUDNN_SHIM_EXPORT __declspec(dllexport)
#elif defined(__GNUC__)
#define XLA_MUSA_MUDNN_SHIM_EXPORT __attribute__((visibility("default")))
#else
#define XLA_MUSA_MUDNN_SHIM_EXPORT
#endif

#ifdef __cplusplus
extern "C" {
#endif

XLA_MUSA_MUDNN_SHIM_EXPORT const XlaMusaMuDnnApiV1* xla_musa_mudnn_get_api_v1(
    void);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // XLA_STREAM_EXECUTOR_MUSA_MUDNN_SHIM_MUDNN_SHIM_ABI_H_
