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

#ifndef XLA_STREAM_EXECUTOR_MUSA_MUBLAS_SHIM_MUBLAS_SHIM_ABI_H_
#define XLA_STREAM_EXECUTOR_MUSA_MUBLAS_SHIM_MUBLAS_SHIM_ABI_H_

#include <stddef.h>
#include <stdint.h>

// This header is the SDK-free boundary between the XLA PJRT plugin and the
// vendor-linked muBLAS shim. Keep the ABI C-compatible, append-only, and free
// of MUSA or muBLAS types. Handles, streams, and device addresses are opaque.

#define XLA_MUSA_MUBLAS_ABI_VERSION_1 UINT32_C(1)

typedef int32_t XlaMusaMuBlasStatus;
enum {
  XLA_MUSA_MUBLAS_STATUS_SUCCESS = 0,
  XLA_MUSA_MUBLAS_STATUS_INVALID_ARGUMENT = 1,
  XLA_MUSA_MUBLAS_STATUS_NOT_SUPPORTED = 2,
  XLA_MUSA_MUBLAS_STATUS_OUT_OF_RANGE = 3,
  XLA_MUSA_MUBLAS_STATUS_RESOURCE_EXHAUSTED = 4,
  XLA_MUSA_MUBLAS_STATUS_VENDOR_ERROR = 5,
};

typedef uint32_t XlaMusaMuBlasDataType;
enum {
  XLA_MUSA_MUBLAS_DATA_TYPE_F16 = 1,
  XLA_MUSA_MUBLAS_DATA_TYPE_F32 = 2,
  XLA_MUSA_MUBLAS_DATA_TYPE_F64 = 3,
  XLA_MUSA_MUBLAS_DATA_TYPE_BF16 = 4,
};

typedef uint32_t XlaMusaMuBlasComputeType;
enum {
  XLA_MUSA_MUBLAS_COMPUTE_TYPE_F16 = 1,
  XLA_MUSA_MUBLAS_COMPUTE_TYPE_F32 = 2,
  XLA_MUSA_MUBLAS_COMPUTE_TYPE_F64 = 3,
};

typedef uint32_t XlaMusaMuBlasOperation;
enum {
  XLA_MUSA_MUBLAS_OPERATION_NONE = 0,
  XLA_MUSA_MUBLAS_OPERATION_TRANSPOSE = 1,
  XLA_MUSA_MUBLAS_OPERATION_CONJUGATE_TRANSPOSE = 2,
};

typedef uint64_t XlaMusaMuBlasCapabilities;
enum {
  XLA_MUSA_MUBLAS_CAPABILITY_GEMM_F16 = UINT64_C(1) << 0,
  XLA_MUSA_MUBLAS_CAPABILITY_GEMM_F32 = UINT64_C(1) << 1,
  XLA_MUSA_MUBLAS_CAPABILITY_GEMM_F64 = UINT64_C(1) << 2,
  // Reserved for a future shim that has qualified BF16 support. It is not
  // part of the ABI v1 required capability mask.
  XLA_MUSA_MUBLAS_CAPABILITY_GEMM_BF16 = UINT64_C(1) << 3,
};

#define XLA_MUSA_MUBLAS_CAPABILITIES_V1                                        \
  (XLA_MUSA_MUBLAS_CAPABILITY_GEMM_F16 | XLA_MUSA_MUBLAS_CAPABILITY_GEMM_F32 | \
   XLA_MUSA_MUBLAS_CAPABILITY_GEMM_F64)

typedef XlaMusaMuBlasStatus (*XlaMusaMuBlasCreateFn)(void** handle);
typedef XlaMusaMuBlasStatus (*XlaMusaMuBlasDestroyFn)(void* handle);
typedef XlaMusaMuBlasStatus (*XlaMusaMuBlasSetStreamFn)(void* handle,
                                                        void* stream);
typedef XlaMusaMuBlasStatus (*XlaMusaMuBlasGetVersionFn)(void* handle,
                                                         int32_t* version);

// Executes a homogeneous column-major GEMM:
//
//   C = alpha * op(A) * op(B) + beta * C
//
// A and B have `input_type`; C has `output_type`. Alpha and beta are F32 for
// F16 and F32 inputs, matching StreamExecutor's BlasSupport contract; they are
// F64 for F64. The supported v1 combinations are homogeneous
// F16/compute-F16, homogeneous F32/compute-F32, and homogeneous
// F64/compute-F64. All dimensions are signed 64-bit at this boundary and are
// range-checked before entering the LP64 vendor API.
typedef XlaMusaMuBlasStatus (*XlaMusaMuBlasGemmFn)(
    void* handle, XlaMusaMuBlasDataType input_type,
    XlaMusaMuBlasDataType output_type, XlaMusaMuBlasComputeType compute_type,
    XlaMusaMuBlasOperation trans_a, XlaMusaMuBlasOperation trans_b, int64_t m,
    int64_t n, int64_t k, const void* alpha, const void* a, int64_t lda,
    const void* b, int64_t ldb, const void* beta, void* c, int64_t ldc);

typedef struct XlaMusaMuBlasApiV1 {
  // Callers must check both fields before reading function pointers. Newer
  // compatible shims may append fields and increase `struct_size`.
  uint32_t struct_size;
  uint32_t abi_version;
  XlaMusaMuBlasCapabilities capabilities;

  XlaMusaMuBlasCreateFn create;
  XlaMusaMuBlasDestroyFn destroy;
  XlaMusaMuBlasSetStreamFn set_stream;
  XlaMusaMuBlasGetVersionFn get_version;
  XlaMusaMuBlasGemmFn gemm;
} XlaMusaMuBlasApiV1;

#define XLA_MUSA_MUBLAS_API_V1_MIN_STRUCT_SIZE \
  (offsetof(XlaMusaMuBlasApiV1, gemm) + sizeof(((XlaMusaMuBlasApiV1*)0)->gemm))

typedef const XlaMusaMuBlasApiV1* (*XlaMusaMuBlasGetApiV1Fn)(void);

#if defined(_WIN32)
#define XLA_MUSA_MUBLAS_SHIM_EXPORT __declspec(dllexport)
#elif defined(__GNUC__)
#define XLA_MUSA_MUBLAS_SHIM_EXPORT __attribute__((visibility("default")))
#else
#define XLA_MUSA_MUBLAS_SHIM_EXPORT
#endif

#ifdef __cplusplus
extern "C" {
#endif

XLA_MUSA_MUBLAS_SHIM_EXPORT const XlaMusaMuBlasApiV1*
xla_musa_mublas_get_api_v1(void);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // XLA_STREAM_EXECUTOR_MUSA_MUBLAS_SHIM_MUBLAS_SHIM_ABI_H_
