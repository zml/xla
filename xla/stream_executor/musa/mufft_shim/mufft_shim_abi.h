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

#ifndef XLA_STREAM_EXECUTOR_MUSA_MUFFT_SHIM_MUFFT_SHIM_ABI_H_
#define XLA_STREAM_EXECUTOR_MUSA_MUFFT_SHIM_MUFFT_SHIM_ABI_H_

#include <stddef.h>
#include <stdint.h>

// This header is the SDK-free boundary between the XLA PJRT plugin and the
// vendor-linked muFFT shim. Keep the ABI C-compatible, append-only, and free
// of MUSA or muFFT types. Plans, streams, and device addresses are opaque.

#define XLA_MUSA_MUFFT_ABI_VERSION_1 UINT32_C(1)

typedef int32_t XlaMusaMuFftStatus;
enum {
  XLA_MUSA_MUFFT_STATUS_SUCCESS = 0,
  XLA_MUSA_MUFFT_STATUS_INVALID_ARGUMENT = 1,
  XLA_MUSA_MUFFT_STATUS_NOT_SUPPORTED = 2,
  XLA_MUSA_MUFFT_STATUS_OUT_OF_RANGE = 3,
  XLA_MUSA_MUFFT_STATUS_RESOURCE_EXHAUSTED = 4,
  XLA_MUSA_MUFFT_STATUS_FAILED_PRECONDITION = 5,
  XLA_MUSA_MUFFT_STATUS_VENDOR_ERROR = 6,
};

// Transform types are normalized at the SDK-free boundary. Complex transform
// direction is supplied separately to ExecC2C and ExecZ2Z.
typedef uint32_t XlaMusaMuFftType;
enum {
  XLA_MUSA_MUFFT_TYPE_C2C = 1,
  XLA_MUSA_MUFFT_TYPE_R2C = 2,
  XLA_MUSA_MUFFT_TYPE_C2R = 3,
  XLA_MUSA_MUFFT_TYPE_Z2Z = 4,
  XLA_MUSA_MUFFT_TYPE_D2Z = 5,
  XLA_MUSA_MUFFT_TYPE_Z2D = 6,
};

typedef uint32_t XlaMusaMuFftDirection;
enum {
  XLA_MUSA_MUFFT_DIRECTION_FORWARD = 1,
  XLA_MUSA_MUFFT_DIRECTION_INVERSE = 2,
};

typedef uint64_t XlaMusaMuFftCapabilities;
enum {
  XLA_MUSA_MUFFT_CAPABILITY_C2C = UINT64_C(1) << 0,
  XLA_MUSA_MUFFT_CAPABILITY_R2C = UINT64_C(1) << 1,
  XLA_MUSA_MUFFT_CAPABILITY_C2R = UINT64_C(1) << 2,
  XLA_MUSA_MUFFT_CAPABILITY_Z2Z = UINT64_C(1) << 3,
  XLA_MUSA_MUFFT_CAPABILITY_D2Z = UINT64_C(1) << 4,
  XLA_MUSA_MUFFT_CAPABILITY_Z2D = UINT64_C(1) << 5,
  XLA_MUSA_MUFFT_CAPABILITY_RANK_1_TO_3 = UINT64_C(1) << 6,
  XLA_MUSA_MUFFT_CAPABILITY_EXTERNAL_WORKSPACE = UINT64_C(1) << 7,
  XLA_MUSA_MUFFT_CAPABILITY_STREAM_BINDING = UINT64_C(1) << 8,
};

#define XLA_MUSA_MUFFT_CAPABILITIES_V1                             \
  (XLA_MUSA_MUFFT_CAPABILITY_C2C | XLA_MUSA_MUFFT_CAPABILITY_R2C | \
   XLA_MUSA_MUFFT_CAPABILITY_C2R | XLA_MUSA_MUFFT_CAPABILITY_Z2Z | \
   XLA_MUSA_MUFFT_CAPABILITY_D2Z | XLA_MUSA_MUFFT_CAPABILITY_Z2D | \
   XLA_MUSA_MUFFT_CAPABILITY_RANK_1_TO_3 |                         \
   XLA_MUSA_MUFFT_CAPABILITY_EXTERNAL_WORKSPACE |                  \
   XLA_MUSA_MUFFT_CAPABILITY_STREAM_BINDING)

typedef struct XlaMusaMuFftVersion {
  int32_t major;
  int32_t minor;
  int32_t patch;
} XlaMusaMuFftVersion;

typedef XlaMusaMuFftStatus (*XlaMusaMuFftGetVersionFn)(
    XlaMusaMuFftVersion* version);
typedef XlaMusaMuFftStatus (*XlaMusaMuFftCreateFn)(void** plan);
typedef XlaMusaMuFftStatus (*XlaMusaMuFftDestroyFn)(void* plan);

// Creates a batched rank-dimensional plan with caller-owned workspace.
// Layout values are unsigned 64-bit so the plugin can pass StreamExecutor's
// native representation without narrowing. The shim rejects values that do
// not fit the qualified vendor LP32 plan API. `input_embed` and
// `output_embed` may be null, in which case the vendor's default contiguous
// layout is selected.
typedef XlaMusaMuFftStatus (*XlaMusaMuFftMakePlanManyFn)(
    void* plan, int32_t rank, const uint64_t* element_count,
    const uint64_t* input_embed, uint64_t input_stride, uint64_t input_distance,
    const uint64_t* output_embed, uint64_t output_stride,
    uint64_t output_distance, XlaMusaMuFftType type, uint64_t batch_count,
    uint64_t* workspace_size_bytes);

typedef XlaMusaMuFftStatus (*XlaMusaMuFftSetWorkAreaFn)(void* plan,
                                                        void* workspace);
typedef XlaMusaMuFftStatus (*XlaMusaMuFftSetStreamFn)(void* plan, void* stream);
typedef XlaMusaMuFftStatus (*XlaMusaMuFftExecC2CFn)(
    void* plan, void* input, void* output, XlaMusaMuFftDirection direction);
typedef XlaMusaMuFftStatus (*XlaMusaMuFftExecR2CFn)(void* plan, void* input,
                                                    void* output);
typedef XlaMusaMuFftStatus (*XlaMusaMuFftExecC2RFn)(void* plan, void* input,
                                                    void* output);
typedef XlaMusaMuFftStatus (*XlaMusaMuFftExecZ2ZFn)(
    void* plan, void* input, void* output, XlaMusaMuFftDirection direction);
typedef XlaMusaMuFftStatus (*XlaMusaMuFftExecD2ZFn)(void* plan, void* input,
                                                    void* output);
typedef XlaMusaMuFftStatus (*XlaMusaMuFftExecZ2DFn)(void* plan, void* input,
                                                    void* output);

typedef struct XlaMusaMuFftApiV1 {
  // Callers must check both fields before reading function pointers. Newer
  // compatible shims may append fields and increase `struct_size`.
  uint32_t struct_size;
  uint32_t abi_version;
  XlaMusaMuFftCapabilities capabilities;

  XlaMusaMuFftGetVersionFn get_version;
  XlaMusaMuFftCreateFn create;
  XlaMusaMuFftDestroyFn destroy;
  XlaMusaMuFftMakePlanManyFn make_plan_many;
  XlaMusaMuFftSetWorkAreaFn set_work_area;
  XlaMusaMuFftSetStreamFn set_stream;
  XlaMusaMuFftExecC2CFn exec_c2c;
  XlaMusaMuFftExecR2CFn exec_r2c;
  XlaMusaMuFftExecC2RFn exec_c2r;
  XlaMusaMuFftExecZ2ZFn exec_z2z;
  XlaMusaMuFftExecD2ZFn exec_d2z;
  XlaMusaMuFftExecZ2DFn exec_z2d;
} XlaMusaMuFftApiV1;

#define XLA_MUSA_MUFFT_API_V1_MIN_STRUCT_SIZE \
  (offsetof(XlaMusaMuFftApiV1, exec_z2d) +    \
   sizeof(((XlaMusaMuFftApiV1*)0)->exec_z2d))

typedef const XlaMusaMuFftApiV1* (*XlaMusaMuFftGetApiV1Fn)(void);

#if defined(_WIN32)
#define XLA_MUSA_MUFFT_SHIM_EXPORT __declspec(dllexport)
#elif defined(__GNUC__)
#define XLA_MUSA_MUFFT_SHIM_EXPORT __attribute__((visibility("default")))
#else
#define XLA_MUSA_MUFFT_SHIM_EXPORT
#endif

#ifdef __cplusplus
extern "C" {
#endif

XLA_MUSA_MUFFT_SHIM_EXPORT const XlaMusaMuFftApiV1* xla_musa_mufft_get_api_v1(
    void);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // XLA_STREAM_EXECUTOR_MUSA_MUFFT_SHIM_MUFFT_SHIM_ABI_H_
