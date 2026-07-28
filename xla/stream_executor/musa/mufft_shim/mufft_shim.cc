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
#include <limits>

#include "mufft.h"
#include "xla/stream_executor/musa/mufft_shim/mufft_shim_abi.h"

namespace {

XlaMusaMuFftStatus ToShimStatus(mufftResult status) {
  switch (status) {
    case MUFFT_SUCCESS:
      return XLA_MUSA_MUFFT_STATUS_SUCCESS;
    case MUFFT_INVALID_TYPE:
    case MUFFT_INVALID_VALUE:
    case MUFFT_UNALIGNED_DATA:
      return XLA_MUSA_MUFFT_STATUS_INVALID_ARGUMENT;
    case MUFFT_INVALID_SIZE:
      return XLA_MUSA_MUFFT_STATUS_OUT_OF_RANGE;
    case MUFFT_ALLOC_FAILED:
      return XLA_MUSA_MUFFT_STATUS_RESOURCE_EXHAUSTED;
    case MUFFT_INVALID_PLAN:
    case MUFFT_SETUP_FAILED:
    case MUFFT_INCOMPLETE_PARAMETER_LIST:
    case MUFFT_INVALID_DEVICE:
    case MUFFT_NO_WORKSPACE:
    case MUFFT_LICENSE_ERROR:
      return XLA_MUSA_MUFFT_STATUS_FAILED_PRECONDITION;
    case MUFFT_NOT_IMPLEMENTED:
    case MUFFT_NOT_SUPPORTED:
      return XLA_MUSA_MUFFT_STATUS_NOT_SUPPORTED;
    case MUFFT_INTERNAL_ERROR:
    case MUFFT_EXEC_FAILED:
    case MUFFT_PARSE_ERROR:
    default:
      return XLA_MUSA_MUFFT_STATUS_VENDOR_ERROR;
  }
}

XlaMusaMuFftStatus GetVersion(XlaMusaMuFftVersion* version) {
  if (version == nullptr) {
    return XLA_MUSA_MUFFT_STATUS_INVALID_ARGUMENT;
  }
  XlaMusaMuFftVersion result = {};
  mufftResult status = mufftGetProperty(MUFFT_MAJOR_VERSION, &result.major);
  if (status != MUFFT_SUCCESS) return ToShimStatus(status);
  status = mufftGetProperty(MUFFT_MINOR_VERSION, &result.minor);
  if (status != MUFFT_SUCCESS) return ToShimStatus(status);
  status = mufftGetProperty(MUFFT_PATCH_LEVEL, &result.patch);
  if (status != MUFFT_SUCCESS) return ToShimStatus(status);
  *version = result;
  return XLA_MUSA_MUFFT_STATUS_SUCCESS;
}

XlaMusaMuFftStatus Create(void** plan) {
  if (plan == nullptr) {
    return XLA_MUSA_MUFFT_STATUS_INVALID_ARGUMENT;
  }
  *plan = nullptr;
  mufftHandle native_plan = nullptr;
  mufftResult status = mufftCreate(&native_plan);
  if (status != MUFFT_SUCCESS) return ToShimStatus(status);

  status = mufftSetAutoAllocation(native_plan, 0);
  if (status != MUFFT_SUCCESS) {
    mufftDestroy(native_plan);
    return ToShimStatus(status);
  }
  *plan = native_plan;
  return XLA_MUSA_MUFFT_STATUS_SUCCESS;
}

XlaMusaMuFftStatus Destroy(void* plan) {
  if (plan == nullptr) {
    return XLA_MUSA_MUFFT_STATUS_INVALID_ARGUMENT;
  }
  return ToShimStatus(mufftDestroy(static_cast<mufftHandle>(plan)));
}

bool FitsPositiveInt32(uint64_t value) {
  return value > 0 &&
         value <= static_cast<uint64_t>(std::numeric_limits<int32_t>::max());
}

bool FitsNonNegativeInt32(uint64_t value) {
  return value <= static_cast<uint64_t>(std::numeric_limits<int32_t>::max());
}

XlaMusaMuFftStatus DownsizePositiveArray(const uint64_t* source, int32_t rank,
                                         int* destination) {
  if (source == nullptr || destination == nullptr) {
    return XLA_MUSA_MUFFT_STATUS_INVALID_ARGUMENT;
  }
  for (int32_t index = 0; index < rank; ++index) {
    if (!FitsPositiveInt32(source[index])) {
      return XLA_MUSA_MUFFT_STATUS_OUT_OF_RANGE;
    }
    destination[index] = static_cast<int>(source[index]);
  }
  return XLA_MUSA_MUFFT_STATUS_SUCCESS;
}

XlaMusaMuFftStatus ToNativeType(XlaMusaMuFftType type, mufftType* native_type) {
  if (native_type == nullptr) {
    return XLA_MUSA_MUFFT_STATUS_INVALID_ARGUMENT;
  }
  switch (type) {
    case XLA_MUSA_MUFFT_TYPE_C2C:
      *native_type = MUFFT_C2C;
      return XLA_MUSA_MUFFT_STATUS_SUCCESS;
    case XLA_MUSA_MUFFT_TYPE_R2C:
      *native_type = MUFFT_R2C;
      return XLA_MUSA_MUFFT_STATUS_SUCCESS;
    case XLA_MUSA_MUFFT_TYPE_C2R:
      *native_type = MUFFT_C2R;
      return XLA_MUSA_MUFFT_STATUS_SUCCESS;
    case XLA_MUSA_MUFFT_TYPE_Z2Z:
      *native_type = MUFFT_Z2Z;
      return XLA_MUSA_MUFFT_STATUS_SUCCESS;
    case XLA_MUSA_MUFFT_TYPE_D2Z:
      *native_type = MUFFT_D2Z;
      return XLA_MUSA_MUFFT_STATUS_SUCCESS;
    case XLA_MUSA_MUFFT_TYPE_Z2D:
      *native_type = MUFFT_Z2D;
      return XLA_MUSA_MUFFT_STATUS_SUCCESS;
    default:
      return XLA_MUSA_MUFFT_STATUS_NOT_SUPPORTED;
  }
}

XlaMusaMuFftStatus ToNativeDirection(XlaMusaMuFftDirection direction,
                                     int* native_direction) {
  if (native_direction == nullptr) {
    return XLA_MUSA_MUFFT_STATUS_INVALID_ARGUMENT;
  }
  switch (direction) {
    case XLA_MUSA_MUFFT_DIRECTION_FORWARD:
      *native_direction = MUFFT_FORWARD;
      return XLA_MUSA_MUFFT_STATUS_SUCCESS;
    case XLA_MUSA_MUFFT_DIRECTION_INVERSE:
      *native_direction = MUFFT_INVERSE;
      return XLA_MUSA_MUFFT_STATUS_SUCCESS;
    default:
      return XLA_MUSA_MUFFT_STATUS_INVALID_ARGUMENT;
  }
}

XlaMusaMuFftStatus MakePlanMany(
    void* plan, int32_t rank, const uint64_t* element_count,
    const uint64_t* input_embed, uint64_t input_stride, uint64_t input_distance,
    const uint64_t* output_embed, uint64_t output_stride,
    uint64_t output_distance, XlaMusaMuFftType type, uint64_t batch_count,
    uint64_t* workspace_size_bytes) {
  if (plan == nullptr || element_count == nullptr ||
      workspace_size_bytes == nullptr) {
    return XLA_MUSA_MUFFT_STATUS_INVALID_ARGUMENT;
  }
  *workspace_size_bytes = 0;
  if (rank <= 0) return XLA_MUSA_MUFFT_STATUS_INVALID_ARGUMENT;
  if (rank > 3) return XLA_MUSA_MUFFT_STATUS_NOT_SUPPORTED;
  if (!FitsNonNegativeInt32(input_stride) ||
      !FitsNonNegativeInt32(input_distance) ||
      !FitsNonNegativeInt32(output_stride) ||
      !FitsNonNegativeInt32(output_distance) ||
      !FitsPositiveInt32(batch_count)) {
    return XLA_MUSA_MUFFT_STATUS_OUT_OF_RANGE;
  }

  int native_element_count[3] = {};
  int native_input_embed[3] = {};
  int native_output_embed[3] = {};
  XlaMusaMuFftStatus status =
      DownsizePositiveArray(element_count, rank, native_element_count);
  if (status != XLA_MUSA_MUFFT_STATUS_SUCCESS) return status;
  if (input_embed != nullptr) {
    status = DownsizePositiveArray(input_embed, rank, native_input_embed);
    if (status != XLA_MUSA_MUFFT_STATUS_SUCCESS) return status;
  }
  if (output_embed != nullptr) {
    status = DownsizePositiveArray(output_embed, rank, native_output_embed);
    if (status != XLA_MUSA_MUFFT_STATUS_SUCCESS) return status;
  }

  mufftType native_type;
  status = ToNativeType(type, &native_type);
  if (status != XLA_MUSA_MUFFT_STATUS_SUCCESS) return status;

  size_t native_workspace_size_bytes = 0;
  mufftResult result = mufftMakePlanMany(
      static_cast<mufftHandle>(plan), static_cast<int>(rank),
      native_element_count,
      input_embed == nullptr ? nullptr : native_input_embed,
      static_cast<int>(input_stride), static_cast<int>(input_distance),
      output_embed == nullptr ? nullptr : native_output_embed,
      static_cast<int>(output_stride), static_cast<int>(output_distance),
      native_type, static_cast<int>(batch_count), &native_workspace_size_bytes);
  if (result != MUFFT_SUCCESS) return ToShimStatus(result);
  *workspace_size_bytes = static_cast<uint64_t>(native_workspace_size_bytes);
  return XLA_MUSA_MUFFT_STATUS_SUCCESS;
}

XlaMusaMuFftStatus SetWorkArea(void* plan, void* workspace) {
  if (plan == nullptr) {
    return XLA_MUSA_MUFFT_STATUS_INVALID_ARGUMENT;
  }
  return ToShimStatus(
      mufftSetWorkArea(static_cast<mufftHandle>(plan), workspace));
}

XlaMusaMuFftStatus SetStream(void* plan, void* stream) {
  if (plan == nullptr) {
    return XLA_MUSA_MUFFT_STATUS_INVALID_ARGUMENT;
  }
  return ToShimStatus(mufftSetStream(static_cast<mufftHandle>(plan),
                                     static_cast<musaStream_t>(stream)));
}

XlaMusaMuFftStatus ExecC2C(void* plan, void* input, void* output,
                           XlaMusaMuFftDirection direction) {
  if (plan == nullptr || input == nullptr || output == nullptr) {
    return XLA_MUSA_MUFFT_STATUS_INVALID_ARGUMENT;
  }
  int native_direction = 0;
  XlaMusaMuFftStatus status = ToNativeDirection(direction, &native_direction);
  if (status != XLA_MUSA_MUFFT_STATUS_SUCCESS) return status;
  return ToShimStatus(mufftExecC2C(
      static_cast<mufftHandle>(plan), static_cast<mufftComplex*>(input),
      static_cast<mufftComplex*>(output), native_direction));
}

XlaMusaMuFftStatus ExecR2C(void* plan, void* input, void* output) {
  if (plan == nullptr || input == nullptr || output == nullptr) {
    return XLA_MUSA_MUFFT_STATUS_INVALID_ARGUMENT;
  }
  return ToShimStatus(mufftExecR2C(static_cast<mufftHandle>(plan),
                                   static_cast<mufftReal*>(input),
                                   static_cast<mufftComplex*>(output)));
}

XlaMusaMuFftStatus ExecC2R(void* plan, void* input, void* output) {
  if (plan == nullptr || input == nullptr || output == nullptr) {
    return XLA_MUSA_MUFFT_STATUS_INVALID_ARGUMENT;
  }
  return ToShimStatus(mufftExecC2R(static_cast<mufftHandle>(plan),
                                   static_cast<mufftComplex*>(input),
                                   static_cast<mufftReal*>(output)));
}

XlaMusaMuFftStatus ExecZ2Z(void* plan, void* input, void* output,
                           XlaMusaMuFftDirection direction) {
  if (plan == nullptr || input == nullptr || output == nullptr) {
    return XLA_MUSA_MUFFT_STATUS_INVALID_ARGUMENT;
  }
  int native_direction = 0;
  XlaMusaMuFftStatus status = ToNativeDirection(direction, &native_direction);
  if (status != XLA_MUSA_MUFFT_STATUS_SUCCESS) return status;
  return ToShimStatus(mufftExecZ2Z(
      static_cast<mufftHandle>(plan), static_cast<mufftDoubleComplex*>(input),
      static_cast<mufftDoubleComplex*>(output), native_direction));
}

XlaMusaMuFftStatus ExecD2Z(void* plan, void* input, void* output) {
  if (plan == nullptr || input == nullptr || output == nullptr) {
    return XLA_MUSA_MUFFT_STATUS_INVALID_ARGUMENT;
  }
  return ToShimStatus(mufftExecD2Z(static_cast<mufftHandle>(plan),
                                   static_cast<mufftDoubleReal*>(input),
                                   static_cast<mufftDoubleComplex*>(output)));
}

XlaMusaMuFftStatus ExecZ2D(void* plan, void* input, void* output) {
  if (plan == nullptr || input == nullptr || output == nullptr) {
    return XLA_MUSA_MUFFT_STATUS_INVALID_ARGUMENT;
  }
  return ToShimStatus(mufftExecZ2D(static_cast<mufftHandle>(plan),
                                   static_cast<mufftDoubleComplex*>(input),
                                   static_cast<mufftDoubleReal*>(output)));
}

const XlaMusaMuFftApiV1 kApiV1 = {
    sizeof(XlaMusaMuFftApiV1),
    XLA_MUSA_MUFFT_ABI_VERSION_1,
    XLA_MUSA_MUFFT_CAPABILITIES_V1,
    GetVersion,
    Create,
    Destroy,
    MakePlanMany,
    SetWorkArea,
    SetStream,
    ExecC2C,
    ExecR2C,
    ExecC2R,
    ExecZ2Z,
    ExecD2Z,
    ExecZ2D,
};

}  // namespace

extern "C" XLA_MUSA_MUFFT_SHIM_EXPORT const XlaMusaMuFftApiV1*
xla_musa_mufft_get_api_v1(void) {
  return &kApiV1;
}
