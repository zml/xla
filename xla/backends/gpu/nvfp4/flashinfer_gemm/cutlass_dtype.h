// Slimmed replacement for flashinfer/cutlass_utils.cuh: the FP4 SM120 GEMM
// template only needs flashinfer::cutlass_dtype<T>::type. The upstream header
// also drags in cutlass/util/* (host_tensor, command_line, reference kernels)
// which we don't want in this TU. Keep only the dtype trait.
#ifndef FLASHINFER_FP4SPIKE_CUTLASS_DTYPE_H_
#define FLASHINFER_FP4SPIKE_CUTLASS_DTYPE_H_

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "cutlass/numeric_types.h"

namespace flashinfer {

template <typename T>
struct cutlass_dtype {
  using type = T;
};

template <>
struct cutlass_dtype<half> {
  using type = cutlass::half_t;
};

template <>
struct cutlass_dtype<__nv_bfloat16> {
  using type = cutlass::bfloat16_t;
};

template <typename T>
using cutlass_dtype_t = typename cutlass_dtype<T>::type;

}  // namespace flashinfer

#endif  // FLASHINFER_FP4SPIKE_CUTLASS_DTYPE_H_
