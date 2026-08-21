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

#include "xla/stream_executor/rocm/rocm_host_overhead_benchmark_kernels.h"

#include <cstddef>
#include <cstdint>
#include <utility>

namespace stream_executor::gpu {
namespace {

template <size_t Index>
struct KernelArg {
  uint64_t value;
};

__device__ uint64_t benchmark_sink;

template <size_t... Indices>
__global__ void HostOverheadBenchmarkKernel(KernelArg<Indices>... args) {
  uint64_t sum = (uint64_t{0} + ... + args.value);
  if (threadIdx.x == 0 && sum == ~uint64_t{0}) {
    benchmark_sink = sum;
  }
}

template <size_t... Indices>
hipError_t GetKernel(hipFunction_t* function,
                     std::index_sequence<Indices...>) {
  return hipGetFuncBySymbol(
      function,
      reinterpret_cast<const void*>(&HostOverheadBenchmarkKernel<Indices...>));
}

}  // namespace

hipError_t GetRocmHostOverheadBenchmarkKernel(int arity,
                                              hipFunction_t* function) {
  switch (arity) {
    case 0:
      return GetKernel(function, std::make_index_sequence<0>());
    case 1:
      return GetKernel(function, std::make_index_sequence<1>());
    case 4:
      return GetKernel(function, std::make_index_sequence<4>());
    case 16:
      return GetKernel(function, std::make_index_sequence<16>());
    case 64:
      return GetKernel(function, std::make_index_sequence<64>());
    default:
      return hipErrorInvalidValue;
  }
}

}  // namespace stream_executor::gpu
