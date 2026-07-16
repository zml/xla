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

#ifndef XLA_STREAM_EXECUTOR_MUSA_MUSA_DEVICE_PROPERTIES_H_
#define XLA_STREAM_EXECUTOR_MUSA_MUSA_DEVICE_PROPERTIES_H_

#include <cstdint>
#include <string>

namespace stream_executor::musa {

// Raw, runtime-discovered MUSA device facts. Compiler policy must not be added
// to this structure; in particular, logical subgroup width is a separate
// target ABI fact.
struct MusaDeviceProperties {
  std::string name;
  std::string pci_bus_id;
  uint64_t total_memory_bytes = 0;
  int compute_capability_major = 0;
  int compute_capability_minor = 0;
  int max_threads_per_block = 0;
  int max_block_dim_x = 0;
  int max_block_dim_y = 0;
  int max_block_dim_z = 0;
  int max_grid_dim_x = 0;
  int max_grid_dim_y = 0;
  int max_grid_dim_z = 0;
  int max_shared_memory_per_block = 0;
  int max_shared_memory_per_multiprocessor = 0;
  int max_shared_memory_per_block_optin = 0;
  int reserved_shared_memory_per_block = 0;
  int max_registers_per_block = 0;
  int max_registers_per_multiprocessor = 0;
  int max_threads_per_multiprocessor = 0;
  int max_blocks_per_multiprocessor = 0;
  int multiprocessor_count = 0;
  int hardware_warp_size = 0;
  int clock_rate_khz = 0;
  int memory_clock_rate_khz = 0;
  int memory_bus_width_bits = 0;
  int l2_cache_size_bytes = 0;
  int texture_alignment_bytes = 0;
  int texture_pitch_alignment_bytes = 0;
  int total_constant_memory_bytes = 0;
  bool ecc_enabled = false;
};

}  // namespace stream_executor::musa

#endif  // XLA_STREAM_EXECUTOR_MUSA_MUSA_DEVICE_PROPERTIES_H_
