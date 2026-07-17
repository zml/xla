// Copyright 2026 The OpenXLA Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ==============================================================================
// RUN: not emitters_opt %s \
// RUN:   -xla-lower-to-llvm-gpu="gpu_device_info='threads_per_warp: 128 musa_compute_capability {architecture: \"mp_21\" major: 2 minor: 1 hardware_warp_size: 128 logical_subgroup_size: 32}'" \
// RUN:   2>&1 | FileCheck %s

module {
  func.func @unsupported_shuffle_width(%value: i32, %offset: i32) -> i32 {
    %width = arith.constant 16 : i32
    %shuffled, %valid = gpu.shuffle down %value, %offset, %width : i32
    return %shuffled : i32
  }
}

// CHECK: failed to legalize operation 'gpu.shuffle'
