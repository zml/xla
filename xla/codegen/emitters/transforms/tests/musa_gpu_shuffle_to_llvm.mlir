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
// RUN: emitters_opt %s \
// RUN:   -xla-lower-to-llvm-gpu="gpu_device_info='threads_per_warp: 128 musa_compute_capability {architecture: \"mp_21\" major: 2 minor: 1 hardware_warp_size: 128 logical_subgroup_size: 32}'" \
// RUN:   | FileCheck %s

module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index,64 : i32>>} {
  func.func @shuffle_down_f32(%value: f32, %offset: i32) -> (f32, i1) {
    %width = arith.constant 32 : i32
    %shuffled, %valid = gpu.shuffle down %value, %offset, %width : f32
    return %shuffled, %valid : f32, i1
  }

  func.func @shuffle_up_i32(%value: i32, %offset: i32) -> (i32, i1) {
    %width = arith.constant 32 : i32
    %shuffled, %valid = gpu.shuffle up %value, %offset, %width : i32
    return %shuffled, %valid : i32, i1
  }

  func.func @shuffle_xor_f64(%value: f64, %offset: i32) -> (f64, i1) {
    %width = arith.constant 32 : i32
    %shuffled, %valid = gpu.shuffle xor %value, %offset, %width : f64
    return %shuffled, %valid : f64, i1
  }

  func.func @shuffle_idx_i32(%value: i32, %offset: i32) -> (i32, i1) {
    %width = arith.constant 32 : i32
    %shuffled, %valid = gpu.shuffle idx %value, %offset, %width : i32
    return %shuffled, %valid : i32, i1
  }
}

// CHECK-DAG: llvm.func @__xla_musa_v1_read_tid_x() -> i32 attributes {memory_effects = #llvm.memory_effects<other = none, argMem = none, inaccessibleMem = none, errnoMem = none, targetMem0 = none, targetMem1 = none>, no_unwind}
// CHECK-DAG: llvm.func @__xla_musa_v1_subgroup_read_lane_i32(i32, i32) -> i32 attributes {convergent, memory_effects = #llvm.memory_effects<other = none, argMem = none, inaccessibleMem = none, errnoMem = none, targetMem0 = none, targetMem1 = none>, no_unwind, will_return}
// CHECK-LABEL: llvm.func @shuffle_down_f32
// CHECK: %[[LANE:.*]] = llvm.call @__xla_musa_v1_read_tid_x() : () -> i32
// CHECK: %[[LOGICAL:.*]] = llvm.and %[[LANE]], {{.*}} : i32
// CHECK: %[[SOURCE:.*]] = llvm.add %[[LOGICAL]], {{.*}} : i32
// CHECK: %[[VALID:.*]] = llvm.icmp "ult" %[[SOURCE]], {{.*}} : i32
// CHECK: llvm.select %[[VALID]]
// CHECK: llvm.call @__xla_musa_v1_subgroup_read_lane_i32({{.*}}) : (i32, i32) -> i32
// CHECK-LABEL: llvm.func @shuffle_up_i32
// CHECK: llvm.sub
// CHECK: llvm.icmp "ule"
// CHECK-LABEL: llvm.func @shuffle_xor_f64
// CHECK: llvm.xor
// A 64-bit value is decomposed into two i32 words, and both use the same
// qualified logical source lane.
// CHECK-COUNT-2: llvm.call @__xla_musa_v1_subgroup_read_lane_i32({{.*}}) : (i32, i32) -> i32
// CHECK-LABEL: llvm.func @shuffle_idx_i32
// CHECK: llvm.icmp "ult"
// CHECK: llvm.call @__xla_musa_v1_subgroup_read_lane_i32({{.*}}) : (i32, i32) -> i32
// CHECK-NOT: nvvm.
// CHECK-NOT: rocdl.
