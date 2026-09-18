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
  func.func @qualified_launch_coordinates() -> index {
    %tid_x = gpu.thread_id x
    %tid_y = gpu.thread_id y
    %tid_z = gpu.thread_id z
    %bid_x = gpu.block_id x
    %bid_y = gpu.block_id y
    %bid_z = gpu.block_id z
    %bdim_x = gpu.block_dim x
    %bdim_y = gpu.block_dim y
    %bdim_z = gpu.block_dim z
    %gdim_x = gpu.grid_dim x
    %gdim_y = gpu.grid_dim y
    %gdim_z = gpu.grid_dim z
    gpu.barrier
    %0 = arith.addi %tid_x, %tid_y : index
    %1 = arith.addi %0, %tid_z : index
    %2 = arith.addi %1, %bid_x : index
    %3 = arith.addi %2, %bid_y : index
    %4 = arith.addi %3, %bid_z : index
    %5 = arith.addi %4, %bdim_x : index
    %6 = arith.addi %5, %bdim_y : index
    %7 = arith.addi %6, %bdim_z : index
    %8 = arith.addi %7, %gdim_x : index
    %9 = arith.addi %8, %gdim_y : index
    %10 = arith.addi %9, %gdim_z : index
    return %10 : index
  }
}

// CHECK: llvm.data_layout = "e-p:64:64:64:64-p1:64:64:64:64-p2:64:64:64:64-p3:32:32-p4:32:32-p5:64:64-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128"
// CHECK: llvm.target_triple = "mtgpu-mt-musa"
// Shim declarations are inserted at module scope before the lowered kernel.
// CHECK-DAG: llvm.func @__xla_musa_v1_read_ctaid_x() -> i32 attributes {convergent, memory_effects = #llvm.memory_effects<other = none, argMem = none, inaccessibleMem = none, errnoMem = none, targetMem0 = none, targetMem1 = none>, no_unwind}
// CHECK-DAG: llvm.func @__xla_musa_v1_read_tid_x() -> i32 attributes {memory_effects = #llvm.memory_effects<other = none, argMem = none, inaccessibleMem = none, errnoMem = none, targetMem0 = none, targetMem1 = none>, no_unwind}
// CHECK-DAG: llvm.func @__xla_musa_v1_workgroup_barrier() attributes {convergent, no_unwind}
// CHECK-LABEL: llvm.func @qualified_launch_coordinates
// Coordinate values are unsigned i32 quantities. Widening must preserve the
// upper half of their range instead of sign-extending it.
// CHECK: llvm.call @__xla_musa_v1_read_tid_x() : () -> i32
// CHECK-NEXT: llvm.zext
// CHECK: llvm.call @__xla_musa_v1_read_tid_y() : () -> i32
// CHECK-NEXT: llvm.zext
// CHECK: llvm.call @__xla_musa_v1_read_tid_z() : () -> i32
// CHECK-NEXT: llvm.zext
// CHECK: llvm.call @__xla_musa_v1_read_ctaid_x() : () -> i32
// CHECK-NEXT: llvm.zext
// CHECK: llvm.call @__xla_musa_v1_read_ctaid_y() : () -> i32
// CHECK-NEXT: llvm.zext
// CHECK: llvm.call @__xla_musa_v1_read_ctaid_z() : () -> i32
// CHECK-NEXT: llvm.zext
// CHECK: llvm.call @__xla_musa_v1_read_ntid_x() : () -> i32
// CHECK-NEXT: llvm.zext
// CHECK: llvm.call @__xla_musa_v1_read_ntid_y() : () -> i32
// CHECK-NEXT: llvm.zext
// CHECK: llvm.call @__xla_musa_v1_read_ntid_z() : () -> i32
// CHECK-NEXT: llvm.zext
// CHECK: llvm.call @__xla_musa_v1_read_nctaid_x() : () -> i32
// CHECK-NEXT: llvm.zext
// CHECK: llvm.call @__xla_musa_v1_read_nctaid_y() : () -> i32
// CHECK-NEXT: llvm.zext
// CHECK: llvm.call @__xla_musa_v1_read_nctaid_z() : () -> i32
// CHECK-NEXT: llvm.zext
// CHECK-NEXT: llvm.call @__xla_musa_v1_workgroup_barrier() : () -> ()
// CHECK-NOT: nvvm.
// CHECK-NOT: rocdl.
