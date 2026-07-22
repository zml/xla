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
// RUN: emitters_opt %s --allow-unregistered-dialect \
// RUN:   -xla-lower-tensors="gpu_device_info='threads_per_warp: 128 musa_compute_capability {architecture: \"mp_21\" major: 2 minor: 1 hardware_warp_size: 128 logical_subgroup_size: 32}'" \
// RUN:   | FileCheck %s

func.func @atomic_add_i32(%input: tensor<8xi32>, %index: index, %update: i32)
    -> tensor<8xi32> {
  %result = xla.atomic_rmw %input[%index] : tensor<8xi32> {
    ^bb0(%current : i32):
      %sum = arith.addi %current, %update : i32
      xla.yield %sum : i32
  }
  return %result : tensor<8xi32>
}

// CHECK-LABEL: func.func @atomic_add_i32
// CHECK: %[[ADDR:.*]] = llvm.getelementptr
// CHECK-NEXT: %[[GLOBAL:.*]] = llvm.addrspacecast %[[ADDR]] : !llvm.ptr to !llvm.ptr<1>
// CHECK-NEXT: %[[INIT:.*]] = llvm.load %[[GLOBAL]] : !llvm.ptr<1> -> i32
// CHECK: scf.while
// CHECK: %[[PAIR:.*]] = llvm.cmpxchg %[[GLOBAL]], {{.*}}, {{.*}} monotonic monotonic {alignment = 4 : i64} : !llvm.ptr<1>, i32
// CHECK-NOT: llvm.atomicrmw

func.func @atomic_add_f32(%input: tensor<8xf32>, %index: index, %update: f32)
    -> tensor<8xf32> {
  %result = xla.atomic_rmw %input[%index] : tensor<8xf32> {
    ^bb0(%current : f32):
      %sum = arith.addf %current, %update : f32
      xla.yield %sum : f32
  }
  return %result : tensor<8xf32>
}

// CHECK-LABEL: func.func @atomic_add_f32
// CHECK: %[[ADDR_F32:.*]] = llvm.getelementptr
// CHECK-NEXT: %[[GLOBAL_F32:.*]] = llvm.addrspacecast %[[ADDR_F32]] : !llvm.ptr to !llvm.ptr<1>
// CHECK-NEXT: %[[INIT_F32:.*]] = llvm.load %[[GLOBAL_F32]] : !llvm.ptr<1> -> i32
// CHECK: scf.while
// CHECK: arith.bitcast {{.*}} : i32 to f32
// CHECK: arith.bitcast {{.*}} : f32 to i32
// CHECK: llvm.cmpxchg %[[GLOBAL_F32]], {{.*}}, {{.*}} monotonic monotonic {alignment = 4 : i64} : !llvm.ptr<1>, i32
// CHECK-NOT: llvm.atomicrmw
