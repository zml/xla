; Copyright 2026 The OpenXLA Authors.
; SPDX-License-Identifier: Apache-2.0

source_filename = "all_shims"
target datalayout = "e-p:64:64:64:64-p1:64:64:64:64-p2:64:64:64:64-p3:32:32-p4:32:32-p5:64:64-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128"
target triple = "mtgpu-mt-musa"

define void @kernel() {
entry:
  %clock32 = call i32 @__xla_musa_v1_clock32()
  %clock64 = call i64 @__xla_musa_v1_clock64()
  %ctaid_x = call i32 @__xla_musa_v1_read_ctaid_x()
  %ctaid_y = call i32 @__xla_musa_v1_read_ctaid_y()
  %ctaid_z = call i32 @__xla_musa_v1_read_ctaid_z()
  %nctaid_x = call i32 @__xla_musa_v1_read_nctaid_x()
  %nctaid_y = call i32 @__xla_musa_v1_read_nctaid_y()
  %nctaid_z = call i32 @__xla_musa_v1_read_nctaid_z()
  %ntid_x = call i32 @__xla_musa_v1_read_ntid_x()
  %ntid_y = call i32 @__xla_musa_v1_read_ntid_y()
  %ntid_z = call i32 @__xla_musa_v1_read_ntid_z()
  %tid_x = call i32 @__xla_musa_v1_read_tid_x()
  %tid_y = call i32 @__xla_musa_v1_read_tid_y()
  %tid_z = call i32 @__xla_musa_v1_read_tid_z()
  %read_lane = call i32 @__xla_musa_v1_subgroup_read_lane_i32(i32 %tid_x, i32 0)
  call void @__xla_musa_v1_workgroup_barrier()
  ret void
}

declare i32 @__xla_musa_v1_clock32() #0
declare i64 @__xla_musa_v1_clock64() #0
declare i32 @__xla_musa_v1_read_ctaid_x() #1
declare i32 @__xla_musa_v1_read_ctaid_y() #1
declare i32 @__xla_musa_v1_read_ctaid_z() #1
declare i32 @__xla_musa_v1_read_nctaid_x() #2
declare i32 @__xla_musa_v1_read_nctaid_y() #2
declare i32 @__xla_musa_v1_read_nctaid_z() #2
declare i32 @__xla_musa_v1_read_ntid_x() #2
declare i32 @__xla_musa_v1_read_ntid_y() #2
declare i32 @__xla_musa_v1_read_ntid_z() #2
declare i32 @__xla_musa_v1_read_tid_x() #2
declare i32 @__xla_musa_v1_read_tid_y() #2
declare i32 @__xla_musa_v1_read_tid_z() #2
declare i32 @__xla_musa_v1_subgroup_read_lane_i32(i32, i32) #4
declare void @__xla_musa_v1_workgroup_barrier() #3

attributes #0 = { inaccessiblememonly nounwind }
attributes #1 = { convergent nounwind readnone }
attributes #2 = { nounwind readnone }
attributes #3 = { convergent nounwind }
attributes #4 = { convergent nounwind readnone willreturn }
