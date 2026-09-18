; Copyright 2026 The OpenXLA Authors.
; SPDX-License-Identifier: Apache-2.0

source_filename = "ignored/source/path"
target datalayout = "e-p:64:64:64:64-p1:64:64:64:64-p2:64:64:64:64-p3:32:32-p4:32:32-p5:64:64-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128"
target triple = "mtgpu-mt-musa"

define void @kernel(ptr addrspace(1) noalias align 4 dereferenceable(4) %out) #0 {
entry:
  %tid = call i32 @__xla_musa_v1_read_tid_x()
  store i32 %tid, ptr addrspace(1) %out, align 4
  ret void
}

declare i32 @__xla_musa_v1_read_tid_x() #1

attributes #0 = { "xla.musa.kernel.v1" }
attributes #1 = { nounwind memory(none) }

!llvm.ident = !{!0}
!0 = !{!"current-llvm-producer"}
