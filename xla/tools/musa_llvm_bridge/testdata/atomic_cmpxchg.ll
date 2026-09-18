; Copyright 2026 The OpenXLA Authors.
; SPDX-License-Identifier: Apache-2.0

source_filename = "atomic_cmpxchg"
target datalayout = "e-p:64:64:64:64-p1:64:64:64:64-p2:64:64:64:64-p3:32:32-p4:32:32-p5:64:64-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128"
target triple = "mtgpu-mt-musa"

define void @kernel(ptr addrspace(1) %out) {
entry:
  %pair = cmpxchg ptr addrspace(1) %out, i32 0, i32 1 monotonic monotonic, align 4
  ret void
}
