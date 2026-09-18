; Copyright 2026 The OpenXLA Authors.
; SPDX-License-Identifier: Apache-2.0

source_filename = "raw_vendor_intrinsic"
target datalayout = "e-p:64:64:64:64-p1:64:64:64:64-p2:64:64:64:64-p3:32:32-p4:32:32-p5:64:64-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128"
target triple = "mtgpu-mt-musa"

define void @kernel() {
entry:
  %value = call i32 @llvm.musa.read.ptx.sreg.tid.x()
  ret void
}

declare i32 @llvm.musa.read.ptx.sreg.tid.x()
