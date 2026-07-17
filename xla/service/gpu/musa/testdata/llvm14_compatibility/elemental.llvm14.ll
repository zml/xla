; ModuleID = 'compatibility_corpus'
source_filename = "compatibility_corpus"
target datalayout = "e-p:64:64:64:64-p1:64:64:64:64-p2:64:64:64:64-p3:32:32-p4:32:32-p5:64:64-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128"
target triple = "mtgpu-mt-musa"

define void @kernel(ptr addrspace(1) %out) {
entry:
  %tid = call i32 @__xla_musa_v1_read_tid_x()
  store i32 %tid, ptr addrspace(1) %out, align 4
  ret void
}

declare i32 @__xla_musa_v1_read_tid_x() #0

attributes #0 = { nounwind readnone }
