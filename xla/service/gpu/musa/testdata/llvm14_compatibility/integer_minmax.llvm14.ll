; ModuleID = 'integer_minmax_profile'
source_filename = "integer_minmax_profile"
target datalayout = "e-p:64:64:64:64-p1:64:64:64:64-p2:64:64:64:64-p3:32:32-p4:32:32-p5:64:64-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128"
target triple = "mtgpu-mt-musa"

define void @kernel(ptr addrspace(1) %out) {
entry:
  %smin = call i32 @llvm.smin.i32(i32 7, i32 3)
  %smax = call i32 @llvm.smax.i32(i32 %smin, i32 5)
  %umin = call i32 @llvm.umin.i32(i32 %smax, i32 4)
  %umax = call i32 @llvm.umax.i32(i32 %umin, i32 6)
  store i32 %umax, ptr addrspace(1) %out, align 4
  ret void
}

declare i32 @llvm.smin.i32(i32, i32) #0

declare i32 @llvm.smax.i32(i32, i32) #0

declare i32 @llvm.umin.i32(i32, i32) #0

declare i32 @llvm.umax.i32(i32, i32) #0

attributes #0 = { nofree nosync nounwind speculatable willreturn readnone }
