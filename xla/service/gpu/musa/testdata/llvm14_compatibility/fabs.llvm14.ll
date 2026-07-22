; ModuleID = 'fabs_profile'
source_filename = "fabs_profile"
target datalayout = "e-p:64:64:64:64-p1:64:64:64:64-p2:64:64:64:64-p3:32:32-p4:32:32-p5:64:64-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128"
target triple = "mtgpu-mt-musa"

define void @kernel(ptr addrspace(1) %out) {
entry:
  %value = call float @llvm.fabs.f32(float -2.000000e+00)
  %bits = bitcast float %value to i32
  store i32 %bits, ptr addrspace(1) %out, align 4
  ret void
}

declare float @llvm.fabs.f32(float) #0

attributes #0 = { nofree nosync nounwind speculatable willreturn readnone }
