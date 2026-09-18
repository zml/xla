; ModuleID = 'floating_minmax_profile'
source_filename = "floating_minmax_profile"
target datalayout = "e-p:64:64:64:64-p1:64:64:64:64-p2:64:64:64:64-p3:32:32-p4:32:32-p5:64:64-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128"
target triple = "mtgpu-mt-musa"

define void @kernel(ptr addrspace(1) %out, float %lhs, float %rhs) {
entry:
  %minimum.ordered_compare = fcmp olt float %lhs, %rhs
  %minimum.ordered_value = select i1 %minimum.ordered_compare, float %lhs, float %rhs
  %minimum.equal = fcmp oeq float %lhs, %rhs
  %minimum.lhs_bits = bitcast float %lhs to i32
  %minimum.rhs_bits = bitcast float %rhs to i32
  %minimum.equal_bits = or i32 %minimum.lhs_bits, %minimum.rhs_bits
  %minimum.equal_value = bitcast i32 %minimum.equal_bits to float
  %minimum.zero_ordered_value = select i1 %minimum.equal, float %minimum.equal_value, float %minimum.ordered_value
  %minimum.canonical_nan = bitcast i32 2143289344 to float
  %minimum.unordered = fcmp uno float %lhs, %rhs
  %minimum.result = select i1 %minimum.unordered, float %minimum.canonical_nan, float %minimum.zero_ordered_value
  %maximum.ordered_compare = fcmp ogt float %minimum.result, %rhs
  %maximum.ordered_value = select i1 %maximum.ordered_compare, float %minimum.result, float %rhs
  %maximum.equal = fcmp oeq float %minimum.result, %rhs
  %maximum.lhs_bits = bitcast float %minimum.result to i32
  %maximum.rhs_bits = bitcast float %rhs to i32
  %maximum.equal_bits = and i32 %maximum.lhs_bits, %maximum.rhs_bits
  %maximum.equal_value = bitcast i32 %maximum.equal_bits to float
  %maximum.zero_ordered_value = select i1 %maximum.equal, float %maximum.equal_value, float %maximum.ordered_value
  %maximum.canonical_nan = bitcast i32 2143289344 to float
  %maximum.unordered = fcmp uno float %minimum.result, %rhs
  %maximum.result = select i1 %maximum.unordered, float %maximum.canonical_nan, float %maximum.zero_ordered_value
  %minnum.ordered_compare = fcmp olt float %maximum.result, %lhs
  %minnum.ordered_value = select i1 %minnum.ordered_compare, float %maximum.result, float %lhs
  %minnum.equal = fcmp oeq float %maximum.result, %lhs
  %minnum.lhs_bits = bitcast float %maximum.result to i32
  %minnum.rhs_bits = bitcast float %lhs to i32
  %minnum.equal_bits = or i32 %minnum.lhs_bits, %minnum.rhs_bits
  %minnum.equal_value = bitcast i32 %minnum.equal_bits to float
  %minnum.zero_ordered_value = select i1 %minnum.equal, float %minnum.equal_value, float %minnum.ordered_value
  %minnum.lhs_nan = fcmp uno float %maximum.result, %maximum.result
  %minnum.rhs_nan = fcmp uno float %lhs, %lhs
  %minnum.rhs_checked_value = select i1 %minnum.rhs_nan, float %maximum.result, float %minnum.zero_ordered_value
  %minnum.result = select i1 %minnum.lhs_nan, float %lhs, float %minnum.rhs_checked_value
  %maxnum.ordered_compare = fcmp ogt float %minnum.result, %rhs
  %maxnum.ordered_value = select i1 %maxnum.ordered_compare, float %minnum.result, float %rhs
  %maxnum.equal = fcmp oeq float %minnum.result, %rhs
  %maxnum.lhs_bits = bitcast float %minnum.result to i32
  %maxnum.rhs_bits = bitcast float %rhs to i32
  %maxnum.equal_bits = and i32 %maxnum.lhs_bits, %maxnum.rhs_bits
  %maxnum.equal_value = bitcast i32 %maxnum.equal_bits to float
  %maxnum.zero_ordered_value = select i1 %maxnum.equal, float %maxnum.equal_value, float %maxnum.ordered_value
  %maxnum.lhs_nan = fcmp uno float %minnum.result, %minnum.result
  %maxnum.rhs_nan = fcmp uno float %rhs, %rhs
  %maxnum.rhs_checked_value = select i1 %maxnum.rhs_nan, float %minnum.result, float %maxnum.zero_ordered_value
  %maxnum.result = select i1 %maxnum.lhs_nan, float %rhs, float %maxnum.rhs_checked_value
  %bits = bitcast float %maxnum.result to i32
  store i32 %bits, ptr addrspace(1) %out, align 4
  ret void
}
