; Hand-authored textual AIR for a single fast FP32 matmul kernel.
;
; This file intentionally does not depend on MSL source. It can be assembled
; directly with `air-as` and linked with `metallib`.

source_filename = "xla/service/gpu/metal_air_experiment/matmul_air_direct.ll"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v27-apple-macosx15.0.0"

%struct.MatmulParams = type { i32, i32, i32, i32 }

define void @matmul_simdgroup_8x8(
    float addrspace(1)* nocapture noundef readonly "air-buffer-no-alias" %a,
    float addrspace(1)* nocapture noundef readonly "air-buffer-no-alias" %b,
    float addrspace(1)* nocapture noundef writeonly "air-buffer-no-alias" %c,
    %struct.MatmulParams addrspace(2)* nocapture noundef readonly align 4 dereferenceable(16) "air-buffer-no-alias" %params,
    <3 x i32> noundef %group_id,
    i32 noundef %simdgroup_id) local_unnamed_addr #0 {
entry:
  %local_tile_y = lshr i32 %simdgroup_id, 2
  %group_y = extractelement <3 x i32> %group_id, i64 1
  %tile_y_base = shl i32 %group_y, 1
  %tile_y = add i32 %tile_y_base, %local_tile_y
  %row = shl i32 %tile_y, 3
  %group_x = extractelement <3 x i32> %group_id, i64 0
  %group_col_base = shl i32 %group_x, 5
  %sg_shifted = shl i32 %simdgroup_id, 3
  %local_col = and i32 %sg_shifted, 24
  %col = or i32 %group_col_base, %local_col
  %zero = tail call fast <64 x float> @air.simdgroup_matrix_8x8_init_filled.v64f32.f32(float 0.000000e+00) #3
  %k_ptr = getelementptr inbounds %struct.MatmulParams, %struct.MatmulParams addrspace(2)* %params, i64 0, i32 2
  %k = load i32, i32 addrspace(2)* %k_ptr, align 4
  %n_ptr = getelementptr inbounds %struct.MatmulParams, %struct.MatmulParams addrspace(2)* %params, i64 0, i32 1
  %n = load i32, i32 addrspace(2)* %n_ptr, align 4
  %k_is_zero = icmp eq i32 %k, 0
  br i1 %k_is_zero, label %store, label %loop

loop:
  %base = phi i32 [ 0, %entry ], [ %next_base, %loop ]
  %acc = phi <64 x float> [ %zero, %entry ], [ %acc_next, %loop ]
  %a_row_offset = mul i32 %row, %k
  %a_index32 = add i32 %a_row_offset, %base
  %a_index = zext i32 %a_index32 to i64
  %a_ptr = getelementptr inbounds float, float addrspace(1)* %a, i64 %a_index
  %k64 = zext i32 %k to i64
  %a_tile = tail call fast <64 x float> @air.simdgroup_matrix_8x8_load.v64f32.p1f32(float addrspace(1)* nocapture readonly %a_ptr, i64 %k64, <2 x i64> zeroinitializer, i1 false) #4
  %b_row_offset = mul i32 %base, %n
  %b_index32 = add i32 %b_row_offset, %col
  %b_index = zext i32 %b_index32 to i64
  %b_ptr = getelementptr inbounds float, float addrspace(1)* %b, i64 %b_index
  %n64 = zext i32 %n to i64
  %b_tile = tail call fast <64 x float> @air.simdgroup_matrix_8x8_load.v64f32.p1f32(float addrspace(1)* nocapture readonly %b_ptr, i64 %n64, <2 x i64> zeroinitializer, i1 false) #4
  %acc_next = tail call fast <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f32.v64f32.v64f32(<64 x float> %a_tile, <64 x float> %b_tile, <64 x float> %acc) #3
  %next_base = add i32 %base, 8
  %more = icmp ult i32 %next_base, %k
  br i1 %more, label %loop, label %store

store:
  %final_acc = phi <64 x float> [ %zero, %entry ], [ %acc_next, %loop ]
  %c_row_offset = mul i32 %row, %n
  %c_index32 = add i32 %c_row_offset, %col
  %c_index = zext i32 %c_index32 to i64
  %c_ptr = getelementptr inbounds float, float addrspace(1)* %c, i64 %c_index
  %n64_store = zext i32 %n to i64
  tail call void @air.simdgroup_matrix_8x8_store.v64f32.p1f32(<64 x float> %final_acc, float addrspace(1)* nocapture writeonly %c_ptr, i64 %n64_store, <2 x i64> zeroinitializer, i1 false) #5
  ret void
}

define void @matmul_relu_simdgroup_8x8(
    float addrspace(1)* nocapture noundef readonly "air-buffer-no-alias" %a,
    float addrspace(1)* nocapture noundef readonly "air-buffer-no-alias" %b,
    float addrspace(1)* nocapture noundef writeonly "air-buffer-no-alias" %c,
    %struct.MatmulParams addrspace(2)* nocapture noundef readonly align 4 dereferenceable(16) "air-buffer-no-alias" %params,
    <3 x i32> noundef %group_id,
    i32 noundef %simdgroup_id) local_unnamed_addr #0 {
entry:
  %local_tile_y = lshr i32 %simdgroup_id, 2
  %group_y = extractelement <3 x i32> %group_id, i64 1
  %tile_y_base = shl i32 %group_y, 1
  %tile_y = add i32 %tile_y_base, %local_tile_y
  %row = shl i32 %tile_y, 3
  %group_x = extractelement <3 x i32> %group_id, i64 0
  %group_col_base = shl i32 %group_x, 5
  %sg_shifted = shl i32 %simdgroup_id, 3
  %local_col = and i32 %sg_shifted, 24
  %col = or i32 %group_col_base, %local_col
  %zero = tail call fast <64 x float> @air.simdgroup_matrix_8x8_init_filled.v64f32.f32(float 0.000000e+00) #3
  %k_ptr = getelementptr inbounds %struct.MatmulParams, %struct.MatmulParams addrspace(2)* %params, i64 0, i32 2
  %k = load i32, i32 addrspace(2)* %k_ptr, align 4
  %n_ptr = getelementptr inbounds %struct.MatmulParams, %struct.MatmulParams addrspace(2)* %params, i64 0, i32 1
  %n = load i32, i32 addrspace(2)* %n_ptr, align 4
  %k_is_zero = icmp eq i32 %k, 0
  br i1 %k_is_zero, label %store, label %loop

loop:
  %base = phi i32 [ 0, %entry ], [ %next_base, %loop ]
  %acc = phi <64 x float> [ %zero, %entry ], [ %acc_next, %loop ]
  %a_row_offset = mul i32 %row, %k
  %a_index32 = add i32 %a_row_offset, %base
  %a_index = zext i32 %a_index32 to i64
  %a_ptr = getelementptr inbounds float, float addrspace(1)* %a, i64 %a_index
  %k64 = zext i32 %k to i64
  %a_tile = tail call fast <64 x float> @air.simdgroup_matrix_8x8_load.v64f32.p1f32(float addrspace(1)* nocapture readonly %a_ptr, i64 %k64, <2 x i64> zeroinitializer, i1 false) #4
  %b_row_offset = mul i32 %base, %n
  %b_index32 = add i32 %b_row_offset, %col
  %b_index = zext i32 %b_index32 to i64
  %b_ptr = getelementptr inbounds float, float addrspace(1)* %b, i64 %b_index
  %n64 = zext i32 %n to i64
  %b_tile = tail call fast <64 x float> @air.simdgroup_matrix_8x8_load.v64f32.p1f32(float addrspace(1)* nocapture readonly %b_ptr, i64 %n64, <2 x i64> zeroinitializer, i1 false) #4
  %acc_next = tail call fast <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f32.v64f32.v64f32(<64 x float> %a_tile, <64 x float> %b_tile, <64 x float> %acc) #3
  %next_base = add i32 %base, 8
  %more = icmp ult i32 %next_base, %k
  br i1 %more, label %loop, label %store

store:
  %final_acc = phi <64 x float> [ %zero, %entry ], [ %acc_next, %loop ]
  %relu_mask = fcmp fast ogt <64 x float> %final_acc, zeroinitializer
  %relu_acc = select <64 x i1> %relu_mask, <64 x float> %final_acc, <64 x float> zeroinitializer
  %c_row_offset = mul i32 %row, %n
  %c_index32 = add i32 %c_row_offset, %col
  %c_index = zext i32 %c_index32 to i64
  %c_ptr = getelementptr inbounds float, float addrspace(1)* %c, i64 %c_index
  %n64_store = zext i32 %n to i64
  tail call void @air.simdgroup_matrix_8x8_store.v64f32.p1f32(<64 x float> %relu_acc, float addrspace(1)* nocapture writeonly %c_ptr, i64 %n64_store, <2 x i64> zeroinitializer, i1 false) #5
  ret void
}

declare <64 x float> @air.simdgroup_matrix_8x8_init_filled.v64f32.f32(float) local_unnamed_addr #1
declare <64 x float> @air.simdgroup_matrix_8x8_load.v64f32.p1f32(float addrspace(1)* nocapture readonly, i64, <2 x i64>, i1) local_unnamed_addr #2
declare <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f32.v64f32.v64f32(<64 x float>, <64 x float>, <64 x float>) local_unnamed_addr #1
declare void @air.simdgroup_matrix_8x8_store.v64f32.p1f32(<64 x float>, float addrspace(1)* nocapture writeonly, i64, <2 x i64>, i1) local_unnamed_addr #6

attributes #0 = { convergent mustprogress nounwind "approx-func-fp-math"="true" "frame-pointer"="all" "min-legal-vector-width"="2048" "no-builtins" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="true" }
attributes #1 = { convergent mustprogress nounwind willreturn }
attributes #2 = { convergent mustprogress nofree nounwind readonly willreturn }
attributes #3 = { convergent nounwind willreturn }
attributes #4 = { convergent nounwind readonly willreturn }
attributes #5 = { convergent nounwind willreturn writeonly }
attributes #6 = { convergent mustprogress nounwind willreturn writeonly }

!air.kernel = !{!0, !25}
!llvm.module.flags = !{!10, !11, !12, !13, !14, !15, !16, !17}
!air.compile_options = !{!18, !19, !20}
!llvm.ident = !{!21}
!air.version = !{!22}
!air.language_version = !{!23}
!air.source_file_name = !{!24}

!0 = !{void (float addrspace(1)*, float addrspace(1)*, float addrspace(1)*, %struct.MatmulParams addrspace(2)*, <3 x i32>, i32)* @matmul_simdgroup_8x8, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5, !6, !8, !9}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"a"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"b"}
!5 = !{i32 2, !"air.buffer", !"air.location_index", i32 2, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"c"}
!6 = !{i32 3, !"air.buffer", !"air.buffer_size", i32 16, !"air.location_index", i32 3, i32 1, !"air.read", !"air.address_space", i32 2, !"air.struct_type_info", !7, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"MatmulParams", !"air.arg_name", !"params"}
!7 = !{i32 0, i32 4, i32 0, !"uint", !"m", i32 4, i32 4, i32 0, !"uint", !"n", i32 8, i32 4, i32 0, !"uint", !"k", i32 12, i32 4, i32 0, !"uint", !"reserved"}
!8 = !{i32 4, !"air.threadgroup_position_in_grid", !"air.arg_type_name", !"uint3", !"air.arg_name", !"group_id"}
!9 = !{i32 5, !"air.simdgroup_index_in_threadgroup", !"air.arg_type_name", !"uint", !"air.arg_name", !"simdgroup_id"}
!10 = !{i32 1, !"wchar_size", i32 4}
!11 = !{i32 7, !"air.max_device_buffers", i32 31}
!12 = !{i32 7, !"air.max_constant_buffers", i32 31}
!13 = !{i32 7, !"air.max_threadgroup_buffers", i32 31}
!14 = !{i32 7, !"air.max_textures", i32 128}
!15 = !{i32 7, !"air.max_read_write_textures", i32 8}
!16 = !{i32 7, !"air.max_samplers", i32 16}
!17 = !{i32 7, !"frame-pointer", i32 2}
!18 = !{!"air.compile.denorms_disable"}
!19 = !{!"air.compile.fast_math_enable"}
!20 = !{!"air.compile.framebuffer_fetch_enable"}
!21 = !{!"xla direct AIR experiment"}
!22 = !{i32 2, i32 7, i32 0}
!23 = !{!"Metal", i32 3, i32 2, i32 0}
!24 = !{!"xla/service/gpu/metal_air_experiment/matmul_air_direct.ll"}
!25 = !{void (float addrspace(1)*, float addrspace(1)*, float addrspace(1)*, %struct.MatmulParams addrspace(2)*, <3 x i32>, i32)* @matmul_relu_simdgroup_8x8, !1, !2}
