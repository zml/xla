// The XTile module TileAndEmitXTileModule emits for a canonical NVFP4
// block-scaled matmul, captured verbatim from the shared emitter.
//
// RUN: xtile_to_tilebc %s | FileCheck %s

#indexing_map = #xla.indexing_map<"(pid)[k] -> (pid * 4 + k), domain: pid in [0, 1], k in [0, 3]">
#indexing_map1 = #xla.indexing_map<"(pid_0) -> ((pid_0 mod 4) * 128), domain: pid_0 in [0, 7]">
#indexing_map2 = #xla.indexing_map<"(pid_0) -> ((pid_0 / 4) * 128), domain: pid_0 in [0, 7]">
#indexing_map3 = #xla.indexing_map<"(pid_0) -> ((pid_0 mod 4) * 8), domain: pid_0 in [0, 7]">
#indexing_map4 = #xla.indexing_map<"(pid_0) -> (pid_0 * 128), domain: pid_0 in [0, 1]">
module {
  xtile.entry_func @triton_fn(%arg0: memref<128x256xi8>, %arg1: memref<512x128xi8>, %arg2: memref<128x32xf8E4M3FN>, %arg3: memref<32x256xf8E4M3FN>, %arg4: memref<128x256xf32>, %arg5: index) attributes {num_opaque_args = 0 : i32} {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32>
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %0 = scf.for %arg6 = %c0 to %c4 step %c1 iter_args(%arg7 = %cst) -> (tensor<128x128xf32>) {
      %2 = xla.apply_indexing #indexing_map(%arg5)[%arg6]
      %c0_1 = arith.constant 0 : index
      %3 = xla.apply_indexing #indexing_map1(%2)
      %c2 = arith.constant 2 : index
      %4 = arith.divsi %3, %c2 : index
      %5 = xtile.extract %arg0[%c0_1, %4] [128, 64] [1, 1] : memref<128x256xi8> -> tensor<128x64xi8>
      %6 = xla.apply_indexing #indexing_map1(%2)
      %7 = xla.apply_indexing #indexing_map2(%2)
      %c2_2 = arith.constant 2 : index
      %8 = arith.divsi %7, %c2_2 : index
      %9 = xtile.extract %arg1[%6, %8] [128, 64] [1, 1] : memref<512x128xi8> -> tensor<128x64xi8>
      %c0_3 = arith.constant 0 : index
      %10 = xla.apply_indexing #indexing_map3(%2)
      %11 = xtile.extract %arg2[%c0_3, %10] [128, 8] [1, 1] : memref<128x32xf8E4M3FN> -> tensor<128x8xf8E4M3FN>
      %12 = xla.apply_indexing #indexing_map3(%2)
      %13 = xla.apply_indexing #indexing_map2(%2)
      %14 = xtile.extract %arg3[%12, %13] [8, 128] [1, 1] : memref<32x256xf8E4M3FN> -> tensor<8x128xf8E4M3FN>
      %15 = stablehlo.transpose %14, dims = [1, 0] : (tensor<8x128xf8E4M3FN>) -> tensor<128x8xf8E4M3FN>
      %16 = xtile.dot_scaled %5 scale %11, %9 scale %15 {dot_dimension_numbers = #stablehlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>, fastMath = true, lhs_elem_type = f4E2M1FN, rhs_elem_type = f4E2M1FN, rhs_k_pack = false} : tensor<128x64xi8>, tensor<128x8xf8E4M3FN> * tensor<128x64xi8>, tensor<128x8xf8E4M3FN> -> tensor<128x128xf32>
      %17 = arith.addf %arg7, %16 : tensor<128x128xf32>
      scf.yield %17 : tensor<128x128xf32>
    }
    %c0_0 = arith.constant 0 : index
    %1 = xla.apply_indexing #indexing_map4(%arg5)
    xtile.insert %0 into %arg4[%c0_0, %1] [128, 128] [1, 1] : tensor<128x128xf32> -> memref<128x256xf32>
    xtile.return
  }
}

// The kernel takes buffer pointers only; every shape and stride is static and
// lives in the tensor_view types, because that is what XLA's KernelThunk passes.
// CHECK: cuda_tile.module @kernels
// CHECK: entry @triton_fn(
// CHECK-SAME: tile<ptr<i8>>
// CHECK-SAME: tile<ptr<f8E4M3FN>>
// CHECK-SAME: tile<ptr<f32>>

// CHECK: make_tensor_view {{.*}} : tensor_view<128x256xf32, strides=[256,1]>
// CHECK: make_partition_view {{.*}} partition_view<tile=(128x128)

// The single linear XTile tile id becomes the x component of the tile grid.
// CHECK: get_tile_block_id

// The K loop carries the f32 accumulator, exactly as the scf.for did.
// CHECK: %[[FOR:.*]] = for %{{.*}} in ({{.*}}) : tile<i32> iter_values(%[[ACC:.*]] = {{.*}}) -> (tile<128x128xf32>)

// Packed fp4 is unpacked in registers rather than converted.
// CHECK: unpack {{.*}} : tile<8192xi8> -> tile<16384xf4E2M1FN>

// The rhs scale reaches mmaf_scaled as [K/16, N]: the stablehlo.transpose that
// dot_algorithms.cc adds for tt.dot_scaled's [N, K/16] order is peeled off, not
// emitted and undone.
// CHECK: mmaf_scaled {{.*}}, %[[ACC]], {{.*}} : tile<128x128xf4E2M1FN>, tile<128x128xf4E2M1FN>, tile<128x128xf32>, tile<128x8xf8E4M3FN>, tile<8x128xf8E4M3FN>
// CHECK: continue

// CHECK: store_view_tko
// CHECK: return
