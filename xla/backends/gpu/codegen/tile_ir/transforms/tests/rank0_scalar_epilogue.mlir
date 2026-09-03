// An epilogue reading a rank-0 buffer: `out = in * broadcast(scale)`, the shape
// of an NVFP4 alpha epilogue. cuda_tile has no 0-dimension view, so the buffer
// is described as one element of a rank-1 view and the 1xf32 tile reshaped back
// down to rank 0.
//
// RUN: xtile_to_tilebc %s | FileCheck %s

module {
  xtile.entry_func @scalar_epilogue(%arg0: memref<128x128xf32>,
                                    %arg1: memref<f32>,
                                    %arg2: memref<128x128xf32>,
                                    %arg3: index)
      attributes {num_opaque_args = 0 : i32} {
    %c0 = arith.constant 0 : index
    %0 = xtile.extract %arg0[%c0, %c0] [128, 128] [1, 1]
        : memref<128x128xf32> -> tensor<128x128xf32>
    %1 = xtile.extract %arg1[] [] []
        : memref<f32> -> tensor<f32>
    %2 = stablehlo.broadcast_in_dim %1, dims = []
        : (tensor<f32>) -> tensor<128x128xf32>
    %3 = stablehlo.multiply %0, %2 : tensor<128x128xf32>
    xtile.insert %3 into %arg2[%c0, %c0] [128, 128] [1, 1]
        : tensor<128x128xf32> -> memref<128x128xf32>
    xtile.return
  }
}

// The array operands keep the usual 16-byte claim, which is what lets the
// assembler use TMA and vectorise.
// CHECK: assume div_by<16>

// The scalar claims only its own width. Claiming 16 on a rank-0 buffer is a
// claim we cannot back -- it can sit at any naturally-aligned offset -- and the
// assembler vectorises on it until the driver faults inside cuLaunchKernel.
// It is viewed as one element of a rank-1 buffer, never as a 0-dimension view.
// CHECK: assume div_by<4>
// CHECK-NEXT: make_tensor_view %{{.*}}, shape = [1], strides = [1]
// CHECK-NEXT: make_partition_view %{{.*}} : partition_view<tile=(1),

// The load is spelled at that rank too -- one index, a 1xf32 tile -- and
// reshaped back down to the rank-0 tile the broadcast consumes.
// CHECK: load_view_tko{{.*}}partition_view<tile=(1),{{.*}}-> tile<1xf32>
// CHECK-NEXT: reshape %{{.*}} : tile<1xf32> -> tile<f32>
// CHECK: broadcast %{{.*}} -> tile<128x128xf32>
