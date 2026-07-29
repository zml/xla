// RUN: xla-translate -split-input-file -mlir-hlo-to-hlo-text -verify-diagnostics %s | FileCheck %s

// Needs ScaledDotOp whitelisted in HloLegalizeToStablehlo + gen_hlo_op_writer.
module @scaled_dot_weight_only {
  func.func @main(%lhs: tensor<512x3840xbf16>, %rhs: tensor<4096x3840xf4E2M1FN>,
                  %lhs_scale: tensor<1x1xbf16>,
                  %rhs_scale: tensor<4096x240xf8E4M3FN>) -> tensor<512x4096xbf16> {
    // CHECK: %[[ARG0:.+]] = bf16[512,3840] parameter(0)
    // CHECK: %[[ARG1:.+]] = f4e2m1fn[4096,3840] parameter(1)
    // CHECK: %[[ARG2:.+]] = bf16[1,1] parameter(2)
    // CHECK: %[[ARG3:.+]] = f8e4m3fn[4096,240] parameter(3)
    // CHECK: bf16[512,4096] scaled-dot(%[[ARG0]], %[[ARG1]], %[[ARG2]], %[[ARG3]]), lhs_contracting_dims={1}, rhs_contracting_dims={1}
    %0 = "mhlo.scaled_dot"(%lhs, %rhs, %lhs_scale, %rhs_scale) {
      dot_dimension_numbers = #mhlo.dot<
        lhs_batching_dimensions = [],
        rhs_batching_dimensions = [],
        lhs_contracting_dimensions = [1],
        rhs_contracting_dimensions = [1]
      >,
      precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]
    } : (tensor<512x3840xbf16>, tensor<4096x3840xf4E2M1FN>, tensor<1x1xbf16>,
         tensor<4096x240xf8E4M3FN>) -> tensor<512x4096xbf16>
    func.return %0 : tensor<512x4096xbf16>
  }
}

// -----

module @scaled_dot_batch {
  func.func @main(%lhs: tensor<4x8x64xf8E4M3FN>, %rhs: tensor<4x16x64xf8E4M3FN>,
                  %lhs_scale: tensor<4x8x4xf8E8M0FNU>,
                  %rhs_scale: tensor<4x16x4xf8E8M0FNU>) -> tensor<4x8x16xf32> {
    // CHECK: %[[ARG0:.+]] = f8e4m3fn[4,8,64] parameter(0)
    // CHECK: %[[ARG1:.+]] = f8e4m3fn[4,16,64] parameter(1)
    // CHECK: %[[ARG2:.+]] = f8e8m0fnu[4,8,4] parameter(2)
    // CHECK: %[[ARG3:.+]] = f8e8m0fnu[4,16,4] parameter(3)
    // CHECK: f32[4,8,16] scaled-dot(%[[ARG0]], %[[ARG1]], %[[ARG2]], %[[ARG3]]), lhs_batch_dims={0}, lhs_contracting_dims={2}, rhs_batch_dims={0}, rhs_contracting_dims={2}
    %0 = "mhlo.scaled_dot"(%lhs, %rhs, %lhs_scale, %rhs_scale) {
      dot_dimension_numbers = #mhlo.dot<
        lhs_batching_dimensions = [0],
        rhs_batching_dimensions = [0],
        lhs_contracting_dimensions = [2],
        rhs_contracting_dimensions = [2]
      >,
      precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]
    } : (tensor<4x8x64xf8E4M3FN>, tensor<4x16x64xf8E4M3FN>, tensor<4x8x4xf8E8M0FNU>,
         tensor<4x16x4xf8E8M0FNU>) -> tensor<4x8x16xf32>
    func.return %0 : tensor<4x8x16xf32>
  }
}
