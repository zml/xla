// Copyright 2026 The OpenXLA Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ==============================================================================
// RUN: xla-translate -split-input-file -mlir-hlo-to-hlo-text -verify-diagnostics %s | FileCheck %s

// This test covers the whole export path at once: xla-translate ->
// ConvertMlirHloToHlo first runs createHloLegalizeToStablehloPass
// (allow_xla_features_=true), so it fails unless mhlo::ScaledDotOp is
// whitelisted there AND the op is in both gen_hlo_op_writer.td lists AND the
// ScaledDotOp ExportXlaOp exists.
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
