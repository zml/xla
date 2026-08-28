HloModule flydsl_s4_channel_scaled_multi_batch_gemm___B0__x__B1__x__M__x__N__x__K__

gemm {
  weights.s4 = s4[__B0__,__B1__,__K__,__M__]{3,2,1,0:E(4)} parameter(0)
  weights.s8 = s8[__B0__,__B1__,__K__,__M__]{3,2,1,0}
      convert(weights.s4)
  weights.bf16 = bf16[__B0__,__B1__,__K__,__M__]{3,2,1,0}
      convert(weights.s8)
  scales = bf16[__B0__,__M__]{1,0} parameter(1)
  scales.broadcast = bf16[__B0__,__B1__,__K__,__M__]{3,2,1,0}
      broadcast(scales), dimensions={0,3}
  weights.scaled = bf16[__B0__,__B1__,__K__,__M__]{3,2,1,0}
      multiply(weights.bf16, scales.broadcast)
  activations = bf16[__B0__,__B1__,__K__,__N__]{3,2,1,0} parameter(2)
  ROOT dot = f32[__B1__,__B0__,__M__,__N__]{3,2,1,0}
      dot(weights.scaled, activations),
      lhs_batch_dims={1,0}, rhs_batch_dims={1,0},
      lhs_contracting_dims={2}, rhs_contracting_dims={2}
}

ENTRY main {
  weights = s4[__B0__,__B1__,__K__,__M__]{3,2,1,0:E(4)} parameter(0)
  scales = bf16[__B0__,__M__]{1,0} parameter(1)
  activations = bf16[__B0__,__B1__,__K__,__N__]{3,2,1,0} parameter(2)
  ROOT fusion = f32[__B1__,__B0__,__M__,__N__]{3,2,1,0}
      fusion(weights, scales, activations), kind=kCustom, calls=gemm,
      backend_config={"fusion_backend_config":{"kind":"__triton_gemm"}}
}
