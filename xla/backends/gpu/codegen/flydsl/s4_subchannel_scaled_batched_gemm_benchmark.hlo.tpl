HloModule flydsl_s4_subchannel_scaled_batched_gemm___B__x__M__x__N__x__K__

gemm {
  weights.s4 = s4[__B__,__K__,__M__]{2,1,0:E(4)} parameter(0)
  weights.s8 = s8[__B__,__K__,__M__]{2,1,0} convert(weights.s4)
  weights.groups.s8 = s8[__B__,__KG__,256,__M__]{3,2,1,0}
      bitcast(weights.s8)
  weights.groups = bf16[__B__,__KG__,256,__M__]{3,2,1,0}
      convert(weights.groups.s8)
  scales = bf16[__B__,__KG__,1,__M__]{3,2,1,0} parameter(1)
  scales.view = bf16[__B__,__KG__,__M__]{2,1,0} bitcast(scales)
  scales.broadcast = bf16[__B__,__KG__,256,__M__]{3,2,1,0}
      broadcast(scales.view), dimensions={0,1,3}
  weights.scaled.groups = bf16[__B__,__KG__,256,__M__]{3,2,1,0}
      multiply(weights.groups, scales.broadcast)
  weights.scaled = bf16[__B__,__K__,__M__]{2,1,0}
      bitcast(weights.scaled.groups)
  activations = bf16[__N__,__B__,1,__K__]{3,2,1,0} parameter(2)
  activations.view = bf16[__N__,__B__,__K__]{2,1,0}
      bitcast(activations)
  ROOT dot = f32[__B__,__M__,__N__]{2,1,0}
      dot(weights.scaled, activations.view),
      lhs_batch_dims={0}, lhs_contracting_dims={1},
      rhs_batch_dims={1}, rhs_contracting_dims={2}
}

ENTRY main {
  weights = s4[__B__,__K__,__M__]{2,1,0:E(4)} parameter(0)
  scales = bf16[__B__,__KG__,1,__M__]{3,2,1,0} parameter(1)
  activations = bf16[__N__,__B__,1,__K__]{3,2,1,0} parameter(2)
  ROOT fusion = f32[__B__,__M__,__N__]{2,1,0}
      fusion(weights, scales, activations), kind=kCustom, calls=gemm,
      backend_config={"fusion_backend_config":{"kind":"__triton_gemm"}}
}
