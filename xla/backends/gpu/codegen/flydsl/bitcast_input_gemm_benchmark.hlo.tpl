HloModule flydsl_f32_bitcast_inputs_bf16_gemm___M__x__N__x__K__

gemm {
  lhs_f32_physical = f32[__K__,__M__]{1,0} parameter(0)
  rhs_f32_physical = f32[__N__,__K__]{1,0} parameter(1)
  lhs_f32 = f32[__M__,__K__]{1,0} bitcast(lhs_f32_physical)
  lhs = bf16[__M__,__K__]{1,0} convert(lhs_f32)
  rhs_bf16_physical = bf16[__N__,__K__]{1,0} convert(rhs_f32_physical)
  rhs = bf16[__K__,__N__]{0,1} bitcast(rhs_bf16_physical)
  dot = f32[__M__,__N__]{1,0} dot(lhs, rhs),
      lhs_contracting_dims={1}, rhs_contracting_dims={0}
  ROOT convert = bf16[__M__,__N__]{1,0} convert(dot)
}

ENTRY main {
  lhs = f32[__K__,__M__]{1,0} parameter(0)
  rhs = f32[__N__,__K__]{1,0} parameter(1)
  ROOT fusion = bf16[__M__,__N__]{1,0} fusion(lhs, rhs),
      kind=kCustom, calls=gemm,
      backend_config={"fusion_backend_config":{"kind":"__triton_gemm"}}
}
