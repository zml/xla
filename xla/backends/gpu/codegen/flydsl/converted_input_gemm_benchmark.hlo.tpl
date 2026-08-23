HloModule flydsl_f32_inputs_bf16_gemm___M__x__N__x__K__

gemm {
  lhs_f32 = f32[__M__,__K__]{1,0} parameter(0)
  rhs_f32 = f32[__K__,__N__]{__RHS_LAYOUT__} parameter(1)
  lhs = bf16[__M__,__K__]{1,0} convert(lhs_f32)
  rhs = bf16[__K__,__N__]{__RHS_LAYOUT__} convert(rhs_f32)
  dot = f32[__M__,__N__]{1,0} dot(lhs, rhs),
      lhs_contracting_dims={1}, rhs_contracting_dims={0}
  ROOT convert = bf16[__M__,__N__]{1,0} convert(dot)
}

ENTRY main {
  lhs = f32[__M__,__K__]{1,0} parameter(0)
  rhs = f32[__K__,__N__]{__RHS_LAYOUT__} parameter(1)
  ROOT fusion = bf16[__M__,__N__]{1,0} fusion(lhs, rhs),
      kind=kCustom, calls=gemm,
      backend_config={"fusion_backend_config":{"kind":"__triton_gemm"}}
}
