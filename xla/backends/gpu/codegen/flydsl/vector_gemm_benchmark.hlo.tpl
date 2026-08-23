HloModule flydsl_f32_dot_column_bias_epilogue___M__x__N__x__K__

gemm {
  lhs = bf16[__M__,__K__]{1,0} parameter(0)
  rhs = bf16[__K__,__N__]{__RHS_LAYOUT__} parameter(1)
  bias = f32[__N__]{0} parameter(2)
  dot = f32[__M__,__N__]{1,0} dot(lhs, rhs),
      lhs_contracting_dims={1}, rhs_contracting_dims={0}
  broadcast = f32[__M__,__N__]{1,0} broadcast(bias), dimensions={1}
  add = f32[__M__,__N__]{1,0} add(dot, broadcast)
  ROOT convert = bf16[__M__,__N__]{1,0} convert(add)
}

ENTRY main {
  lhs = bf16[__M__,__K__]{1,0} parameter(0)
  rhs = bf16[__K__,__N__]{__RHS_LAYOUT__} parameter(1)
  bias = f32[__N__]{0} parameter(2)
  ROOT fusion = bf16[__M__,__N__]{1,0} fusion(lhs, rhs, bias),
      kind=kCustom, calls=gemm,
      backend_config={"fusion_backend_config":{"kind":"__triton_gemm"}}
}
