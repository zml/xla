HloModule flydsl_f32_dot_epilogue_chain___M__x__N__x__K__

gemm {
  lhs = bf16[__M__,__K__]{1,0} parameter(0)
  rhs = bf16[__K__,__N__]{__RHS_LAYOUT__} parameter(1)
  scale = f32[__N__]{0} parameter(2)
  bias = f32[] parameter(3)
  dot = f32[__M__,__N__]{1,0} dot(lhs, rhs),
      lhs_contracting_dims={1}, rhs_contracting_dims={0}
  scale_broadcast = f32[__M__,__N__]{1,0} broadcast(scale), dimensions={1}
  multiply = f32[__M__,__N__]{1,0} multiply(dot, scale_broadcast)
  bias_broadcast = f32[__M__,__N__]{1,0} broadcast(bias), dimensions={}
  add = f32[__M__,__N__]{1,0} add(multiply, bias_broadcast)
  negate = f32[__M__,__N__]{1,0} negate(add)
  ROOT convert = bf16[__M__,__N__]{1,0} convert(negate)
}

ENTRY main {
  lhs = bf16[__M__,__K__]{1,0} parameter(0)
  rhs = bf16[__K__,__N__]{__RHS_LAYOUT__} parameter(1)
  scale = f32[__N__]{0} parameter(2)
  bias = f32[] parameter(3)
  ROOT fusion = bf16[__M__,__N__]{1,0} fusion(lhs, rhs, scale, bias),
      kind=kCustom, calls=gemm,
      backend_config={"fusion_backend_config":{"kind":"__triton_gemm"}}
}
