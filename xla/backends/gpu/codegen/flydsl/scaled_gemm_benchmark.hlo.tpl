HloModule flydsl_bf16_scaled_gemm___M__x__N__x__K__

gemm {
  lhs = bf16[__M__,__K__]{1,0} parameter(0)
  rhs = bf16[__K__,__N__]{__RHS_LAYOUT__} parameter(1)
  lhs_scale = bf16[1,1]{1,0} parameter(2)
  rhs_scale = bf16[1,1]{1,0} parameter(3)
  ROOT dot = bf16[__M__,__N__]{1,0} scaled-dot(
      lhs, rhs, lhs_scale, rhs_scale),
      lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

ENTRY main {
  lhs = bf16[__M__,__K__]{1,0} parameter(0)
  rhs = bf16[__K__,__N__]{__RHS_LAYOUT__} parameter(1)
  lhs_scale = bf16[1,1]{1,0} parameter(2)
  rhs_scale = bf16[1,1]{1,0} parameter(3)
  ROOT fusion = bf16[__M__,__N__]{1,0} fusion(
      lhs, rhs, lhs_scale, rhs_scale),
      kind=kCustom, calls=gemm,
      backend_config={"fusion_backend_config":{"kind":"__triton_gemm"}}
}
