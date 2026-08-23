HloModule flydsl_fnuz_fp8_scaled_gemm___M__x__N__x__K__

gemm {
  lhs = f8e4m3fnuz[__M__,__K__]{1,0} parameter(0)
  rhs = f8e5m2fnuz[__K__,__N__]{__RHS_LAYOUT__} parameter(1)
  lhs_scale = bf16[1,1]{1,0} parameter(2)
  rhs_scale = bf16[1,1]{1,0} parameter(3)
  ROOT dot = f32[__M__,__N__]{1,0} scaled-dot(
      lhs, rhs, lhs_scale, rhs_scale),
      lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

ENTRY main {
  lhs = f8e4m3fnuz[__M__,__K__]{1,0} parameter(0)
  rhs = f8e5m2fnuz[__K__,__N__]{__RHS_LAYOUT__} parameter(1)
  lhs_scale = bf16[1,1]{1,0} parameter(2)
  rhs_scale = bf16[1,1]{1,0} parameter(3)
  ROOT fusion = f32[__M__,__N__]{1,0} fusion(
      lhs, rhs, lhs_scale, rhs_scale), kind=kCustom, calls=gemm,
      backend_config={"fusion_backend_config":{"kind":"__triton_gemm"}}
}
