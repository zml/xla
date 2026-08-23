HloModule flydsl_fnuz_fp8_batched_scaled_gemm___B__x__M__x__N__x__K__

gemm {
  lhs = f8e4m3fnuz[__B__,__M__,__K__]{2,1,0} parameter(0)
  rhs = f8e5m2fnuz[__B__,__K__,__N__]{__BATCH_RHS_LAYOUT__} parameter(1)
  lhs_scale = bf16[1,1,1]{2,1,0} parameter(2)
  rhs_scale = bf16[1,1,1]{2,1,0} parameter(3)
  ROOT dot = f32[__B__,__M__,__N__]{2,1,0} scaled-dot(
      lhs, rhs, lhs_scale, rhs_scale),
      lhs_batch_dims={0}, rhs_batch_dims={0},
      lhs_contracting_dims={2}, rhs_contracting_dims={1}
}

ENTRY main {
  lhs = f8e4m3fnuz[__B__,__M__,__K__]{2,1,0} parameter(0)
  rhs = f8e5m2fnuz[__B__,__K__,__N__]{__BATCH_RHS_LAYOUT__} parameter(1)
  lhs_scale = bf16[1,1,1]{2,1,0} parameter(2)
  rhs_scale = bf16[1,1,1]{2,1,0} parameter(3)
  ROOT fusion = f32[__B__,__M__,__N__]{2,1,0} fusion(
      lhs, rhs, lhs_scale, rhs_scale), kind=kCustom, calls=gemm,
      backend_config={"fusion_backend_config":{"kind":"__triton_gemm"}}
}
