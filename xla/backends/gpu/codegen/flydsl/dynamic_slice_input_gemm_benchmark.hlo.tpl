HloModule flydsl_bf16_dynamic_slice_inputs_gemm___M__x__N__x__K__

gemm {
  lhs_physical = bf16[__MP__,__KP__]{1,0} parameter(0)
  rhs_physical = bf16[__KP__,__NP__]{__RHS_LAYOUT__} parameter(1)
  lhs_start_m = s32[] parameter(2)
  lhs_start_k = s32[] parameter(3)
  rhs_start_k = s32[] parameter(4)
  rhs_start_n = s32[] parameter(5)
  lhs = bf16[__M__,__K__]{1,0} dynamic-slice(
      lhs_physical, lhs_start_m, lhs_start_k),
      dynamic_slice_sizes={__M__,__K__}
  rhs = bf16[__K__,__N__]{__RHS_LAYOUT__} dynamic-slice(
      rhs_physical, rhs_start_k, rhs_start_n),
      dynamic_slice_sizes={__K__,__N__}
  ROOT dot = bf16[__M__,__N__]{1,0} dot(lhs, rhs),
      lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

ENTRY main {
  lhs = bf16[__MP__,__KP__]{1,0} parameter(0)
  rhs = bf16[__KP__,__NP__]{__RHS_LAYOUT__} parameter(1)
  lhs_start_m = s32[] parameter(2)
  lhs_start_k = s32[] parameter(3)
  rhs_start_k = s32[] parameter(4)
  rhs_start_n = s32[] parameter(5)
  ROOT fusion = bf16[__M__,__N__]{1,0} fusion(
      lhs, rhs, lhs_start_m, lhs_start_k, rhs_start_k, rhs_start_n),
      kind=kCustom, calls=gemm,
      backend_config={"fusion_backend_config":{"kind":"__triton_gemm"}}
}
