HloModule flydsl_f32_concat_inputs_bf16_gemm___M__x__N__x__K__

gemm {
  lhs0_f32 = f32[__MH__,__K__]{1,0} parameter(0)
  lhs1_f32 = f32[__MH__,__K__]{1,0} parameter(1)
  rhs0_f32 = f32[__K__,__NH__]{__RHS_LAYOUT__} parameter(2)
  rhs1_f32 = f32[__K__,__NH__]{__RHS_LAYOUT__} parameter(3)
  lhs0 = bf16[__MH__,__K__]{1,0} convert(lhs0_f32)
  lhs1 = bf16[__MH__,__K__]{1,0} convert(lhs1_f32)
  rhs0 = bf16[__K__,__NH__]{__RHS_LAYOUT__} convert(rhs0_f32)
  rhs1 = bf16[__K__,__NH__]{__RHS_LAYOUT__} convert(rhs1_f32)
  lhs = bf16[__M__,__K__]{1,0} concatenate(lhs0, lhs1), dimensions={0}
  rhs = bf16[__K__,__N__]{__RHS_LAYOUT__} concatenate(rhs0, rhs1), dimensions={1}
  dot = f32[__M__,__N__]{1,0} dot(lhs, rhs),
      lhs_contracting_dims={1}, rhs_contracting_dims={0}
  ROOT convert = bf16[__M__,__N__]{1,0} convert(dot)
}

ENTRY main {
  lhs0 = f32[__MH__,__K__]{1,0} parameter(0)
  lhs1 = f32[__MH__,__K__]{1,0} parameter(1)
  rhs0 = f32[__K__,__NH__]{__RHS_LAYOUT__} parameter(2)
  rhs1 = f32[__K__,__NH__]{__RHS_LAYOUT__} parameter(3)
  ROOT fusion = bf16[__M__,__N__]{1,0} fusion(lhs0, lhs1, rhs0, rhs1),
      kind=kCustom, calls=gemm,
      backend_config={"fusion_backend_config":{"kind":"__triton_gemm"}}
}
