HloModule flydsl_int4_gemm___M__x__N__x__K__

gemm {
  lhs = bf16[__M__,__K__]{1,0} parameter(0)
  rhs.s4 = s4[__K__,__N__]{__RHS_LAYOUT__:E(4)} parameter(1)
  rhs.s8 = s8[__K__,__N__]{__RHS_LAYOUT__} convert(rhs.s4)
  rhs.bf16 = bf16[__K__,__N__]{__RHS_LAYOUT__} convert(rhs.s8)
  ROOT dot = f32[__M__,__N__]{1,0} dot(lhs, rhs.bf16),
      lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

ENTRY main {
  lhs = bf16[__M__,__K__]{1,0} parameter(0)
  rhs = s4[__K__,__N__]{__RHS_LAYOUT__:E(4)} parameter(1)
  ROOT fusion = f32[__M__,__N__]{1,0} fusion(lhs, rhs), kind=kCustom,
      calls=gemm,
      backend_config={"fusion_backend_config":{
        "kind":"__triton_gemm"}}
}
