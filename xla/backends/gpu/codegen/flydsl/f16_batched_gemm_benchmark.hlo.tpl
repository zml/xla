HloModule flydsl_f16_batched_gemm___B__x__M__x__N__x__K__

gemm {
  lhs = f16[__B__,__M__,__K__]{2,1,0} parameter(0)
  rhs = f16[__B__,__K__,__N__]{__BATCH_RHS_LAYOUT__} parameter(1)
  ROOT dot = f16[__B__,__M__,__N__]{2,1,0} dot(lhs, rhs),
      lhs_batch_dims={0}, rhs_batch_dims={0},
      lhs_contracting_dims={2}, rhs_contracting_dims={1}
}

ENTRY main {
  lhs = f16[__B__,__M__,__K__]{2,1,0} parameter(0)
  rhs = f16[__B__,__K__,__N__]{__BATCH_RHS_LAYOUT__} parameter(1)
  ROOT fusion = f16[__B__,__M__,__N__]{2,1,0} fusion(lhs, rhs),
      kind=kCustom, calls=gemm,
      backend_config={"fusion_backend_config":{"kind":"__triton_gemm"}}
}
