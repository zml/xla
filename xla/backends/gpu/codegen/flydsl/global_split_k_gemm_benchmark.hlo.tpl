HloModule flydsl_bf16_global_split_k_gemm___B__x__M__x__N__x__K__

gemm {
  lhs = bf16[__M__,__B__,__K__]{2,1,0} parameter(0)
  rhs = bf16[__B__,__K__,__N__]{2,1,0} parameter(1)
  ROOT dot = f32[__B__,__M__,__N__]{2,1,0} dot(lhs, rhs),
      lhs_batch_dims={1}, rhs_batch_dims={0},
      lhs_contracting_dims={2}, rhs_contracting_dims={1}
}

ENTRY main {
  lhs = bf16[__M__,__B__,__K__]{2,1,0} parameter(0)
  rhs = bf16[__B__,__K__,__N__]{2,1,0} parameter(1)
  ROOT fusion = f32[__B__,__M__,__N__]{2,1,0} fusion(lhs, rhs),
      kind=kCustom, calls=gemm,
      backend_config={"fusion_backend_config":{"kind":"__triton_gemm"}}
}
