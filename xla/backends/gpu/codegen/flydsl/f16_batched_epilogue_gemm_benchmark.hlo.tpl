HloModule flydsl_f16_batched_epilogue_gemm___B__x__M__x__N__x__K__

gemm {
  lhs = f16[__B__,__M__,__K__]{2,1,0} parameter(0)
  rhs = f16[__B__,__K__,__N__]{__BATCH_RHS_LAYOUT__} parameter(1)
  column_bias = f16[__N__]{0} parameter(2)
  batch_scale = f16[__B__]{0} parameter(3)
  row_bias = f16[__M__]{0} parameter(4)
  dot = f16[__B__,__M__,__N__]{2,1,0} dot(lhs, rhs),
      lhs_batch_dims={0}, rhs_batch_dims={0},
      lhs_contracting_dims={2}, rhs_contracting_dims={1}
  column_broadcast = f16[__B__,__M__,__N__]{2,1,0} broadcast(column_bias),
      dimensions={2}
  add_column = f16[__B__,__M__,__N__]{2,1,0} add(dot, column_broadcast)
  batch_broadcast = f16[__B__,__M__,__N__]{2,1,0} broadcast(batch_scale),
      dimensions={0}
  multiply_batch = f16[__B__,__M__,__N__]{2,1,0} multiply(add_column,
      batch_broadcast)
  row_broadcast = f16[__B__,__M__,__N__]{2,1,0} broadcast(row_bias),
      dimensions={1}
  ROOT add_row = f16[__B__,__M__,__N__]{2,1,0} add(multiply_batch,
      row_broadcast)
}

ENTRY main {
  lhs = f16[__B__,__M__,__K__]{2,1,0} parameter(0)
  rhs = f16[__B__,__K__,__N__]{__BATCH_RHS_LAYOUT__} parameter(1)
  column_bias = f16[__N__]{0} parameter(2)
  batch_scale = f16[__B__]{0} parameter(3)
  row_bias = f16[__M__]{0} parameter(4)
  ROOT fusion = f16[__B__,__M__,__N__]{2,1,0} fusion(lhs, rhs, column_bias,
      batch_scale, row_bias), kind=kCustom, calls=gemm,
      backend_config={"fusion_backend_config":{"kind":"__triton_gemm"}}
}
