XLA_MUSA_MUBLAS_1.0 {
  global:
    xla_musa_mublas_get_api_v1;
};

XLA_MUSA_MUBLAS_2.0 {
  global:
    xla_musa_mublas_get_api_v2;

  local:
    *;
} XLA_MUSA_MUBLAS_1.0;
