HloModule fly_paged_attention_decode_benchmark

ENTRY main {
  q = __TYPE__[__B__,__HQ__,128]{2,1,0} parameter(0)
  k = __TYPE__[__BLOCKS__,__PAGE__,__HKV__,128]{3,2,1,0} parameter(1)
  v = __TYPE__[__BLOCKS__,__PAGE__,__HKV__,128]{3,2,1,0} parameter(2)
  used_k_argument = s32[__B__]{0} parameter(3)
  used_k = s32[__B__]{0} constant({__USED_K_VALUES__})
  table = s32[__B__,__PAGES__]{1,0} parameter(4)
  scale = f32[] constant(0.0883883476)
  ROOT attention = __TYPE__[__B__,__HQ__,128]{2,1,0}
    custom-call(q, k, v, used_k, table, scale),
    custom_call_target="__fly$paged_attention_decode"
}
