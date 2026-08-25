HloModule fly_paged_attention_decode_preconfigured_benchmark

fly_paged_attention {
  q = __TYPE__[__B__,__HQ__,128]{2,1,0} parameter(0)
  k = __TYPE__[__BLOCKS__,__PAGE__,__HKV__,128]{3,2,1,0} parameter(1)
  v = __TYPE__[__BLOCKS__,__PAGE__,__HKV__,128]{3,2,1,0} parameter(2)
  used_k = s32[__B__]{0} parameter(3)
  table = s32[__B__,__PAGES__]{1,0} parameter(4)
  scale = f32[] constant(0.0883883476)
  ROOT attention = __TYPE__[__B__,__HQ__,128]{2,1,0}
    custom-call(q, k, v, used_k, table, scale),
    custom_call_target="__fly$paged_attention_decode"
}

ENTRY main {
  q = __TYPE__[__B__,__HQ__,128]{2,1,0} parameter(0)
  k = __TYPE__[__BLOCKS__,__PAGE__,__HKV__,128]{3,2,1,0} parameter(1)
  v = __TYPE__[__BLOCKS__,__PAGE__,__HKV__,128]{3,2,1,0} parameter(2)
  used_k = s32[__B__]{0} parameter(3)
  table = s32[__B__,__PAGES__]{1,0} parameter(4)
  ROOT fusion = __TYPE__[__B__,__HQ__,128]{2,1,0}
    fusion(q, k, v, used_k, table), kind=kCustom,
    calls=fly_paged_attention,
    backend_config={"fusion_backend_config":{"kind":"__fly","block_level_fusion_config":{"output_tiles":[{"sizes":["1","1","128"]}],"num_warps":"4","num_ctas":"1","num_stages":"1","waves_per_eu":"2"}}}
}
