HloModule fly_paged_attention_segmented_decode_preconfigured_benchmark

fly_paged_attention_segmented_producer {
  q = __TYPE__[__B__,__HQ__,128]{2,1,0} parameter(0)
  k = __TYPE__[__BLOCKS__,__PAGE__,__HKV__,128]{3,2,1,0} parameter(1)
  v = __TYPE__[__BLOCKS__,__PAGE__,__HKV__,128]{3,2,1,0} parameter(2)
  used_k = s32[__B__]{0} parameter(3)
  table = s32[__B__,__PAGES__]{1,0} parameter(4)
  scale = f32[] constant(0.0883883476)
  ROOT partials = (f32[__B__,__HQ__,__SEGMENTS__,128]{3,2,1,0}, f32[__B__,__HQ__,__SEGMENTS__]{2,1,0}, f32[__B__,__HQ__,__SEGMENTS__]{2,1,0})
    custom-call(q, k, v, used_k, table, scale),
    custom_call_target="__fly$paged_attention_decode_segmented_producer"
}

fly_paged_attention_segmented_reducer {
  partial_o = f32[__B__,__HQ__,__SEGMENTS__,128]{3,2,1,0} parameter(0)
  partial_m = f32[__B__,__HQ__,__SEGMENTS__]{2,1,0} parameter(1)
  partial_l = f32[__B__,__HQ__,__SEGMENTS__]{2,1,0} parameter(2)
  ROOT output = __TYPE__[__B__,__HQ__,128]{2,1,0}
    custom-call(partial_o, partial_m, partial_l),
    custom_call_target="__fly$paged_attention_decode_segmented_reducer"
}

ENTRY main {
  q = __TYPE__[__B__,__HQ__,128]{2,1,0} parameter(0)
  k = __TYPE__[__BLOCKS__,__PAGE__,__HKV__,128]{3,2,1,0} parameter(1)
  v = __TYPE__[__BLOCKS__,__PAGE__,__HKV__,128]{3,2,1,0} parameter(2)
  used_k = s32[__B__]{0} parameter(3)
  table = s32[__B__,__PAGES__]{1,0} parameter(4)
  producer = (f32[__B__,__HQ__,__SEGMENTS__,128]{3,2,1,0}, f32[__B__,__HQ__,__SEGMENTS__]{2,1,0}, f32[__B__,__HQ__,__SEGMENTS__]{2,1,0})
    fusion(q, k, v, used_k, table), kind=kCustom,
    calls=fly_paged_attention_segmented_producer,
    backend_config={"fusion_backend_config":{"kind":"__fly","block_level_fusion_config":{"output_tiles":[{"sizes":["1","1","1","128"]},{"sizes":["1","1","1"]},{"sizes":["1","1","1"]}],"num_warps":"4","num_ctas":"1","num_stages":"1","waves_per_eu":"2"}}}
  partial_o = f32[__B__,__HQ__,__SEGMENTS__,128]{3,2,1,0} get-tuple-element(producer), index=0
  partial_m = f32[__B__,__HQ__,__SEGMENTS__]{2,1,0} get-tuple-element(producer), index=1
  partial_l = f32[__B__,__HQ__,__SEGMENTS__]{2,1,0} get-tuple-element(producer), index=2
  ROOT reducer = __TYPE__[__B__,__HQ__,128]{2,1,0}
    fusion(partial_o, partial_m, partial_l), kind=kCustom,
    calls=fly_paged_attention_segmented_reducer,
    backend_config={"fusion_backend_config":{"kind":"__fly","block_level_fusion_config":{"output_tiles":[{"sizes":["1","1","128"]}],"num_warps":"2","num_ctas":"1","num_stages":"1","waves_per_eu":"2"}}}
}
