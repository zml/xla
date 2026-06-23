// KV-parallel flash-decode kernel adapted from llama.cpp's kernel_flash_attn_ext_vec
// (+ _reduce, ggml-metal.metal): the inner loop (Q*K, online softmax, O+=P*V) and
// reductions are transcribed, then specialized for ZML decode (bf16 q/k/v/out read
// directly; causality from the device token_index, inline). nsg/nwg = function
// constants 422/423 (reduce 500/501).

#include <metal_stdlib>
using namespace metal;

#define PAD2(x, n) (((x) + (n) - 1) & ~((n) - 1))
#define FOR_UNROLL(x) _Pragma("clang loop unroll(full)") for (x)
#define N_SIMDWIDTH 32

typedef struct {
    int32_t  ne01; int32_t ne02; int32_t ne03;
    uint64_t nb01; uint64_t nb02; uint64_t nb03;
    int32_t  ne11; int32_t ne_12_2; int32_t ne_12_3;
    int32_t  ns10; uint64_t nb11; uint64_t nb12; uint64_t nb13;
    int32_t  ns20; uint64_t nb21; uint64_t nb22; uint64_t nb23;
    int32_t  ne31; int32_t ne32; int32_t ne33;
    uint64_t nb31; uint64_t nb32; uint64_t nb33;
    int32_t  ne1;  int32_t ne2;  int32_t ne3;
    float    scale; float max_bias; float m0; float m1;
    int32_t  n_head_log2; float logit_softcap;
} fa_vec_args;

typedef struct { int32_t nrows; } fa_vec_reduce_args;

constant int32_t FC_ns10 [[function_constant(420)]];  // K position stride (elems)
constant int32_t FC_ns20 [[function_constant(421)]];  // V position stride (elems)
constant int32_t FC_nsg [[function_constant(422)]];
constant int32_t FC_nwg [[function_constant(423)]];
constant int32_t FC_reduce_DV  [[function_constant(500)]];
constant int32_t FC_reduce_NWG [[function_constant(501)]];

typedef half4   q4_t;
typedef bfloat4 k4_t;
typedef bfloat4 v4_t;
typedef float   qk_t;
typedef float   s_t;
typedef float4  s4_t;
typedef float4  o4_t;

kernel void fa_vec(
        constant fa_vec_args & args,
        device const char * q,
        device const char * k,
        device const char * v,
        device const int  * tok,
        device const uint * layer,
        device       char * dst,
        threadgroup  half * shmem_f16 [[threadgroup(0)]],
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort  tiisg[[thread_index_in_simdgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]]) {
    constexpr short DK  = 128;
    constexpr short DV  = 128;
    constexpr short NE  = 1;
    constexpr short C   = 32;

#define NWG (FC_nwg)
#define NSG (FC_nsg)
#define NS10 (FC_ns10)
#define NS20 (FC_ns20)

    const int tok0 = tok[0];

    const short iwg = tgpig[2]%NWG;

    const ushort iq3 = tgpig[2]/NWG;
    const ushort iq2 = tgpig[1];
    const ushort iq1 = tgpig[0];

    constexpr short DK4 = DK/4;
    constexpr short DV4 = DV/4;

    constexpr short PK  = PAD2(DK, 128);
    constexpr short PK4 = PK/4;

    constexpr short PV  = PAD2(DV, 128);
    constexpr short PV4 = PV/4;

    constexpr short NW  = N_SIMDWIDTH;
    constexpr short NL  = NW/NE;
    constexpr short SH  = 4*C;

    threadgroup q4_t  * sq4 = (threadgroup q4_t  *) (shmem_f16 +                      0*PK);
    threadgroup s_t   * ss  = (threadgroup s_t   *) (shmem_f16 +   sgitg*SH       + NSG*PK);
    threadgroup s4_t  * ss4 = (threadgroup s4_t  *) (shmem_f16 +   sgitg*SH       + NSG*PK);
    threadgroup half  * sm  = (threadgroup half  *) (shmem_f16 +   sgitg*SH + 2*C + NSG*PK);
    threadgroup o4_t  * so4 = (threadgroup o4_t  *) (shmem_f16 + 2*sgitg*PV       + NSG*PK + NSG*SH);

    so4 += tiisg;

    {
        q += iq1*args.nb01 + iq2*args.nb02 + iq3*args.nb03;

        const short ikv2 = iq2/(args.ne02/args.ne_12_2);
        const short ikv3 = iq3/(args.ne03/args.ne_12_3);

        // Full-cache feed: K/V are the whole [n_layer, seqlen, n_kv, hd] cache;
        // offset to layer L (= layer[0]). Layer stride = seqlen*(n_kv*hd) elems
        // (= ne11*NS10). layer[0]==0 in the sliced fallback (single-layer operand).
        const uint64_t loff_k = (uint64_t)layer[0]*(uint64_t)args.ne11*(uint64_t)NS10*2;
        const uint64_t loff_v = (uint64_t)layer[0]*(uint64_t)args.ne11*(uint64_t)NS20*2;
        k += loff_k + ikv2*args.nb12 + ikv3*args.nb13;
        v += loff_v + ikv2*args.nb22 + ikv3*args.nb23;
    }

    // load heads from Q to shared memory (SPECIALIZED: q is bf16)
    device const bfloat4 * q4 = (device const bfloat4 *) ((device const char *) q);

    if (iq1 < args.ne01) {
        for (short i = tiisg; i < PK4; i += NW) {
            if (i < DK4) {
                sq4[i] = (q4_t) q4[i];
            } else {
                sq4[i] = (q4_t) 0.0f;
            }
        }
    }

    for (short i = 0; i < DV4/NL; ++i) {
        so4[i*NL] = (o4_t) 0.0f;
    }

    for (short i = tiisg; i < SH/4; i += NW) {
        ss4[i] = (s4_t) 0.0f;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    {
        float S = 0.0f;
        float M = -FLT_MAX/2;

        const short tx = tiisg%NL;
        const short ty = tiisg/NL;

        for (int ic0 = iwg*NSG + sgitg; ; ic0 += NWG*NSG) {
            int ic = ic0*C;
            if (ic >= args.ne11) {
                break;
            }
            // SPECIALIZED causality: per simdgroup ic increases monotonically, so
            // once a whole block is future (ic > tok0) this simdgroup is done.
            if (ic > tok0) {
                break;
            }

            // inline causal mask for the (possibly partial) block
            sm[tiisg] = (ic + tiisg <= tok0) ? (half) 0.0h : (half) (-MAXHALF);

            // Q*K^T
            {
                device      const k4_t * pk4 = (device const k4_t *) (k + ic*args.nb11);
                threadgroup const q4_t * pq4 = sq4;

                pk4 += ty*NS10/4 + tx;
                pq4 += tx;

                qk_t mqk[C/NE] = { [ 0 ... C/NE - 1] = 0.0f };

                FOR_UNROLL (short cc = 0; cc < C/NE; ++cc) {
                    FOR_UNROLL (short ii = 0; ii < DK4/NL; ++ii) {
                        mqk[cc] += dot((float4) pk4[cc*NE*NS10/4 +  ii*NL], (float4) pq4[ii*NL]);
                    }
                    mqk[cc] = simd_sum(mqk[cc]);
                }

                ss[NE*tx + ty] = fma(mqk[tx], args.scale, (qk_t) sm[NE*tx + ty]);
            }

            simdgroup_barrier(mem_flags::mem_threadgroup);

            // online softmax
            {
                const float m = M;
                const float s = ss[tiisg];

                M = simd_max(max(M, s));

                const float ms = exp(m - M);
                const float vs = exp(s - M);

                S = S*ms + simd_sum(vs);

                ss[tiisg] = vs;

                if ((DV4/NL % NW == 0) || ty == 0) {
                    FOR_UNROLL (short ii = 0; ii < DV4/NL; ++ii) {
                        so4[ii*NL] *= ms;
                    }
                }
            }

            simdgroup_barrier(mem_flags::mem_threadgroup);

            // O = O + (Q*K^T)*V
            {
                o4_t lo[DV4/NL];
                FOR_UNROLL (short ii = 0; ii < DV4/NL; ++ii) {
                    lo[ii] = 0.0f;
                }

                device const v4_t * pv4 = (device const v4_t *) (v + ic*args.nb21);

                pv4 += ty*NS20/4 + tx;

                const auto sst = ss + ty;

                FOR_UNROLL (short cc = 0; cc < C/NE; ++cc) {
                    FOR_UNROLL (short ii = 0; ii < DV4/NL; ++ii) {
                        lo[ii] += o4_t(float4(pv4[cc*NE*NS20/4 + ii*NL])*float4(sst[cc*NE]));
                    }
                }

                if ((DV4/NL % NW == 0) || ty == 0) {
                    FOR_UNROLL (short ii = 0; ii < DV4/NL; ++ii) {
                        so4[ii*NL] += lo[ii];
                    }
                }
            }
        }

        if (tiisg == 0) {
            ss[0] = (s_t) S;
            ss[1] = (s_t) M;
        }
    }

    so4 -= tiisg;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // parallel reduce across the NSG simdgroups (VERBATIM)
    for (short r = NSG/2; r > 0; r >>= 1) {
        if (sgitg < r) {
            const float S0 = ss[           0];
            const float S1 = ss[r*(SH/2) + 0];

            const float M0 = ss[           1];
            const float M1 = ss[r*(SH/2) + 1];

            const float M = max(M0, M1);

            const float ms0 = exp(M0 - M);
            const float ms1 = exp(M1 - M);

            const float S = S0*ms0 + S1*ms1;

            if (tiisg == 0) {
                ss[0] = S;
                ss[1] = M;
            }

            for (short i = tiisg; i < DV4; i += NW) {
                so4[i] = so4[i]*ms0 + so4[i + r*PV4]*ms1;
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // final store (SPECIALIZED: bf16 when nwg==1; else f32 per-workgroup partials)
    if (sgitg == 0) {
        const int64_t nrows = args.ne3*args.ne2*args.ne1;
        const int64_t rid   = iq3*args.ne2*args.ne1 + iq2 + iq1*args.ne1;

        if (NWG == 1) {
            const float S = ss[0] == 0.0f ? 0.0f : 1.0f/ss[0];
            device bfloat4 * dst4 = (device bfloat4 *) dst;
            for (short i = tiisg; i < DV4; i += NW) {
                dst4[rid*DV4 + i] = (bfloat4) ((float4) so4[i]*S);
            }
        } else {
            device float4 * dst4 = (device float4 *) dst;
            device float  * dst1 = (device float  *) dst + nrows*DV*NWG;
            for (short i = tiisg; i < DV4; i += NW) {
                dst4[rid*DV4*NWG + NWG*i + iwg] = (float4) so4[i];
            }
            if (tiisg == 0) {
                dst1[rid*(2*NWG) + 2*iwg + 0] = ss[0];
                dst1[rid*(2*NWG) + 2*iwg + 1] = ss[1];
            }
        }
    }

#undef NWG
#undef NSG
#undef NS10
#undef NS20
}

// HEAD-CONTIGUOUS split-K decode. Same verified inner loop (Q*K, online softmax,
// O+=P*V) as fa_vec, but the parallelization is transposed to defeat the strided
// KV read of OUR position-major cache [..,seqlen,n_kv,hd]:
//   * NSG = n_kv simdgroups, simdgroup sgitg == KV head sgitg.
//   * grid (1, n_groups, NWG): tgpig[1] = the query-head-within-group; the query
//     head a simdgroup owns is (sgitg*n_groups + tgpig[1]).
//   * positions are split across the NWG workgroups (ic0 strides by NWG); ALL
//     simdgroups of a threadgroup walk the SAME blocks, each reading ITS head.
// At a fixed position ic the n_kv simdgroups read k+ic*nb11 + {0,hd,2hd,..}*2 =
// the n_kv*hd contiguous run -> one sequential load, not n_kv strided ones. Each
// simdgroup is an independent, complete attention for its query head over its
// position stripe (NO cross-simdgroup reduce); it writes an f32 partial that the
// (unchanged) fa_vec_reduce combines across the NWG workgroups.
kernel void fa_vec_hc(
        constant fa_vec_args & args,
        device const char * q,
        device const char * k,
        device const char * v,
        device const int  * tok,
        device const uint * layer,
        device       char * dst,
        threadgroup  half * shmem_f16 [[threadgroup(0)]],
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort  tiisg[[thread_index_in_simdgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]]) {
    constexpr short DK  = 128;
    constexpr short DV  = 128;
    constexpr short NE  = 1;
    constexpr short C   = 32;

#define NWG (FC_nwg)
#define NSG (FC_nsg)
#define NS10 (FC_ns10)
#define NS20 (FC_ns20)

    const int tok0 = tok[0];

    const short n_grp = (short) (args.ne02 / args.ne_12_2);  // query heads / KV head

    const short  iwg = (short)  (tgpig[2] % NWG);   // position-split workgroup
    const ushort iq3 = (ushort) (tgpig[2] / NWG);   // batch
    const ushort iq2 = (ushort) (sgitg*n_grp) + (ushort) tgpig[1];  // query head
    const ushort iq1 = tgpig[0];                    // = 0 (decode, ne01 == 1)

    constexpr short DK4 = DK/4;
    constexpr short DV4 = DV/4;

    constexpr short PK  = PAD2(DK, 128);
    constexpr short PK4 = PK/4;
    constexpr short PV  = PAD2(DV, 128);
    constexpr short PV4 = PV/4;
    constexpr short NW  = N_SIMDWIDTH;
    constexpr short NL  = NW/NE;
    constexpr short SH  = 4*C;

    // Per-simdgroup regions (each simdgroup is a distinct query head).
    threadgroup q4_t  * sq4 = (threadgroup q4_t  *) (shmem_f16 +   sgitg*PK);
    threadgroup s_t   * ss  = (threadgroup s_t   *) (shmem_f16 +   sgitg*SH       + NSG*PK);
    threadgroup s4_t  * ss4 = (threadgroup s4_t  *) (shmem_f16 +   sgitg*SH       + NSG*PK);
    threadgroup half  * sm  = (threadgroup half  *) (shmem_f16 +   sgitg*SH + 2*C + NSG*PK);
    threadgroup o4_t  * so4 = (threadgroup o4_t  *) (shmem_f16 + 2*sgitg*PV       + NSG*PK + NSG*SH);

    so4 += tiisg;

    {
        q += iq1*args.nb01 + iq2*args.nb02 + iq3*args.nb03;

        const short ikv2 = sgitg;  // KV head == simdgroup index
        const short ikv3 = iq3/(args.ne03/args.ne_12_3);

        const uint64_t loff_k = (uint64_t)layer[0]*(uint64_t)args.ne11*(uint64_t)NS10*2;
        const uint64_t loff_v = (uint64_t)layer[0]*(uint64_t)args.ne11*(uint64_t)NS20*2;
        k += loff_k + ikv2*args.nb12 + ikv3*args.nb13;
        v += loff_v + ikv2*args.nb22 + ikv3*args.nb23;
    }

    // this simdgroup's query head -> its own Q slot (SPECIALIZED: q is bf16)
    device const bfloat4 * q4 = (device const bfloat4 *) ((device const char *) q);

    if (iq1 < args.ne01) {
        for (short i = tiisg; i < PK4; i += NW) {
            sq4[i] = (i < DK4) ? (q4_t) q4[i] : (q4_t) 0.0f;
        }
    }

    for (short i = 0; i < DV4/NL; ++i) {
        so4[i*NL] = (o4_t) 0.0f;
    }

    for (short i = tiisg; i < SH/4; i += NW) {
        ss4[i] = (s4_t) 0.0f;
    }

    simdgroup_barrier(mem_flags::mem_threadgroup);

    {
        float S = 0.0f;
        float M = -FLT_MAX/2;

        const short tx = tiisg%NL;
        const short ty = tiisg/NL;

        // POSITION-SPLIT: stride by NWG (not NWG*NSG) — sgitg now selects the head,
        // not a position stripe.
        for (int ic0 = iwg; ; ic0 += NWG) {
            int ic = ic0*C;
            if (ic >= args.ne11) {
                break;
            }
            if (ic > tok0) {
                break;
            }

            sm[tiisg] = (ic + tiisg <= tok0) ? (half) 0.0h : (half) (-MAXHALF);

            // Q*K^T
            {
                device      const k4_t * pk4 = (device const k4_t *) (k + ic*args.nb11);
                threadgroup const q4_t * pq4 = sq4;

                pk4 += ty*NS10/4 + tx;
                pq4 += tx;

                qk_t mqk[C/NE] = { [ 0 ... C/NE - 1] = 0.0f };

                FOR_UNROLL (short cc = 0; cc < C/NE; ++cc) {
                    FOR_UNROLL (short ii = 0; ii < DK4/NL; ++ii) {
                        mqk[cc] += dot((float4) pk4[cc*NE*NS10/4 +  ii*NL], (float4) pq4[ii*NL]);
                    }
                    mqk[cc] = simd_sum(mqk[cc]);
                }

                ss[NE*tx + ty] = fma(mqk[tx], args.scale, (qk_t) sm[NE*tx + ty]);
            }

            simdgroup_barrier(mem_flags::mem_threadgroup);

            // online softmax
            {
                const float m = M;
                const float s = ss[tiisg];

                M = simd_max(max(M, s));

                const float ms = exp(m - M);
                const float vs = exp(s - M);

                S = S*ms + simd_sum(vs);

                ss[tiisg] = vs;

                if ((DV4/NL % NW == 0) || ty == 0) {
                    FOR_UNROLL (short ii = 0; ii < DV4/NL; ++ii) {
                        so4[ii*NL] *= ms;
                    }
                }
            }

            simdgroup_barrier(mem_flags::mem_threadgroup);

            // O = O + (Q*K^T)*V
            {
                o4_t lo[DV4/NL];
                FOR_UNROLL (short ii = 0; ii < DV4/NL; ++ii) {
                    lo[ii] = 0.0f;
                }

                device const v4_t * pv4 = (device const v4_t *) (v + ic*args.nb21);

                pv4 += ty*NS20/4 + tx;

                const auto sst = ss + ty;

                FOR_UNROLL (short cc = 0; cc < C/NE; ++cc) {
                    FOR_UNROLL (short ii = 0; ii < DV4/NL; ++ii) {
                        lo[ii] += o4_t(float4(pv4[cc*NE*NS20/4 + ii*NL])*float4(sst[cc*NE]));
                    }
                }

                if ((DV4/NL % NW == 0) || ty == 0) {
                    FOR_UNROLL (short ii = 0; ii < DV4/NL; ++ii) {
                        so4[ii*NL] += lo[ii];
                    }
                }
            }
        }

        if (tiisg == 0) {
            ss[0] = (s_t) S;
            ss[1] = (s_t) M;
        }
    }

    so4 -= tiisg;

    simdgroup_barrier(mem_flags::mem_threadgroup);

    // f32 partial per (query head, workgroup) -> fa_vec_reduce combines the NWG
    // workgroups. Every simdgroup stores its OWN query head (no sgitg==0 guard).
    {
        const int64_t nrows = args.ne3*args.ne2*args.ne1;
        const int64_t rid   = iq3*args.ne2*args.ne1 + iq2 + iq1*args.ne1;

        device float4 * dst4 = (device float4 *) dst;
        device float  * dst1 = (device float  *) dst + nrows*DV*NWG;
        for (short i = tiisg; i < DV4; i += NW) {
            dst4[rid*DV4*NWG + NWG*i + iwg] = (float4) so4[i];
        }
        if (tiisg == 0) {
            dst1[rid*(2*NWG) + 2*iwg + 0] = ss[0];
            dst1[rid*(2*NWG) + 2*iwg + 1] = ss[1];
        }
    }

#undef NWG
#undef NSG
#undef NS10
#undef NS20
}

kernel void fa_vec_reduce(
        constant fa_vec_reduce_args & args,
        device  const char * htmp,
        device        char * dst,
        uint   tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
#define NWG (FC_reduce_NWG)
#define DV  (FC_reduce_DV)

    const uint64_t rid = tgpig;
    const short iwg = tiisg;

    device const float * ss = (device const float *) htmp + (uint64_t)args.nrows*DV*NWG;

    float S = ss[rid*(2*NWG) + 2*iwg + 0];
    float M = ss[rid*(2*NWG) + 2*iwg + 1];

    const float m  = simd_max(M);
    const float ms = exp(M - m);

    S = simd_sum(S*ms);
    S = S == 0.0f ? 0.0f : 1.0f/S;

    const short DV4 = DV/4;

    device const float4  * htmp4 = (device const float4  *) htmp + rid*DV4*NWG;
    device       bfloat4 * dst4  = (device       bfloat4 *) dst  + rid*DV4;

    for (short i = sgitg; i < DV4; i += NWG) {
        const float4 v = simd_sum(htmp4[i*NWG + iwg]*ms);
        if (iwg == 0) {
            dst4[i] = (bfloat4) (v*S);
        }
    }

#undef NWG
#undef DV
}
