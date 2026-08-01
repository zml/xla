// Prefill flash-attention adapted from llama.cpp's kernel_flash_attn_ext (the
// simdgroup-matrix FA-2 kernel, ggml-metal.metal), specialized for ZML prefill
// (bf16 q/k/v/out; inline 2D causality, no materialized mask; gated on
// seqlen%64==0). K/V position strides = function constants 430/431.

#include <metal_stdlib>
#include <metal_simdgroup_matrix>
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
} fa_ext_args;

constant int32_t FC_ns10 [[function_constant(430)]];  // K position stride (elems)
constant int32_t FC_ns20 [[function_constant(431)]];  // V position stride (elems)
// On-GPU prefill row clamp (U1). When true, the kernel takes the real prompt
// length as a device pointer (buffer 7) and early-returns Q-row blocks entirely
// past it, so the HOST dispatches the FULL static grid and never reads num_tokens
// host-side. That host read raced its GPU producer on Metal (there is no totally-
// ordered host/GPU path; see metal_gemm_thunk's MB_TOKCLAMP). False = no clamp arg
// and the full grid runs (legacy: prefill calls without a num_tokens operand).
constant bool FC_tokclamp [[function_constant(432)]];

kernel void fa_ext(
        constant fa_ext_args & args,
        device const char * q,
        device const char * k,
        device const char * v,
        device const int  * tok,
        device const uint * layer,
        device       char * dst,
        device const int  * num_tokens [[buffer(7), function_constant(FC_tokclamp)]],
        threadgroup  half * shmem_f16 [[threadgroup(0)]],
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort  tiisg[[thread_index_in_simdgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]]) {

    typedef bfloat              q_t;
    typedef bfloat4             q4_t;
    typedef simdgroup_bfloat8x8 q8x8_t;
    typedef bfloat              k_t;
    typedef simdgroup_bfloat8x8 k8x8_t;
    typedef bfloat              v_t;
    typedef simdgroup_bfloat8x8 v8x8_t;
    typedef float               qk_t;
    typedef simdgroup_float8x8  qk8x8_t;
    typedef float               s_t;
    typedef float2              s2_t;
    typedef simdgroup_float8x8  s8x8_t;
    typedef half                o_t;
    typedef half4               o4_t;
    typedef simdgroup_half8x8   o8x8_t;

    // Head dim is substituted at compile time (__DK__/__DV__) so one kernel
    // source serves every head size the thunk admits (e.g. LFM2's 64, Llama's
    // 128). All the tile counts below derive from DK/DV; the only places that
    // also need DV>=128 were the two DV4/NW accumulator loops, now written
    // strided so they cover any DV (DV4 < NW just means fewer active lanes).
    constexpr short DK = __DK__, DV = __DV__;
    constexpr short Q = 8, C = 64, NSG = 4;
#define NS10 (FC_ns10)
#define NS20 (FC_ns20)

    const ushort iq3 = tgpig[2];
    const ushort iq2 = tgpig[1];
    const ushort iq1 = tgpig[0]*Q;

    // U1 on-GPU row clamp: skip threadgroups whose entire Q-row block [iq1, iq1+Q)
    // is past the real prompt length. iq1 and num_tokens[0] are both uniform across
    // the threadgroup, so every thread returns together — no barrier divergence
    // (the first threadgroup_barrier is further down). nt<1 falls back to the full
    // grid; for a full prompt nt>=ne01 so nothing is skipped (byte-identical to the
    // unclamped grid). The straddling block runs and writes a few padding rows past
    // num_tokens — harmless (causal + position-wise, no consumer reads them), exactly
    // as the previous host-side grid clamp did.
    if (FC_tokclamp) {
        const int nt = num_tokens[0];
        if (nt >= 1 && iq1 >= (uint) nt) return;
    }

    const int tok0 = tok[0];  // prefill query-position base (0 for a fresh prefill)

    constexpr short DK4  = DK/4;
    constexpr short DK8  = DK/8;
    constexpr short DV4  = DV/4;
    constexpr short PV   = PAD2(DV, 64);
    constexpr short PV4  = PV/4;
    constexpr short PV8  = PV/8;
    constexpr short NW   = N_SIMDWIDTH;
    constexpr short NQ   = Q/NSG;     // = 2
    constexpr short SH   = 2*C;       // = 128 (s_t == float per query)
    constexpr short T    = DK + 2*PV; // = 384 (half units per query)

    threadgroup q_t  * sq  = (threadgroup q_t  *) (shmem_f16 + 0*T);
    threadgroup q4_t * sq4 = (threadgroup q4_t *) (shmem_f16 + 0*T);
    threadgroup o_t  * so  = (threadgroup o_t  *) (shmem_f16 + 0*T + Q*DK);
    threadgroup o4_t * so4 = (threadgroup o4_t *) (shmem_f16 + 0*T + Q*DK);
    threadgroup s_t  * ss  = (threadgroup s_t  *) (shmem_f16 + Q*T);
    threadgroup s2_t * ss2 = (threadgroup s2_t *) (shmem_f16 + Q*T);

    {
        q += iq1*args.nb01 + iq2*args.nb02 + iq3*args.nb03;

        const short ikv2 = iq2/(args.ne02/args.ne_12_2);
        const short ikv3 = iq3/(args.ne03/args.ne_12_3);

        const uint64_t loff_k = (uint64_t)layer[0]*(uint64_t)args.ne11*(uint64_t)NS10*2;
        const uint64_t loff_v = (uint64_t)layer[0]*(uint64_t)args.ne11*(uint64_t)NS20*2;
        k += loff_k + ikv2*args.nb12 + ikv3*args.nb13;
        v += loff_v + ikv2*args.nb22 + ikv3*args.nb23;
    }

    FOR_UNROLL (short jj = 0; jj < NQ; ++jj) {
        const short j = jj*NSG + sgitg;
        device const bfloat4 * q4 = (device const bfloat4 *) ((device const char *) q + j*args.nb01);
        for (short i = tiisg; i < DK4; i += NW) {
            sq4[j*DK4 + i] = (iq1 + j < args.ne01) ? (q4_t) q4[i] : (q4_t) 0;
        }
    }

    FOR_UNROLL (short jj = 0; jj < NQ; ++jj) {
        const short j = jj*NSG + sgitg;
        for (short i = tiisg; i < DV4; i += NW) so4[j*PV4 + i] = 0;
        for (short i = tiisg; i < SH;  i += NW) ss[j*SH + i] = 0.0f;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    float S[NQ] = { [0 ... NQ-1] = 0.0f };
    {
        float M[NQ] = { [0 ... NQ-1] = -FLT_MAX/2 };

        const int q_global_max = tok0 + iq1 + (Q - 1);

        for (int ic0 = 0; ; ++ic0) {
            const int ic = ic0*C;
            if (ic >= args.ne11)   break;
            if (ic > q_global_max) break;  // remaining KV blocks are causally future

            // Q*K^T  (float accum, bf16 inputs, K transposed-load from global)
            {
                device      const k_t * pk = (device const k_t *) (k + ic*args.nb11);
                threadgroup const q_t * pq = sq;
                threadgroup       s_t * ps = ss;

                pk += sgitg*(8*NS10);
                ps += sgitg*(8*1);

                constexpr short NC = (C/8)/NSG;  // = 2

                FOR_UNROLL (short cc = 0; cc < NC; ++cc) {
                    qk8x8_t mqk = make_filled_simdgroup_matrix<qk_t, 8>((qk_t) 0.0f);

                    k8x8_t mk[2];
                    q8x8_t mq[2];
                    for (short i = 0; i < DK8/2; ++i) {
                        simdgroup_barrier(mem_flags::mem_none);
                        simdgroup_load(mq[0], pq + 0*8 + 16*i, DK);
                        simdgroup_load(mq[1], pq + 1*8 + 16*i, DK);
                        simdgroup_load(mk[0], pk + 0*8 + 16*i, NS10, 0, true);
                        simdgroup_load(mk[1], pk + 1*8 + 16*i, NS10, 0, true);
                        simdgroup_barrier(mem_flags::mem_none);
                        simdgroup_multiply_accumulate(mqk, mq[0], mk[0], mqk);
                        simdgroup_multiply_accumulate(mqk, mq[1], mk[1], mqk);
                    }

                    simdgroup_store(mqk, ps, SH, 0, false);

                    pk += 8*(NSG*NS10);
                    ps += 8*(NSG);
                }
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);

            // online softmax with INLINE 2D causal mask
            FOR_UNROLL (short jj = 0; jj < NQ; ++jj) {
                const short j = jj*NSG + sgitg;

                const float m = M[jj];

                float2 s2 = ss2[j*SH/2 + tiisg]*args.scale;

                // lane tiisg holds columns (2*tiisg, 2*tiisg+1) of this C-block.
                // FINITE sentinel (not -INFINITY): fast-math leaves +-inf undefined.
                const float kNegInf = -1e30f;
                const int q_global = tok0 + iq1 + j;
                s2[0] += (ic + 2*tiisg + 0 <= q_global) ? 0.0f : kNegInf;
                s2[1] += (ic + 2*tiisg + 1 <= q_global) ? 0.0f : kNegInf;

                M[jj] = simd_max(max(M[jj], max(s2[0], s2[1])));

                const float  ms  = exp(m  - M[jj]);
                const float2 vs2 = exp(s2 - M[jj]);

                S[jj] = S[jj]*ms + simd_sum(vs2[0] + vs2[1]);

                ss2[j*SH/2 + tiisg] = vs2;

                // Rescale this query's running output by the online-softmax
                // factor. Strided over DV4 so it covers any head dim: DV4 >= NW
                // (hd>=128) loops; DV4 < NW (hd=64 -> DV4=16) just uses 16 lanes.
                for (short i = tiisg; i < DV4; i += NW) {
                    so4[j*PV4 + i] *= ms;
                }
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);

            // O += P*V  (half accum, P float / V bf16, V normal-load from global)
            {
                constexpr short NO = PV8/NSG;  // = 4
                o8x8_t lo[NO];
                {
                    auto sot = so + 8*sgitg;
                    FOR_UNROLL (short ii = 0; ii < NO; ++ii) {
                        simdgroup_load(lo[ii], sot, PV, 0, false);
                        sot += 8*NSG;
                    }
                }
                {
                    device const v_t * pv = (device const v_t *) (v + ic*args.nb21);
                    pv += 8*sgitg;

                    constexpr short NC = (C/8)/2;  // = 4
                    FOR_UNROLL (short cc = 0; cc < NC; ++cc) {
                        s8x8_t vs[2];
                        simdgroup_load(vs[0], ss + 16*cc + 0, SH, 0, false);
                        simdgroup_load(vs[1], ss + 16*cc + 8, SH, 0, false);

                        FOR_UNROLL (short ii = 0; ii < NO/2; ++ii) {  // = 2
                            v8x8_t mv[4];
                            simdgroup_load(mv[0], pv + 0*NSG + 16*ii*NSG + 0*8*NS20, NS20, 0, false);
                            simdgroup_load(mv[1], pv + 8*NSG + 16*ii*NSG + 0*8*NS20, NS20, 0, false);
                            simdgroup_load(mv[2], pv + 0*NSG + 16*ii*NSG + 1*8*NS20, NS20, 0, false);
                            simdgroup_load(mv[3], pv + 8*NSG + 16*ii*NSG + 1*8*NS20, NS20, 0, false);

                            simdgroup_multiply_accumulate(lo[2*ii + 0], vs[0], mv[0], lo[2*ii + 0]);
                            simdgroup_multiply_accumulate(lo[2*ii + 1], vs[0], mv[1], lo[2*ii + 1]);
                            simdgroup_multiply_accumulate(lo[2*ii + 0], vs[1], mv[2], lo[2*ii + 0]);
                            simdgroup_multiply_accumulate(lo[2*ii + 1], vs[1], mv[3], lo[2*ii + 1]);
                        }
                        pv += 2*8*NS20;
                    }
                }
                {
                    auto sot = so + 8*sgitg;
                    FOR_UNROLL (short ii = 0; ii < NO; ++ii) {
                        simdgroup_store(lo[ii], sot, PV, 0, false);
                        sot += 8*NSG;
                    }
                }
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    // store to global memory as bf16, row-major [NQH, q_len, hd] (head-major).
    for (short jj = 0; jj < NQ; ++jj) {
        const short j = jj*NSG + sgitg;
        if (iq1 + j >= args.ne01) break;

        device bfloat4 * dst4 = (device bfloat4 *) dst +
            ((uint64_t)iq3*args.ne2*args.ne1 + (uint64_t)iq2*args.ne1 + (iq1 + j))*DV4;

        const float scale = S[jj] == 0.0f ? 0.0f : 1.0f/S[jj];

        // Strided over DV4 so the store covers any head dim (see the rescale loop
        // above): hd=64 writes its 16 float4s on 16 lanes, hd>=128 loops.
        for (short i = tiisg; i < DV4; i += NW) {
            dst4[i] = (bfloat4) ((float4) so4[j*PV4 + i]*scale);
        }
    }

#undef NS10
#undef NS20
}
