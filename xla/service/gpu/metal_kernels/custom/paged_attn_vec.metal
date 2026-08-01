// Paged-attention vector decode adapted from llama.cpp's kernel_flash_attn_ext_vec
// (the verified contiguous fa_vec) with paged KV addressing (block_tables / in-page
// offset) and per-row varlen via cu_seqlens_q. bf16, DK=DV=128; nsg/block_size/
// pos_stride = function constants 450/451/452.

#include <metal_stdlib>
using namespace metal;

#define PAD2(x, n) (((x) + (n) - 1) & ~((n) - 1))
#define FOR_UNROLL(x) _Pragma("clang loop unroll(full)") for (x)
#define N_SIMDWIDTH 32

typedef struct {
    int32_t num_heads;   // total query heads
    int32_t n_kv;        // KV heads (GQA: kv head = q head / (num_heads/n_kv))
    int32_t max_blocks;  // block-table row stride (entries per sequence)
    int32_t num_seqs;    // cu_seqlens_q has num_seqs+1 entries
    float   scale;
    int32_t sliding_window;  // sliding-window size; < 0 = global (full causal)
} fa_vec_paged_args;

constant int32_t FC_nsg [[function_constant(450)]];  // simdgroups (KV split)
constant int32_t FC_bs  [[function_constant(451)]];  // page block_size (8/16/32)
constant int32_t FC_ps  [[function_constant(452)]];  // in-page position stride, elems (= n_kv*hd)
constant int32_t FC_nwg [[function_constant(453)]];  // position-split workgroups (1 = single-pass; >1 writes f32 partials for fa_vec_paged_reduce)

typedef half4   q4_t;
typedef bfloat4 k4_t;
typedef bfloat4 v4_t;
typedef float   qk_t;
typedef float   s_t;
typedef float4  s4_t;
typedef float4  o4_t;

// Templated on the head dims so the same verified body specializes at
// DK=DV=128 (Llama-shaped) and the large Gemma dims 256/512. Everything scales
// off DK/DV (DK4=DK/4, PK=PAD2(DK,128), the o4_t lo[DV4/NL] register tile, and
// the shmem layout) — only these two were constants. Kernel entry points below.
template <short DK, short DV>
static void fa_vec_paged_impl(
        constant fa_vec_paged_args & args,
        device const char * q,
        device const char * k_cache,
        device const char * v_cache,
        device const int  * block_tables,
        device const int  * seq_lens,
        device const int  * cu_seqlens_q,
        device       char * dst,
        threadgroup  half * shmem_f16,
        uint3   tgpig,
        uint3   tgpg,
        ushort  tiisg,
        ushort  sgitg) {
    constexpr short NE  = 1;
    constexpr short C   = 32;

#define NSG (FC_nsg)
#define BS  (FC_bs)
#define PS  (FC_ps)
#define NWG (FC_nwg)

    const int r   = (int) tgpig[2];  // query ROW in [0, total_q_tokens)
    const ushort iq2 = tgpig[1];     // query head
    const short iwg = (short) tgpig[0];  // position-split workgroup (KV partition)

    constexpr short DK4 = DK/4;
    constexpr short DV4 = DV/4;

    // STATIC row count (= total_q_tokens * num_heads). The split-K partial /
    // reduce must agree on where the S/M region begins; the reduce uses the
    // host's static args.nrows, so the partial pass must too. cu_seqlens_q[
    // num_seqs] is the DYNAMIC active-token count (< total_q_tokens whenever the
    // batch is partly padded), which would place the S/M region at the wrong
    // offset -> garbage. tgpg[2] is the grid's z extent = total_q_tokens.
    const int64_t nrows_static = (int64_t)tgpg[2]*(uint)args.num_heads;

    // PADDING row (beyond the real tokens of this step): write the softmax
    // identity, done. NWG==1 writes bf16 zeros to the output; NWG>1 (split-K)
    // writes THIS workgroup's f32 partial as O=0, S=0, M=-inf so the reduce
    // yields 0. (A bf16 zero-write into the f32 partial scratch would alias a
    // real row's partials and corrupt them.)
    if (r >= cu_seqlens_q[args.num_seqs]) {
        if (sgitg == 0) {
            const int64_t rid = (int64_t)r*(uint)args.num_heads + iq2;
            if (NWG == 1) {
                device bfloat4 * dst4 = (device bfloat4 *) dst;
                for (short i = tiisg; i < DV4; i += N_SIMDWIDTH) {
                    dst4[rid*DV4 + i] = (bfloat4) float4(0.0f);
                }
            } else {
                device float4 * dst4 = (device float4 *) dst;
                device float  * dst1 = (device float  *) dst + nrows_static*DV*NWG;
                for (short i = tiisg; i < DV4; i += N_SIMDWIDTH) {
                    dst4[rid*DV4*NWG + NWG*i + iwg] = (float4) 0.0f;
                }
                if (tiisg == 0) {
                    dst1[rid*(2*NWG) + 2*iwg + 0] = 0.0f;
                    dst1[rid*(2*NWG) + 2*iwg + 1] = -FLT_MAX/2;
                }
            }
        }
        return;
    }

    // VARLEN: binary search for the LAST s with cu[s] <= r (upper-mid form,
    // same as the tiled kernel; empty chunks cu[s]==cu[s+1] never claim rows).
    int seq_idx;
    {
        int lo = 0, hi = args.num_seqs - 1;
        while (lo < hi) {
            const int mid = (lo + hi + 1) / 2;
            if (cu_seqlens_q[mid] <= r) lo = mid; else hi = mid - 1;
        }
        seq_idx = lo;
    }
    const int q_pos      = r - cu_seqlens_q[seq_idx];          // pos in chunk
    const int query_len  = cu_seqlens_q[seq_idx + 1] - cu_seqlens_q[seq_idx];
    const int kv_total   = seq_lens[seq_idx];                  // total KV of seq
    // Causal: this row sees its prefix-context plus the chunk rows up to itself
    // (context_len = kv_total - query_len, same semantics as the tiled kernel).
    const int kv_len     = kv_total - query_len + q_pos + 1;
    // Sliding window: this row attends only [kv_lo, kv_len). Matches the tiled
    // kernel's `kv_pos < q_abs_pos+1-window` mask (q_abs_pos = kv_len-1 here).
    // Global layers pass sliding_window < 0 -> kv_lo = 0 (full causal prefix).
    const int kv_lo      = (args.sliding_window < 0)
                               ? 0
                               : max(0, kv_len - args.sliding_window);

    device const int * pt = block_tables + (uint64_t)seq_idx*(uint)args.max_blocks;

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

    // Q row for (row, head): q is row-major [total_q_tokens, num_heads, hd] bf16.
    // K/V head offset is IN-page (page layout [BS, n_kv, hd]).
    {
        q += ((uint64_t)r*(uint)args.num_heads + iq2)*DK*2;

        const short ikv2 = iq2/(args.num_heads/args.n_kv);
        k_cache += (uint64_t)ikv2*DK*2;
        v_cache += (uint64_t)ikv2*DK*2;
    }

    // load the head's Q to shared memory (q is bf16)
    device const bfloat4 * q4 = (device const bfloat4 *) ((device const char *) q);

    for (short i = tiisg; i < PK4; i += NW) {
        sq4[i] = (i < DK4) ? (q4_t) q4[i] : (q4_t) 0.0f;
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

        // Start at the first KV tile that overlaps the window [kv_lo, kv_len);
        // tiles fully below kv_lo are skipped (the sliding-window win at long
        // context). The in-window tiles are split across NWG workgroups * NSG
        // simdgroups (split-K / flash-decoding): workgroup iwg's simdgroup sgitg
        // takes ic0 = ic0_lo + iwg*NSG + sgitg, +NWG*NSG, ... (NWG=1 => the plain
        // single-pass stride by NSG). This lifts hd512's nsg<=8 occupancy cap at
        // long context.
        const int ic0_lo = kv_lo / C;
        for (int ic0 = ic0_lo + iwg*NSG + sgitg; ; ic0 += NWG*NSG) {
            const int ic = ic0*C;
            if (ic >= kv_len) {
                break;
            }

            // mask positions outside the window [kv_lo, kv_len) (causal upper +
            // sliding lower bound)
            sm[tiisg] = (ic + tiisg < kv_len && ic + tiisg >= kv_lo)
                            ? (half) 0.0h : (half) (-MAXHALF);

            // Q*K^T  (PAGED: per-position pointer through the block table; the
            // tail clamp keeps masked lanes on a valid page)
            {
                threadgroup const q4_t * pq4 = sq4;
                pq4 += tx;

                qk_t mqk[C/NE] = { [ 0 ... C/NE - 1] = 0.0f };

                FOR_UNROLL (short cc = 0; cc < C/NE; ++cc) {
                    const int p = min(ic + cc, kv_len - 1);
                    device const k4_t * pk4 = (device const k4_t *)
                        (k_cache + ((uint64_t)(uint)pt[p/BS]*(uint)(BS*PS) + (uint)(p%BS)*(uint)PS)*2);
                    pk4 += ty*PS/4 + tx;
                    FOR_UNROLL (short ii = 0; ii < DK4/NL; ++ii) {
                        mqk[cc] += dot((float4) pk4[ii*NL], (float4) pq4[ii*NL]);
                    }
                    mqk[cc] = simd_sum(mqk[cc]);
                }

                ss[NE*tx + ty] = fma(mqk[tx], args.scale, (qk_t) sm[NE*tx + ty]);
            }

            simdgroup_barrier(mem_flags::mem_threadgroup);

            // online softmax (VERBATIM)
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

            // O = O + (Q*K^T)*V  (PAGED addressing, same clamp)
            {
                o4_t lo[DV4/NL];
                FOR_UNROLL (short ii = 0; ii < DV4/NL; ++ii) {
                    lo[ii] = 0.0f;
                }

                const auto sst = ss + ty;

                FOR_UNROLL (short cc = 0; cc < C/NE; ++cc) {
                    const int p = min(ic + cc, kv_len - 1);
                    device const v4_t * pv4 = (device const v4_t *)
                        (v_cache + ((uint64_t)(uint)pt[p/BS]*(uint)(BS*PS) + (uint)(p%BS)*(uint)PS)*2);
                    pv4 += ty*PS/4 + tx;
                    FOR_UNROLL (short ii = 0; ii < DV4/NL; ++ii) {
                        lo[ii] += o4_t(float4(pv4[ii*NL])*float4(sst[cc*NE]));
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

    // final store: NWG==1 writes the final bf16 output; NWG>1 writes THIS
    // workgroup's f32 partial (unnormalized O, plus S and M) for
    // fa_vec_paged_reduce to combine across the NWG position-split workgroups.
    // (The cross-simdgroup reduce above already combined this workgroup's NSG
    // simdgroups, so each workgroup contributes one partial.)
    if (sgitg == 0) {
        const int64_t rid = (int64_t)r*(uint)args.num_heads + iq2;
        if (NWG == 1) {
            const float S = ss[0] == 0.0f ? 0.0f : 1.0f/ss[0];
            device bfloat4 * dst4 = (device bfloat4 *) dst;
            for (short i = tiisg; i < DV4; i += NW) {
                dst4[rid*DV4 + i] = (bfloat4) ((float4) so4[i]*S);
            }
        } else {
            // STATIC nrows (must match the reduce's args.nrows); see above.
            device float4 * dst4 = (device float4 *) dst;
            device float  * dst1 = (device float  *) dst + nrows_static*DV*NWG;
            for (short i = tiisg; i < DV4; i += NW) {
                dst4[rid*DV4*NWG + NWG*i + iwg] = (float4) so4[i];
            }
            if (tiisg == 0) {
                dst1[rid*(2*NWG) + 2*iwg + 0] = ss[0];
                dst1[rid*(2*NWG) + 2*iwg + 1] = ss[1];
            }
        }
    }

#undef NSG
#undef BS
#undef PS
#undef NWG
}

// Kernel entry points (same implicit buffer order as the original single kernel,
// so the thunk binds args=0, q=1, k=2, v=3, block_tables=4, seq_lens=5,
// cu_seqlens_q=6, dst=7, shmem=threadgroup(0)). Pick by head_dim in the thunk.
#define FA_VEC_PAGED_KERNEL(NAME, DK_, DV_)                                      \
kernel void NAME(                                                               \
        constant fa_vec_paged_args & args,                                     \
        device const char * q,                                                 \
        device const char * k_cache,                                           \
        device const char * v_cache,                                           \
        device const int  * block_tables,                                      \
        device const int  * seq_lens,                                          \
        device const int  * cu_seqlens_q,                                      \
        device       char * dst,                                               \
        threadgroup  half * shmem_f16 [[threadgroup(0)]],                      \
        uint3   tgpig[[threadgroup_position_in_grid]],                         \
        uint3   tgpg[[threadgroups_per_grid]],                                 \
        ushort  tiisg[[thread_index_in_simdgroup]],                            \
        ushort  sgitg[[simdgroup_index_in_threadgroup]]) {                     \
    fa_vec_paged_impl<DK_, DV_>(args, q, k_cache, v_cache, block_tables,        \
                                seq_lens, cu_seqlens_q, dst, shmem_f16,         \
                                tgpig, tgpg, tiisg, sgitg);                     \
}
FA_VEC_PAGED_KERNEL(fa_vec_paged,       128, 128)  // backward-compatible name
FA_VEC_PAGED_KERNEL(fa_vec_paged_hd256, 256, 256)  // Gemma sliding layers
FA_VEC_PAGED_KERNEL(fa_vec_paged_hd512, 512, 512)  // Gemma global layers

// Split-K reduce: combine the NWG f32 partials (O, S, M) the split-K partial
// pass wrote per (query row, head) into the final bf16 output. One threadgroup
// per row*head; the NWG partials sit in the simdgroup lanes (NWG <= 32, lanes
// >= NWG masked to identity), then an online-softmax rescale + simd_sum.
typedef struct { int32_t nrows; } fa_vec_paged_reduce_args;
constant int32_t FC_red_dv [[function_constant(454)]];  // DV (head_dim)

kernel void fa_vec_paged_reduce(
        constant fa_vec_paged_reduce_args & args,
        device const char * htmp,
        device       char * dst,
        uint   tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    const int nwg = FC_nwg;
    const int DV  = FC_red_dv;
    const uint64_t rid = tgpig;
    const short iwg = tiisg;  // partial index; only [0, nwg) carry real data
    device const float * ss = (device const float *) htmp + (uint64_t)args.nrows*DV*nwg;

    float M = (iwg < nwg) ? ss[rid*(2*nwg) + 2*iwg + 1] : -FLT_MAX/2;
    float S = (iwg < nwg) ? ss[rid*(2*nwg) + 2*iwg + 0] : 0.0f;

    const float m  = simd_max(M);
    const float ms = exp(M - m);
    S = simd_sum(S*ms);
    S = S == 0.0f ? 0.0f : 1.0f/S;

    const int DV4 = DV/4;
    device const float4  * htmp4 = (device const float4  *) htmp + rid*DV4*nwg;
    device       bfloat4 * dst4  = (device       bfloat4 *) dst  + rid*DV4;
    for (int i = sgitg; i < DV4; i += nwg) {
        const float4 part = (iwg < nwg) ? (htmp4[i*nwg + iwg]*ms) : float4(0.0f);
        const float4 v = simd_sum(part);
        if (iwg == 0) {
            dst4[i] = (bfloat4) (v*S);
        }
    }
}
