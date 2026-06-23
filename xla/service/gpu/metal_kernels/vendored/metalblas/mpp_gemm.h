// mpp_gemm.h - Manual threadgroup-tiled GEMM over matmul2d 16x32x16 cooperative tensors.
#ifdef MB_BUILD_MPP_GEMM
#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
#include <metal_cooperative_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;
using namespace mpp::tensor_ops;

#define IN_T        __IN_T__
#define ACC_T       __ACC_T__
#define OUT_T       __OUT_T__
#define BM          __BM__
#define BN          __BN__
#define BK          __BK__
#define WM          __WM__
#define WN          __WN__
#define TRANS_A     __TRANS_A__
#define TRANS_B     __TRANS_B__
#define MN_ALIGNED  __MN_ALIGNED__
#define K_ALIGNED   __K_ALIGNED__
#define SWIZZLE_LOG __SWIZZLE_LOG__
#define RELAXED     __RELAXED__
#define DBUF        __DBUF__

#define SG_SIZE     32
#define WARPS       (WM * WN)
#define TGP_SIZE    (WARPS * SG_SIZE)
#define NBUF        (DBUF ? 2 : 1)

constant constexpr int WT_M = BM / WM;
constant constexpr int WT_N = BN / WN;
constant constexpr int FM   = 16;
constant constexpr int FN   = 32;
constant constexpr int FK   = 16;
constant constexpr int TM   = WT_M / FM;
constant constexpr int TN   = WT_N / FN;

// Per-thread element counts per fragment (16x16=8, 16x32=16).
constant constexpr int A_ELEM_PER_THR = (FM * FK) / 32;     // 8
constant constexpr int B_ELEM_PER_THR = (FK * FN) / 32;     // 16
constant constexpr int C_ELEM_PER_THR = (FM * FN) / 32;     // 16

// PAD avoids threadgroup bank conflicts and keeps VecF loads 16-byte aligned.
// __PAD__ comes from the Python wrapper; default is 16/sizeof(IN_T).
constant constexpr int PAD_A = __PAD__;
constant constexpr int PAD_B = __PAD__;
constant constexpr int LDA_TGP = BK + PAD_A;
constant constexpr int LDB_TGP = BN + PAD_B;

constant constexpr int VEC = 16 / sizeof(IN_T);
constant constexpr int A_TCOLS = BK / VEC;
constant constexpr int A_ROW_STEP = TGP_SIZE / A_TCOLS;
constant constexpr int B_TCOLS = BN / VEC;
constant constexpr int B_ROW_STEP = TGP_SIZE / B_TCOLS;

struct alignas(16) VecF { IN_T v[VEC]; };

static_assert(BM % (FM * WM) == 0, "BM must be multiple of 16*WM");
static_assert(BN % (FN * WN) == 0, "BN must be multiple of 32*WN");
static_assert(BK % FK == 0,         "BK must be multiple of 16");
static_assert(BK % VEC == 0,        "BK must be divisible by VEC");
static_assert(BN % VEC == 0,        "BN must be divisible by VEC");

static inline void load_A_tile(threadgroup IN_T   *As,
                               const device IN_T *A,
                               int lda, int M, int K,
                               int a_row0, int a_col0,
                               int tid, int kbound) {
    int local_row0 = tid / A_TCOLS;
    int local_col0 = (tid % A_TCOLS) * VEC;
    #pragma unroll
    for (int r = 0; r < BM; r += A_ROW_STEP) {
        int rl = local_row0 + r; if (rl >= BM) break;
        VecF acc;
        #pragma unroll
        for (int i=0;i<VEC;++i) acc.v[i] = (IN_T)0;
#if TRANS_A
        // A stored K x M (lda = stride along K): element (m, k) is A[k*lda + m].
        int gm = a_row0 + rl;
        bool m_ok = gm < M;
        #pragma unroll
        for (int i = 0; i < VEC; ++i) {
            int gk = a_col0 + local_col0 + i;
            bool k_ok = gk < kbound;
            acc.v[i] = (m_ok && k_ok) ? A[gk * lda + gm] : (IN_T)0;
        }
#else
        bool m_ok = (a_row0 + rl) < M;
        int gc_k0 = a_col0 + local_col0;
        bool k_full = (gc_k0 + VEC) <= kbound;
        // VecF load needs lda VEC-aligned; else unaligned load corrupts results.
        if (m_ok && k_full && (lda % VEC) == 0) {
            acc = *((const device VecF*)(&A[(a_row0 + rl) * lda + gc_k0]));
        } else {
            #pragma unroll
            for (int i = 0; i < VEC; ++i) {
                int gk = gc_k0 + i; bool ok = m_ok && (gk < kbound);
                acc.v[i] = ok ? A[(a_row0 + rl) * lda + gk] : (IN_T)0;
            }
        }
#endif
        *((threadgroup VecF*)(&As[rl * LDA_TGP + local_col0])) = acc;
    }
}

static inline void load_B_tile(threadgroup IN_T *Bs,
                               const device IN_T *B,
                               int ldb, int N, int K,
                               int b_row0, int b_col0,
                               int tid, int kbound) {
    int local_row0 = tid / B_TCOLS;
    int local_col0 = (tid % B_TCOLS) * VEC;
    int n_global   = b_col0 + local_col0;
    #pragma unroll
    for (int r = 0; r < BK; r += B_ROW_STEP) {
        int rl = local_row0 + r; if (rl >= BK) break;
        VecF acc;
        #pragma unroll
        for (int i=0;i<VEC;++i) acc.v[i] = (IN_T)0;
        int gk = b_row0 + rl;
#if TRANS_B
        bool k_ok = gk < kbound;
        if (k_ok) {
            #pragma unroll
            for (int i = 0; i < VEC; ++i) {
                int gn = n_global + i; bool ok = gn < N;
                acc.v[i] = ok ? B[gn * ldb + gk] : (IN_T)0;
            }
        }
#else
        bool k_ok = gk < kbound;
        bool n_full = (n_global + VEC) <= N;
        // VecF load needs ldb VEC-aligned; else unaligned load corrupts results.
        if (k_ok && n_full && (ldb % VEC) == 0) {
            acc = *((const device VecF*)(&B[gk * ldb + n_global]));
        } else if (k_ok) {
            #pragma unroll
            for (int i = 0; i < VEC; ++i) {
                int gn = n_global + i; bool ok = gn < N;
                acc.v[i] = ok ? B[gk * ldb + gn] : (IN_T)0;
            }
        }
#endif
        *((threadgroup VecF*)(&Bs[rl * LDB_TGP + local_col0])) = acc;
    }
}

// Dims + leading strides packed into one constant buffer (one setBytes, not six).
struct MBGemmDims { int M, N, K, lda, ldb, ldc; };

kernel void mpp_gemm(
    device const IN_T   *A           [[buffer(0)]],
    device const IN_T   *B           [[buffer(1)]],
    device       OUT_T  *C           [[buffer(2)]],
    constant MBGemmDims& gP          [[buffer(3)]],   // packed (M, N, K, lda, ldb, ldc)
#if EPILOGUE
    device const OUT_T *bias         [[buffer(4)]],   // addmm input; bstride = (row, col) broadcast strides
    constant int2&  bstride          [[buffer(5)]],
    constant ACC_T& beta             [[buffer(6)]],
    constant ACC_T& alpha            [[buffer(7)]],
#endif
    uint3        tgid                [[threadgroup_position_in_grid]],
    uint         sgid                [[simdgroup_index_in_threadgroup]],
    uint         lane                [[thread_index_in_simdgroup]])
{
    int gM = gP.M, gN = gP.N, gK = gP.K;
    int gLda = gP.lda, gLdb = gP.ldb, gLdc = gP.ldc;
    // Double-buffered (NBUF=2) when DBUF==1, single buffer otherwise.
    threadgroup IN_T As[NBUF * BM * LDA_TGP];
    threadgroup IN_T Bs[NBUF * BK * LDB_TGP];

    int tid = int(sgid) * SG_SIZE + int(lane);

    int tiles_m = (gM + BM - 1) / BM;
    int tiles_n = (gN + BN - 1) / BN;
    int sw_mask = (1 << SWIZZLE_LOG) - 1;
    int tgy = (int(tgid.y) << SWIZZLE_LOG) | (int(tgid.x) & sw_mask);
    int tgx = int(tgid.x) >> SWIZZLE_LOG;
    if (tgx >= tiles_n || tgy >= tiles_m) return;

    int m_block = tgy * BM;
    int n_block = tgx * BN;

    int warp_row = int(sgid) / WN;
    int warp_col = int(sgid) % WN;
    int warp_m   = warp_row * WT_M;
    int warp_n   = warp_col * WT_N;

    constexpr auto desc = matmul2d_descriptor(
        FM, FN, FK, false, false, RELAXED,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroup> op;

    // Pre-compute per-thread multidim index -> (row, col) maps once.
    // get_multidimensional_index() returns [fastest, slowest], hence [k,m]/[n,k]/[n,m].
    short a_off[A_ELEM_PER_THR];   // idx_m * LDA_TGP + idx_k (within frag)
    short b_off[B_ELEM_PER_THR];
    short c_om[C_ELEM_PER_THR];    // row offset of each ct_c element
    short c_on[C_ELEM_PER_THR];    // col offset
    {
        auto ct_a_proto = op.get_left_input_cooperative_tensor<IN_T, IN_T, ACC_T>();
        int e = 0;
        for (auto it = ct_a_proto.begin(); it != ct_a_proto.end(); ++it, ++e) {
            auto idx = it.get_multidimensional_index();
            a_off[e] = short(idx[1]) * short(LDA_TGP) + short(idx[0]);
        }
        auto ct_b_proto = op.get_right_input_cooperative_tensor<IN_T, IN_T, ACC_T>();
        e = 0;
        for (auto it = ct_b_proto.begin(); it != ct_b_proto.end(); ++it, ++e) {
            auto idx = it.get_multidimensional_index();
            b_off[e] = short(idx[1]) * short(LDB_TGP) + short(idx[0]);
        }
        auto ct_c_proto = op.get_destination_cooperative_tensor<decltype(ct_a_proto), decltype(ct_b_proto), ACC_T>();
        e = 0;
        for (auto it = ct_c_proto.begin(); it != ct_c_proto.end(); ++it, ++e) {
            auto idx = it.get_multidimensional_index();
            c_om[e] = short(idx[1]);
            c_on[e] = short(idx[0]);
        }
    }

    // Per-thread accumulator storage.
    ACC_T Cacc[TM * TN * C_ELEM_PER_THR];
    #pragma unroll
    for (int i = 0; i < TM * TN * C_ELEM_PER_THR; ++i) Cacc[i] = (ACC_T)0;

    // Input fragment storage for the inner K-iter, reused across i, j.
    IN_T Astg[TM * A_ELEM_PER_THR];
    IN_T Bstg[TN * B_ELEM_PER_THR];

    int k_tiles_full = gK / BK;
    int k_tail       = gK - k_tiles_full * BK;

#if DBUF
    // Double-buffered K-loop: issue the next K-tile's loads while the current MMA runs.
    if (k_tiles_full > 0) {
        load_A_tile(As, A, gLda, gM, gK, m_block, 0, tid, BK);
        load_B_tile(Bs, B, gLdb, gN, gK, 0, n_block, tid, BK);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int kt = 0; kt < k_tiles_full; ++kt) {
        int cur_a = (kt & 1) * (BM * LDA_TGP);
        int cur_b = (kt & 1) * (BK * LDB_TGP);
        if (kt + 1 < k_tiles_full) {
            int nxt_a = ((kt + 1) & 1) * (BM * LDA_TGP);
            int nxt_b = ((kt + 1) & 1) * (BK * LDB_TGP);
            load_A_tile(As + nxt_a, A, gLda, gM, gK, m_block, (kt + 1) * BK, tid, (kt + 2) * BK);
            load_B_tile(Bs + nxt_b, B, gLdb, gN, gK, (kt + 1) * BK, n_block, tid, (kt + 2) * BK);
        }
        threadgroup IN_T *As_use = As + cur_a;
        threadgroup IN_T *Bs_use = Bs + cur_b;

        #pragma unroll
        for (int kk = 0; kk < BK; kk += FK) {
            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                int base_row = warp_m + i * FM;
                threadgroup IN_T *src = &As_use[base_row * LDA_TGP + kk];
                #pragma unroll
                for (int e = 0; e < A_ELEM_PER_THR; ++e) {
                    Astg[i * A_ELEM_PER_THR + e] = src[a_off[e]];
                }
            }
            #pragma unroll
            for (int j = 0; j < TN; ++j) {
                int base_col = warp_n + j * FN;
                threadgroup IN_T *src = &Bs_use[kk * LDB_TGP + base_col];
                #pragma unroll
                for (int e = 0; e < B_ELEM_PER_THR; ++e) {
                    Bstg[j * B_ELEM_PER_THR + e] = src[b_off[e]];
                }
            }
            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                auto ct_a = op.get_left_input_cooperative_tensor<IN_T, IN_T, ACC_T>();
                #pragma unroll
                for (int e = 0; e < A_ELEM_PER_THR; ++e) {
                    ct_a[e] = Astg[i * A_ELEM_PER_THR + e];
                }
                #pragma unroll
                for (int j = 0; j < TN; ++j) {
                    auto ct_b = op.get_right_input_cooperative_tensor<IN_T, IN_T, ACC_T>();
                    #pragma unroll
                    for (int e = 0; e < B_ELEM_PER_THR; ++e) {
                        ct_b[e] = Bstg[j * B_ELEM_PER_THR + e];
                    }
                    auto ct_c = op.get_destination_cooperative_tensor<decltype(ct_a), decltype(ct_b), ACC_T>();
                    int frag_off = (i * TN + j) * C_ELEM_PER_THR;
                    #pragma unroll
                    for (int e = 0; e < C_ELEM_PER_THR; ++e) ct_c[e] = Cacc[frag_off + e];
                    op.run(ct_a, ct_b, ct_c);
                    #pragma unroll
                    for (int e = 0; e < C_ELEM_PER_THR; ++e) Cacc[frag_off + e] = ct_c[e];
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
#else
    for (int kt = 0; kt < k_tiles_full; ++kt) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        load_A_tile(As, A, gLda, gM, gK, m_block, kt * BK, tid, (kt + 1) * BK);
        load_B_tile(Bs, B, gLdb, gN, gK, kt * BK, n_block, tid, (kt + 1) * BK);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        #pragma unroll
        for (int kk = 0; kk < BK; kk += FK) {
            // Load all A fragments for this kk (TM of them).
            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                int base_row = warp_m + i * FM;
                threadgroup IN_T *src = &As[base_row * LDA_TGP + kk];
                #pragma unroll
                for (int e = 0; e < A_ELEM_PER_THR; ++e) {
                    Astg[i * A_ELEM_PER_THR + e] = src[a_off[e]];
                }
            }
            // Load all B fragments for this kk (TN of them).
            #pragma unroll
            for (int j = 0; j < TN; ++j) {
                int base_col = warp_n + j * FN;
                threadgroup IN_T *src = &Bs[kk * LDB_TGP + base_col];
                #pragma unroll
                for (int e = 0; e < B_ELEM_PER_THR; ++e) {
                    Bstg[j * B_ELEM_PER_THR + e] = src[b_off[e]];
                }
            }
            // Outer-product MMA loop.
            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                auto ct_a = op.get_left_input_cooperative_tensor<IN_T, IN_T, ACC_T>();
                #pragma unroll
                for (int e = 0; e < A_ELEM_PER_THR; ++e) {
                    ct_a[e] = Astg[i * A_ELEM_PER_THR + e];
                }
                #pragma unroll
                for (int j = 0; j < TN; ++j) {
                    auto ct_b = op.get_right_input_cooperative_tensor<IN_T, IN_T, ACC_T>();
                    #pragma unroll
                    for (int e = 0; e < B_ELEM_PER_THR; ++e) {
                        ct_b[e] = Bstg[j * B_ELEM_PER_THR + e];
                    }
                    auto ct_c = op.get_destination_cooperative_tensor<decltype(ct_a), decltype(ct_b), ACC_T>();
                    int frag_off = (i * TN + j) * C_ELEM_PER_THR;
                    #pragma unroll
                    for (int e = 0; e < C_ELEM_PER_THR; ++e) ct_c[e] = Cacc[frag_off + e];
                    op.run(ct_a, ct_b, ct_c);
                    #pragma unroll
                    for (int e = 0; e < C_ELEM_PER_THR; ++e) Cacc[frag_off + e] = ct_c[e];
                }
            }
        }
    }
#endif

#if !K_ALIGNED
    if (k_tail > 0) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int i = tid; i < BM * LDA_TGP; i += TGP_SIZE) As[i] = (IN_T)0;
        for (int i = tid; i < BK * LDB_TGP; i += TGP_SIZE) Bs[i] = (IN_T)0;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        load_A_tile(As, A, gLda, gM, gK, m_block, k_tiles_full * BK, tid, gK);
        load_B_tile(Bs, B, gLdb, gN, gK, k_tiles_full * BK, n_block, tid, gK);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        #pragma unroll
        for (int kk = 0; kk < BK; kk += FK) {
            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                int base_row = warp_m + i * FM;
                threadgroup IN_T *src = &As[base_row * LDA_TGP + kk];
                #pragma unroll
                for (int e = 0; e < A_ELEM_PER_THR; ++e) {
                    Astg[i * A_ELEM_PER_THR + e] = src[a_off[e]];
                }
            }
            #pragma unroll
            for (int j = 0; j < TN; ++j) {
                int base_col = warp_n + j * FN;
                threadgroup IN_T *src = &Bs[kk * LDB_TGP + base_col];
                #pragma unroll
                for (int e = 0; e < B_ELEM_PER_THR; ++e) {
                    Bstg[j * B_ELEM_PER_THR + e] = src[b_off[e]];
                }
            }
            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                auto ct_a = op.get_left_input_cooperative_tensor<IN_T, IN_T, ACC_T>();
                #pragma unroll
                for (int e = 0; e < A_ELEM_PER_THR; ++e) ct_a[e] = Astg[i * A_ELEM_PER_THR + e];
                #pragma unroll
                for (int j = 0; j < TN; ++j) {
                    auto ct_b = op.get_right_input_cooperative_tensor<IN_T, IN_T, ACC_T>();
                    #pragma unroll
                    for (int e = 0; e < B_ELEM_PER_THR; ++e) ct_b[e] = Bstg[j * B_ELEM_PER_THR + e];
                    auto ct_c = op.get_destination_cooperative_tensor<decltype(ct_a), decltype(ct_b), ACC_T>();
                    int frag_off = (i * TN + j) * C_ELEM_PER_THR;
                    #pragma unroll
                    for (int e = 0; e < C_ELEM_PER_THR; ++e) ct_c[e] = Cacc[frag_off + e];
                    op.run(ct_a, ct_b, ct_c);
                    #pragma unroll
                    for (int e = 0; e < C_ELEM_PER_THR; ++e) Cacc[frag_off + e] = ct_c[e];
                }
            }
        }
    }
#endif

    // Store accumulators to C using the pre-computed (row, col) offsets.
    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            int base_row = m_block + warp_m + i * FM;
            int base_col = n_block + warp_n + j * FN;
            int frag_off = (i * TN + j) * C_ELEM_PER_THR;
            #pragma unroll
            for (int e = 0; e < C_ELEM_PER_THR; ++e) {
                int r = base_row + c_om[e];
                int c = base_col + c_on[e];
#if EPILOGUE
                OUT_T outv = mb_epi<OUT_T, ACC_T, ACC_T>(
                    Cacc[frag_off + e], bias, r * bstride.x + c * bstride.y, beta, alpha);
#else
                OUT_T outv = (OUT_T)Cacc[frag_off + e];
#endif
#if MN_ALIGNED
                C[r * gLdc + c] = outv;
#else
                if (r < gM && c < gN)
                    C[r * gLdc + c] = outv;
#endif
            }
        }
    }
}
#endif  // MB_BUILD_MPP_GEMM
