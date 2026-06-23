// simd_gemm.h - Portable simdgroup_matrix<T,8,8> tiled GEMM (Metal 3; no tensor unit needed).
#ifdef MB_BUILD_SIMD_GEMM
#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>

using namespace metal;

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
#define OUT_IS_ACC  __OUT_IS_ACC__

#define SG_SIZE     32
#define WARPS       (WM * WN)
#define TGP_SIZE    (WARPS * SG_SIZE)

constant constexpr int WT_M = BM / WM;
constant constexpr int WT_N = BN / WN;
constant constexpr int TM   = WT_M / 8;
constant constexpr int TN   = WT_N / 8;

constant constexpr int PAD_A = 16 / sizeof(IN_T);
constant constexpr int PAD_B = 16 / sizeof(IN_T);
constant constexpr int LDA_TGP = BK + PAD_A;
constant constexpr int LDB_TGP = BN + PAD_B;

// 16-byte load granularity: VEC chosen so VecF is always 16 bytes.
constant constexpr int VEC = 16 / sizeof(IN_T);
constant constexpr int A_TCOLS = BK / VEC;
constant constexpr int A_ROW_STEP = TGP_SIZE / A_TCOLS;
constant constexpr int B_TCOLS = BN / VEC;
constant constexpr int B_ROW_STEP = TGP_SIZE / B_TCOLS;

struct alignas(16) VecF { IN_T v[VEC]; };

static_assert(BM % (8 * WM) == 0, "BM must be a multiple of 8*WM");
static_assert(BN % (8 * WN) == 0, "BN must be a multiple of 8*WN");
static_assert(BK % 8 == 0,        "BK must be a multiple of 8");
static_assert(BK % VEC == 0,      "BK must be divisible by VEC");
static_assert(BN % VEC == 0,      "BN must be divisible by VEC");

static inline void load_A_tile(threadgroup IN_T   *As,
                               const device IN_T  *A,
                               int                 lda,
                               int                 M,
                               int                 K,
                               int                 a_row0,
                               int                 a_col0,
                               int                 tid,
                               int                 kbound)
{
    int local_row0 = tid / A_TCOLS;
    int local_col0 = (tid % A_TCOLS) * VEC;
    #pragma unroll
    for (int r = 0; r < BM; r += A_ROW_STEP) {
        int rl = local_row0 + r;
        if (rl >= BM) break;
        VecF acc;
        #pragma unroll
        for (int i = 0; i < VEC; ++i) acc.v[i] = (IN_T)0;
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
                int gk = gc_k0 + i;
                bool ok = m_ok && (gk < kbound);
                acc.v[i] = ok ? A[(a_row0 + rl) * lda + gk] : (IN_T)0;
            }
        }
#endif
        *((threadgroup VecF*)(&As[rl * LDA_TGP + local_col0])) = acc;
    }
}

static inline void load_B_tile(threadgroup IN_T   *Bs,
                               const device IN_T  *B,
                               int                 ldb,
                               int                 N,
                               int                 K,
                               int                 b_row0,
                               int                 b_col0,
                               int                 tid,
                               int                 kbound)
{
    int local_row0 = tid / B_TCOLS;
    int local_col0 = (tid % B_TCOLS) * VEC;
    int n_global   = b_col0 + local_col0;
    #pragma unroll
    for (int r = 0; r < BK; r += B_ROW_STEP) {
        int rl = local_row0 + r;
        if (rl >= BK) break;
        VecF acc;
        #pragma unroll
        for (int i = 0; i < VEC; ++i) acc.v[i] = (IN_T)0;
        int gk = b_row0 + rl;
#if TRANS_B
        bool k_ok = gk < kbound;
        if (k_ok) {
            #pragma unroll
            for (int i = 0; i < VEC; ++i) {
                int gn = n_global + i;
                bool ok = gn < N;
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
                int gn = n_global + i;
                bool ok = gn < N;
                acc.v[i] = ok ? B[gk * ldb + gn] : (IN_T)0;
            }
        }
#endif
        *((threadgroup VecF*)(&Bs[rl * LDB_TGP + local_col0])) = acc;
    }
}

// Dims + leading strides packed into one constant buffer (one setBytes, not six).
struct MBGemmDims { int M, N, K, lda, ldb, ldc; };

kernel void simd_gemm(
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
    threadgroup IN_T As[BM * LDA_TGP];
    threadgroup IN_T Bs[BK * LDB_TGP];

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

    simdgroup_matrix<ACC_T, 8, 8> Cfrag[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; ++i)
        #pragma unroll
        for (int j = 0; j < TN; ++j)
            Cfrag[i][j] = simdgroup_matrix<ACC_T, 8, 8>(0);

    int k_tiles_full = gK / BK;
    int k_tail       = gK - k_tiles_full * BK;

    for (int kt = 0; kt < k_tiles_full; ++kt) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        load_A_tile(As, A, gLda, gM, gK, m_block, kt * BK, tid, (kt + 1) * BK);
        load_B_tile(Bs, B, gLdb, gN, gK, kt * BK, n_block, tid, (kt + 1) * BK);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        #pragma unroll
        for (int kk = 0; kk < BK; kk += 8) {
            simdgroup_matrix<IN_T, 8, 8> Afrag[TM];
            simdgroup_matrix<IN_T, 8, 8> Bfrag[TN];
            #pragma unroll
            for (int i = 0; i < TM; ++i)
                simdgroup_load(Afrag[i],
                               &As[(warp_m + i * 8) * LDA_TGP + kk],
                               LDA_TGP);
            #pragma unroll
            for (int j = 0; j < TN; ++j)
                simdgroup_load(Bfrag[j],
                               &Bs[kk * LDB_TGP + warp_n + j * 8],
                               LDB_TGP);
            #pragma unroll
            for (int i = 0; i < TM; ++i)
                #pragma unroll
                for (int j = 0; j < TN; ++j)
                    simdgroup_multiply_accumulate(Cfrag[i][j], Afrag[i], Bfrag[j], Cfrag[i][j]);
        }
    }

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
        for (int kk = 0; kk < BK; kk += 8) {
            simdgroup_matrix<IN_T, 8, 8> Afrag[TM];
            simdgroup_matrix<IN_T, 8, 8> Bfrag[TN];
            #pragma unroll
            for (int i = 0; i < TM; ++i)
                simdgroup_load(Afrag[i], &As[(warp_m + i * 8) * LDA_TGP + kk], LDA_TGP);
            #pragma unroll
            for (int j = 0; j < TN; ++j)
                simdgroup_load(Bfrag[j], &Bs[kk * LDB_TGP + warp_n + j * 8], LDB_TGP);
            #pragma unroll
            for (int i = 0; i < TM; ++i)
                #pragma unroll
                for (int j = 0; j < TN; ++j)
                    simdgroup_multiply_accumulate(Cfrag[i][j], Afrag[i], Bfrag[j], Cfrag[i][j]);
        }
    }
#endif

    // (row, col) each lane owns within an 8x8 simdgroup_matrix.
    const short qid = lane / 4;
    const short fm  = (qid & 4) + ((lane / 2) % 4);
    const short fn  = (qid & 2) * 2 + (lane % 2) * 2;

    #pragma unroll
    for (int i = 0; i < TM; ++i)
        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            int row = m_block + warp_m + i * 8 + fm;
            int col = n_block + warp_n + j * 8 + fn;
#if EPILOGUE
            // addmm always takes the per-element path: the bulk simdgroup_store can't
            // fold a per-(row,col) bias. The lane owns (row,col) and (row,col+1).
            ACC_T te0 = Cfrag[i][j].thread_elements()[0];
            ACC_T te1 = Cfrag[i][j].thread_elements()[1];
            int cc0 = col, cc1 = col + 1;
            if (row < gM && cc0 < gN)
                C[row * gLdc + cc0] = mb_epi<OUT_T, ACC_T, ACC_T>(
                    te0, bias, row * bstride.x + cc0 * bstride.y, beta, alpha);
            if (row < gM && cc1 < gN)
                C[row * gLdc + cc1] = mb_epi<OUT_T, ACC_T, ACC_T>(
                    te1, bias, row * bstride.x + cc1 * bstride.y, beta, alpha);
#elif MN_ALIGNED
            int row0 = m_block + warp_m + i * 8;
            int col0 = n_block + warp_n + j * 8;
#if OUT_IS_ACC
            simdgroup_store(Cfrag[i][j], &C[row0 * gLdc + col0], gLdc);
#else
            simdgroup_matrix<OUT_T, 8, 8> Cout;
            #pragma unroll
            for (int kk = 0; kk < 2; ++kk)
                Cout.thread_elements()[kk] = (OUT_T)Cfrag[i][j].thread_elements()[kk];
            simdgroup_store(Cout, &C[row0 * gLdc + col0], gLdc);
#endif
            (void)row; (void)col;
#else
            ACC_T te0 = Cfrag[i][j].thread_elements()[0];
            ACC_T te1 = Cfrag[i][j].thread_elements()[1];
            int rr = row;
            int cc0 = col + 0;
            int cc1 = col + 1;
            if (rr < gM && cc0 < gN) C[rr * gLdc + cc0] = OUT_T(te0);
            if (rr < gM && cc1 < gN) C[rr * gLdc + cc1] = OUT_T(te1);
#endif
        }
}
#endif  // MB_BUILD_SIMD_GEMM
