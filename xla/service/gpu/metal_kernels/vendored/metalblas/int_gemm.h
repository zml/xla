// int_gemm.h - Register-tiled integer GEMM (no float-only simdgroup_matrix / tensor
// unit): each thread owns a TM x TN micro-tile of plain integer MACs. Accumulate in
// ACC_T (>= OUT_T), then truncate to OUT_T -> bit-exact vs torch's wrap-on-overflow.
#ifdef MB_BUILD_INT_GEMM
#include <metal_stdlib>
using namespace metal;

#define IN_T     __IN_T__
#define ACC_T    __ACC_T__
#define OUT_T    __OUT_T__
#define BM       __BM__
#define BN       __BN__
#define BK       __BK__
#define TX       __TX__
#define TY       __TY__
#define TRANS_A  __TRANS_A__
#define TRANS_B  __TRANS_B__

#define TGP_SIZE (TX * TY)

// BATCHED (bmm/baddbmm): grid z is the batch index; A/B/C (and bias) offset by their
// per-matrix strides. Defaults to 0 so the 2-D build is byte-identical at runtime.
#ifndef BATCHED
#define BATCHED 0
#endif
#if EPILOGUE
#define MB_BATCH_BUF 8
#else
#define MB_BATCH_BUF 4
#endif
#if BATCHED && EPILOGUE
#define MB_BBAT _bbat                      // per-batch bias base (baddbmm)
#else
#define MB_BBAT 0
#endif

constant constexpr int TM = BM / TY;   // rows each thread owns
constant constexpr int TN = BN / TX;   // cols each thread owns

static_assert(BM % TY == 0, "BM must be a multiple of TY");
static_assert(BN % TX == 0, "BN must be a multiple of TX");

// Dims + leading strides packed into one constant buffer (one setBytes).
struct MBIntGemmDims { int M, N, K, lda, ldb, ldc; };

kernel void int_gemm(
    device const IN_T   *A    [[buffer(0)]],
    device const IN_T   *B    [[buffer(1)]],
    device       OUT_T  *C    [[buffer(2)]],
    constant MBIntGemmDims& gP [[buffer(3)]],   // packed (M, N, K, lda, ldb, ldc)
#if EPILOGUE
    device const OUT_T *bias  [[buffer(4)]],    // addmm input; bstride = (row, col) broadcast strides
    constant int2&  bstride   [[buffer(5)]],
    constant ACC_T& beta      [[buffer(6)]],
    constant ACC_T& alpha     [[buffer(7)]],
#endif
#if BATCHED
    constant int4& batch [[buffer(MB_BATCH_BUF)]],   // (sA, sB, sC, sBias) per-batch element strides
#endif
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]])
{
#if BATCHED
    A += (int64_t)tgid.z * (int64_t)batch.x;
    B += (int64_t)tgid.z * (int64_t)batch.y;
    C += (int64_t)tgid.z * (int64_t)batch.z;
  #if EPILOGUE
    int _bbat = (int)tgid.z * batch.w;
  #endif
#endif
    const int gM = gP.M, gN = gP.N, gK = gP.K;
    const int gLda = gP.lda, gLdb = gP.ldb, gLdc = gP.ldc;

    // As transposed [BK][BM] (contiguous BM slab per k); Bs [BK][BN]. OOB loads
    // zero-fill so the compute loop is branch-free over the full BK.
    threadgroup IN_T As[BK * BM];
    threadgroup IN_T Bs[BK * BN];

    const int m_block = int(tgid.y) * BM;
    const int n_block = int(tgid.x) * BN;

    const int ty = int(tid) / TX;          // this thread's row group
    const int tx = int(tid) % TX;          // this thread's col group
    const int row0 = ty * TM;
    const int col0 = tx * TN;

    ACC_T acc[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; ++i)
        #pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] = (ACC_T)0;

    const int ktiles = (gK + BK - 1) / BK;
    for (int kt = 0; kt < ktiles; ++kt) {
        const int k_base = kt * BK;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Cooperative load of the A tile into As[k*BM + m] (transposed).
        for (int pos = int(tid); pos < BM * BK; pos += TGP_SIZE) {
            int m = pos % BM;
            int k = pos / BM;
            int gm = m_block + m;
            int gk = k_base + k;
            IN_T v = (IN_T)0;
            if (gm < gM && gk < gK)
#if TRANS_A
                v = A[gk * gLda + gm];      // A col-major: elem(m,k) = A[k*lda + m]
#else
                v = A[gm * gLda + gk];      // A row-major: elem(m,k) = A[m*lda + k]
#endif
            As[k * BM + m] = v;
        }
        // Cooperative load of the B tile into Bs[k*BN + n].
        for (int pos = int(tid); pos < BK * BN; pos += TGP_SIZE) {
            int n = pos % BN;
            int k = pos / BN;
            int gk = k_base + k;
            int gn = n_block + n;
            IN_T v = (IN_T)0;
            if (gk < gK && gn < gN)
#if TRANS_B
                v = B[gn * gLdb + gk];      // B col-major: elem(k,n) = B[n*ldb + k]
#else
                v = B[gk * gLdb + gn];      // B row-major: elem(k,n) = B[k*ldb + n]
#endif
            Bs[k * BN + n] = v;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        #pragma unroll
        for (int kk = 0; kk < BK; ++kk) {
            IN_T av[TM];
            IN_T bv[TN];
            #pragma unroll
            for (int i = 0; i < TM; ++i) av[i] = As[kk * BM + row0 + i];
            #pragma unroll
            for (int j = 0; j < TN; ++j) bv[j] = Bs[kk * BN + col0 + j];
            #pragma unroll
            for (int i = 0; i < TM; ++i)
                #pragma unroll
                for (int j = 0; j < TN; ++j)
                    acc[i][j] += (ACC_T)av[i] * (ACC_T)bv[j];
        }
    }

    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        int gm = m_block + row0 + i;
        if (gm >= gM) continue;
        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            int gn = n_block + col0 + j;
            if (gn < gN)
#if EPILOGUE
                C[gm * gLdc + gn] = mb_epi<OUT_T, ACC_T, ACC_T>(
                    acc[i][j], bias, MB_BBAT + gm * bstride.x + gn * bstride.y, beta, alpha);
#else
                C[gm * gLdc + gn] = (OUT_T)acc[i][j];
#endif
        }
    }
}
#endif  // MB_BUILD_INT_GEMM
