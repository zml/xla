// mb_epi.h - shared addmm store epilogue for the real GEMM/GEMV kernels.
// EPILOGUE=0: bare (OUT_T)acc store (== plain matmul). =1: apply bias + beta/alpha.
// Zero beta/alpha is compiled out, so a dropped operand's NaN can't reach C.
#ifndef MB_EPI_H
#define MB_EPI_H

#ifndef EPILOGUE
#define EPILOGUE 0
#endif
#ifndef BETA_NZ
#define BETA_NZ 1
#endif
#ifndef ALPHA_NZ
#define ALPHA_NZ 1
#endif

#if EPILOGUE
// C = alpha*(A@B) + beta*bias. S = accumulate type (float / int / long); bias is
// read at a caller-computed broadcast index (0 stride on stretched dims).
template <typename O, typename A, typename S>
inline O mb_epi(A acc, device const O *bias, int bidx, S beta, S alpha) {
    S v = (S)0;
#if ALPHA_NZ
    v += alpha * (S)acc;
#endif
#if BETA_NZ
    v += beta * (S)bias[bidx];
#endif
    return (O)v;
}
#endif

#endif  // MB_EPI_H
