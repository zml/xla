// complex_pack.h - deinterleave/interleave helpers for the decomposed complex GEMM.
// A complex GEMM C = A@B is computed as four real GEMMs on the real/imag planes;
// these two kernels split the interleaved operands into contiguous real planes and
// fuse the four products back into an interleaved complex result in a single pass.
#ifdef MB_BUILD_COMPLEX_PACK
#include <metal_stdlib>
using namespace metal;

#define C2 __C2__   // interleaved complex element: float2 (complex64) / half2 (complex32)
#define R  __R__    // real component scalar: float / half

// Deinterleave src[i] = (re, im) into two contiguous real planes.
// One thread per complex element; src is read once as a coalesced C2 load.
kernel void complex_split(
    device const C2 *src [[buffer(0)]],
    device       R  *re  [[buffer(1)]],
    device       R  *im  [[buffer(2)]],
    constant uint&   n   [[buffer(3)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= n) return;
    C2 v = src[i];
    re[i] = v.x;
    im[i] = v.y;
}

// Fold the four real products into one interleaved complex result:
//   C = (ar@br - ai@bi) + i*(ar@bi + ai@br) = (P - Q) + i*(S + T).
// Complex addmm folds beta*input + alpha*(.) into this pass; beta/alpha==0 are
// compiled out (NaN-safe drop of that operand).
kernel void complex_combine(
    device const R  *P   [[buffer(0)]],   // ar @ br
    device const R  *Q   [[buffer(1)]],   // ai @ bi
    device const R  *S   [[buffer(2)]],   // ar @ bi
    device const R  *T   [[buffer(3)]],   // ai @ br
    device       C2 *dst [[buffer(4)]],
    constant uint&   n   [[buffer(5)]],
#if EPILOGUE
    device const C2 *bias [[buffer(6)]],
    constant int4&  ep    [[buffer(7)]],    // (N, bias row stride, bias col stride, _)
    constant float& beta_re  [[buffer(8)]],
    constant float& beta_im  [[buffer(9)]],
    constant float& alpha_re [[buffer(10)]],
    constant float& alpha_im [[buffer(11)]],
#endif
    uint i [[thread_position_in_grid]])
{
    if (i >= n) return;
    float re = (float)P[i] - (float)Q[i];
    float im = (float)S[i] + (float)T[i];
#if EPILOGUE
    float outr = 0.0f, outi = 0.0f;
#if ALPHA_NZ
    outr += alpha_re * re - alpha_im * im;
    outi += alpha_re * im + alpha_im * re;
#endif
#if BETA_NZ
    int row = (int)i / ep.x, col = (int)i % ep.x;
    C2 b = bias[row * ep.y + col * ep.z];
    outr += beta_re * (float)b.x - beta_im * (float)b.y;
    outi += beta_re * (float)b.y + beta_im * (float)b.x;
#endif
    dst[i] = C2((R)outr, (R)outi);
#else
    dst[i] = C2((R)re, (R)im);
#endif
}
#endif  // MB_BUILD_COMPLEX_PACK
