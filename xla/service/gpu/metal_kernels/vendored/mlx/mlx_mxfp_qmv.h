// MXFP8/MXFP4 qmv, vendored from MLX fp_quantized (group_size=32, E8M0 scales).
// ABI: x bf16[M,K], w u32-packed, scales u8 E8M0, y bf16[M,N], dims {M,K,N,K/g}.
// Grid (M, ceil(N/8), 1), tg (32, 2, 1).

#include <metal_stdlib>
using namespace metal;

#define MLX_MTL_CONST static constant constexpr const
#ifndef METAL_FUNC
#define METAL_FUNC inline
#endif
MLX_MTL_CONST int SIMD_SIZE = 32;

// ---- fp8.h (verbatim) ----
struct fp8_e4m3 {
  operator float() {
    uint16_t v = (bits & 127) << 7;
    half converted = as_type<half>(v);
    converted *= 256.0;
    auto sign = bits & 128;
    return static_cast<float>(sign ? -converted : converted);
  }
  uint8_t bits;
};

struct fp8_e8m0 {
  operator float() {
    uint32_t out = (bits == 0 ? 0x400000 : (static_cast<uint16_t>(bits) << 23));
    return as_type<float>(out);
  }
  uint8_t bits;
};

// ---- fp4.h (verbatim) ----
struct fp4_e2m1 {
  operator float() {
    half converted = as_type<half>(ushort((bits & 7) << 9));
    converted *= 16384.0;
    return static_cast<float>(bits & 8 ? -converted : converted);
  }
  uint8_t bits;
};

// ---- quantized helpers (verbatim from fp_quantized.h) ----
template <int wsize = 8, int bits = 4>
inline constexpr short get_pack_factor() {
  return wsize / bits;
}

template <int wsize = 8>
inline constexpr short get_bytes_per_pack() {
  return wsize / 8;
}

template <typename T, int group_size>
static inline T dequantize_scale(uint8_t s) {
  if constexpr (group_size == 16) {
    return T(*(thread fp8_e4m3*)(&s));
  } else {
    return T(*(thread fp8_e8m0*)(&s));
  }
}

template <int bits, typename U = float>
struct Dequantize {
  U operator()(uint8_t x) {
    if constexpr (bits == 8) {
      return U(*(thread fp8_e4m3*)(&x));
    } else {
      return U(*(thread fp4_e2m1*)(&x));
    }
  }
};

template <typename T, typename U, int values_per_thread>
inline void load_vector(const device T* x, thread U* x_thread) {
#pragma unroll
  for (int i = 0; i < values_per_thread; i++) {
    x_thread[i] = x[i];
  }
}

template <typename T, typename U, int values_per_thread>
inline void load_vector_safe(const device T* x, thread U* x_thread, int N) {
  for (int i = 0; i < N; i++) {
    x_thread[i] = x[i];
  }
  for (int i = N; i < values_per_thread; i++) {
    x_thread[i] = 0;
  }
}

template <typename U, int values_per_thread, int bits>
inline U qdot(const device uint8_t* w, const thread U* x_thread, U scale) {
  U accum = 0;
  if constexpr (bits == 4) {
    const device uint16_t* ws = (const device uint16_t*)w;
    for (int i = 0; i < (values_per_thread / 4); i++) {
      accum +=
          (x_thread[4 * i] * Dequantize<4>{}(ws[i]) +
           x_thread[4 * i + 1] * Dequantize<4>{}(ws[i] >> 4) +
           x_thread[4 * i + 2] * Dequantize<4>{}(ws[i] >> 8) +
           x_thread[4 * i + 3] * Dequantize<4>{}(ws[i] >> 12));
    }
  } else {
    for (int i = 0; i < values_per_thread; i++) {
      accum += x_thread[i] * Dequantize<8>{}(w[i]);
    }
  }
  return scale * accum;
}

template <typename U, int values_per_thread, int bits>
inline U
qdot_safe(const device uint8_t* w, const thread U* x_thread, U scale, int N) {
  U accum = 0;
  if constexpr (bits == 4) {
    const device uint16_t* ws = (const device uint16_t*)w;
    for (int i = 0; i < (N / 4); i++) {
      accum +=
          (x_thread[4 * i] * Dequantize<4>{}(ws[i]) +
           x_thread[4 * i + 1] * Dequantize<4>{}(ws[i] >> 4) +
           x_thread[4 * i + 2] * Dequantize<4>{}(ws[i] >> 8) +
           x_thread[4 * i + 3] * Dequantize<4>{}(ws[i] >> 12));
    }
  } else {
    for (int i = 0; i < N; i++) {
      accum += x_thread[i] * Dequantize<8>{}(w[i]);
    }
  }
  return scale * accum;
}

// ---- fp_qmv_fast_impl / fp_qmv_impl (verbatim from fp_quantized.h) ----
template <typename T, int group_size, int bits>
METAL_FUNC void fp_qmv_fast_impl(
    const device uint32_t* w,
    const device uint8_t* scales,
    const device T* x,
    device T* y,
    int in_vec_size,
    int out_vec_size,
    uint3 tid,
    uint simd_gid,
    uint simd_lid) {
  constexpr int packs_per_thread = 2;
  constexpr int num_simdgroups = 2;
  constexpr int results_per_simdgroup = 4;
  constexpr int pack_factor = get_pack_factor<32, bits>();
  constexpr int bytes_per_pack = get_bytes_per_pack<32>();
  constexpr int values_per_thread = pack_factor * packs_per_thread;
  constexpr int block_size = values_per_thread * SIMD_SIZE;
  constexpr int scale_step_per_thread = group_size / values_per_thread;

  const device uint8_t* ws = (const device uint8_t*)w;

  typedef float U;
  thread U x_thread[values_per_thread];
  thread U result[results_per_simdgroup] = {0};

  const int in_vec_size_w = in_vec_size * bytes_per_pack / pack_factor;
  const int in_vec_size_g = in_vec_size / group_size;
  const int out_row = tid.y * (num_simdgroups * results_per_simdgroup) +
      simd_gid * results_per_simdgroup;

  ws += out_row * in_vec_size_w + simd_lid * packs_per_thread * bytes_per_pack;
  scales += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
  x += tid.x * in_vec_size + simd_lid * values_per_thread;
  y += tid.x * out_vec_size + out_row;

  for (int k = 0; k < in_vec_size; k += block_size) {
    load_vector<T, U, values_per_thread>(x, x_thread);

    for (int row = 0; row < results_per_simdgroup; row++) {
      auto wl = (const device uint8_t*)(ws + row * in_vec_size_w);
      const device auto* sl = scales + row * in_vec_size_g;

      U s = dequantize_scale<U, group_size>(sl[0]);
      result[row] += qdot<U, values_per_thread, bits>(wl, x_thread, s);
    }

    ws += block_size * bytes_per_pack / pack_factor;
    scales += block_size / group_size;
    x += block_size;
  }

  for (int row = 0; row < results_per_simdgroup; row++) {
    result[row] = simd_sum(result[row]);
    if (simd_lid == 0) {
      y[row] = static_cast<T>(result[row]);
    }
  }
}

template <typename T, int group_size, int bits>
METAL_FUNC void fp_qmv_impl(
    const device uint32_t* w,
    const device uint8_t* scales,
    const device T* x,
    device T* y,
    int in_vec_size,
    int out_vec_size,
    uint3 tid,
    uint simd_gid,
    uint simd_lid) {
  constexpr int num_simdgroups = 2;
  constexpr int results_per_simdgroup = 4;
  constexpr int packs_per_thread = 1;
  constexpr int pack_factor = get_pack_factor<32, bits>();
  constexpr int bytes_per_pack = get_bytes_per_pack<32>();

  constexpr int values_per_thread = pack_factor * packs_per_thread;
  constexpr int block_size = values_per_thread * SIMD_SIZE;
  constexpr int scale_step_per_thread = group_size / values_per_thread;

  const device uint8_t* ws = (const device uint8_t*)w;

  typedef float U;

  thread U x_thread[values_per_thread];
  thread U result[results_per_simdgroup] = {0};

  const int in_vec_size_w = in_vec_size * bytes_per_pack / pack_factor;
  const int in_vec_size_g = in_vec_size / group_size;
  const int out_row = tid.y * (num_simdgroups * results_per_simdgroup) +
      simd_gid * results_per_simdgroup;
  const int used_out_row = min(out_vec_size - results_per_simdgroup, out_row);

  if (out_row >= out_vec_size) {
    return;
  }

  if (out_vec_size < (num_simdgroups * results_per_simdgroup)) {
    ws +=
        out_row * in_vec_size_w + simd_lid * packs_per_thread * bytes_per_pack;
    scales += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
    x += tid.x * in_vec_size + simd_lid * values_per_thread;
    y += tid.x * out_vec_size + out_row;

    int k = 0;
    for (; k < in_vec_size - block_size; k += block_size) {
      load_vector<T, U, values_per_thread>(x, x_thread);

      for (int row = 0;
           row < results_per_simdgroup && out_row + row < out_vec_size;
           row++) {
        auto wl = (const device uint8_t*)(ws + row * in_vec_size_w);
        const device auto* sl = scales + row * in_vec_size_g;

        U s = dequantize_scale<U, group_size>(sl[0]);
        result[row] += qdot<U, values_per_thread, bits>(wl, x_thread, s);
      }

      ws += block_size * bytes_per_pack / pack_factor;
      scales += block_size / group_size;
      x += block_size;
    }
    const int remaining = clamp(
        static_cast<int>(in_vec_size - k - simd_lid * values_per_thread),
        0,
        values_per_thread);
    if (remaining > 0) {
      load_vector_safe<T, U, values_per_thread>(x, x_thread, remaining);

      for (int row = 0;
           row < results_per_simdgroup && out_row + row < out_vec_size;
           row++) {
        auto wl = (const device uint8_t*)(ws + row * in_vec_size_w);
        const device auto* sl = scales + row * in_vec_size_g;

        U s = dequantize_scale<U, group_size>(sl[0]);
        result[row] += qdot<U, values_per_thread, bits>(wl, x_thread, s);
      }
    }

    for (int row = 0;
         row < results_per_simdgroup && out_row + row < out_vec_size;
         row++) {
      result[row] = simd_sum(result[row]);
      if (simd_lid == 0) {
        y[row] = static_cast<T>(result[row]);
      }
    }
  }

  else {
    ws += used_out_row * in_vec_size_w +
        simd_lid * packs_per_thread * bytes_per_pack;
    scales += used_out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
    x += tid.x * in_vec_size + simd_lid * values_per_thread;
    y += tid.x * out_vec_size + used_out_row;

    int k = 0;
    for (; k < in_vec_size - block_size; k += block_size) {
      load_vector<T, U, values_per_thread>(x, x_thread);

      for (int row = 0; row < results_per_simdgroup; row++) {
        auto wl = (const device uint8_t*)(ws + row * in_vec_size_w);
        const device auto* sl = scales + row * in_vec_size_g;

        U s = dequantize_scale<U, group_size>(sl[0]);
        result[row] += qdot<U, values_per_thread, bits>(wl, x_thread, s);
      }

      ws += block_size * bytes_per_pack / pack_factor;
      scales += block_size / group_size;
      x += block_size;
    }
    const int remaining = clamp(
        static_cast<int>(in_vec_size - k - simd_lid * values_per_thread),
        0,
        values_per_thread);
    if (remaining > 0) {
      load_vector_safe<T, U, values_per_thread>(x, x_thread, remaining);

      for (int row = 0; row < results_per_simdgroup; row++) {
        auto wl = (const device uint8_t*)(ws + row * in_vec_size_w);
        const device auto* sl = scales + row * in_vec_size_g;

        U s = dequantize_scale<U, group_size>(sl[0]);
        result[row] +=
            qdot_safe<U, values_per_thread, bits>(wl, x_thread, s, remaining);
      }
    }
    for (int row = 0; row < results_per_simdgroup; row++) {
      result[row] = simd_sum(result[row]);
      if (simd_lid == 0) {
        y[row] = static_cast<T>(result[row]);
      }
    }
  }
}

// ---- thin entry points (ours) ----
// ABI: buffer0 x, buffer1 w, buffer2 scales, buffer3 y, buffer4 dims
// (see file header). Grid = (M, ceil(N / 8), 1), threadgroup = (32, 2, 1).
#define MXFP_QMV_ENTRY(NAME, IMPL, GS, BITS)                              \
  kernel void NAME(                                                       \
      const device bfloat* x [[buffer(0)]],                              \
      const device uint32_t* w [[buffer(1)]],                            \
      const device uint8_t* scales [[buffer(2)]],                        \
      device bfloat* y [[buffer(3)]],                                    \
      const constant int4& dims [[buffer(4)]],                           \
      uint3 tid [[threadgroup_position_in_grid]],                        \
      uint simd_gid [[simdgroup_index_in_threadgroup]],                  \
      uint simd_lid [[thread_index_in_simdgroup]]) {                     \
    IMPL<bfloat, GS, BITS>(                                              \
        w, scales, x, y, dims.y, dims.z, tid, simd_gid, simd_lid);       \
  }

MXFP_QMV_ENTRY(mxfp8_qmv_fast, fp_qmv_fast_impl, 32, 8)
MXFP_QMV_ENTRY(mxfp8_qmv, fp_qmv_impl, 32, 8)
MXFP_QMV_ENTRY(mxfp4_qmv_fast, fp_qmv_fast_impl, 32, 4)
MXFP_QMV_ENTRY(mxfp4_qmv, fp_qmv_impl, 32, 4)
