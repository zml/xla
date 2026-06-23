// metalblas_dispatch.h
//
// STANDALONE C++17 reimplementation of the metalBLAS matmul dispatcher
// DECISIONS (pure routing/tile-selection logic). No torch, no Metal, no GPU.
//
// This file (and its .cc) do NOT depend on xla/ or zml/ in any way. They live
// entirely under metalBLAS/port and mirror the pure pickers of
// metalblas/dispatch.py plus the path-selection + launch-dim arithmetic of
// matmul()/_int_matmul()/_complex_matmul()/_dispatch_gemv() exactly as the
// GPU-free reference dumper port/gen_golden.py does.
//
// HAS_TENSOR_UNIT is hardcoded false (matches the golden env
// METALBLAS_HAS_TENSOR_UNIT=0); HAS_METAL4 is hardcoded true (matches the
// dumper constant for the M4-class target).

#ifndef METALBLAS_PORT_DISPATCH_H_
#define METALBLAS_PORT_DISPATCH_H_

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace metalblas_port {

// The 10 dtypes the dispatcher routes (golden.json `dtype` strings).
enum class Dtype {
  kFloat32,
  kFloat16,
  kBfloat16,
  kInt8,
  kUint8,
  kInt16,
  kInt32,
  kInt64,
  kComplex64,
  kComplex32,
};

// Parse / render the JSON dtype name. ParseDtype aborts on an unknown name.
Dtype ParseDtype(const std::string& name);
const char* DtypeName(Dtype dt);

// Compile-time target flags. These match the environment under which the
// golden reference (port/golden.json) was produced:
//   METALBLAS_HAS_TENSOR_UNIT=0  -> _HAS_TENSOR_UNIT == false (pre-M5 GEMV branches)
//   METALBLAS_HAS_METAL4=1       -> has_metal4() modeled as true (M4-class target)
constexpr bool kHasTensorUnit = false;
constexpr bool kHasMetal4 = true;

// Autotuner margins (dispatch.py:237,240). Emitted deterministically even though
// the timing autotuner never runs in this port.
constexpr double kAutotuneMargin = 0.03;
constexpr double kTallNarrowMargin = 0.01;

// Routing inputs: the 9 keys of golden.json's `inputs` object. trans_a/trans_b
// here are the *resolved* transpose flags _resolve_inputs would produce; the
// dumper feeds them directly (it precomputes lda/ldb/strides for each align case).
struct Inputs {
  int64_t m = 0;
  int64_t n = 0;
  int64_t k = 0;
  Dtype dtype = Dtype::kFloat32;
  bool trans_a = false;
  bool trans_b = false;
  int64_t lda = 0;
  int64_t ldb = 0;
  int64_t offset = 0;
};

// Routing decision: the 14 keys of golden.json's `decision` object, same order.
struct Decision {
  std::string path;                       // complex|int|gemv_t|gemv_nt|gemv_bt|gemm
  std::vector<int64_t> tile;              // [] / 3 / 5 / 6 elems by kernel
  int64_t vec = 0;                        // GEMV columns-per-lane, else 0
  int64_t nwarps = 0;                     // GEMV simdgroups/threadgroup, else 0
  int64_t ncols = 0;                      // gemv_bt output-col blocking (1 = none); else 0
  std::array<int64_t, 3> threads{0, 0, 0};
  std::array<int64_t, 3> group{0, 0, 0};
  int64_t swizzle_log = 0;
  bool trans_a = false;                   // echo of inputs
  bool trans_b = false;                   // echo of inputs
  int64_t lda = 0;                        // echo of inputs
  int64_t ldb = 0;                        // echo of inputs
  std::vector<std::array<int64_t, 3>> candidates;  // m5_tensor candidate family
  double margin = kAutotuneMargin;        // 0.03 default, 0.01 tall-narrow
};

// The full routing decision, mirroring matmul()'s path selection
// (dispatch.py:1020-1156) exactly as gen_golden.py's decide() does.
Decision decide(const Inputs& in);

// --- Pure pickers (exposed for completeness / testing). All bit-for-bit
// ports of the like-named functions in dispatch.py. ---

int64_t threadgroup_bytes(int64_t BM, int64_t BN, int64_t BK,
                          int64_t dtype_bytes);
int64_t round_swizzle_log(int64_t tiles_m, int64_t tiles_n);

// (BM, BN, BK, WM, WN)
std::array<int64_t, 5> pick_simd_tile(int64_t M, int64_t N, int64_t K,
                                      Dtype dtype);
// (BM, BN, NSG)
std::array<int64_t, 3> pick_m5_tensor_tile(int64_t M, int64_t N, int64_t K,
                                           Dtype dtype);
// (BM, BN, BK, WM, WN, dbuf)  -- dbuf returned as 0/1 in slot 5
std::array<int64_t, 6> pick_m5_tile(int64_t M, int64_t N, int64_t K,
                                    Dtype dtype);
// (BM, BN, BK, TX, TY)
std::array<int64_t, 5> pick_int_tile(int64_t M, int64_t N, int64_t K,
                                     Dtype dtype);

// (candidate list, margin). candidate[0] == primary.
struct M5TensorCandidates {
  std::vector<std::array<int64_t, 3>> candidates;
  double margin;
};
M5TensorCandidates m5_tensor_tile_candidates(int64_t M, int64_t N, int64_t K,
                                             Dtype dtype);

// Regime gates + candidate spec families (verified GPU-free; never the
// deterministic decision but ported for completeness).
bool is_splitk_regime(int64_t M, int64_t N, int64_t K, Dtype dtype);
bool is_conv_regime(int64_t M, int64_t N, int64_t K, Dtype dtype);
std::vector<std::array<int64_t, 4>> splitk_specs(int64_t M, int64_t N,
                                                 int64_t K);          // (BM,BN,NSG,G)
std::vector<std::array<int64_t, 3>> conv_specs(int64_t M, int64_t N,
                                               int64_t K);            // (BMW,BNO,NSG)

// Batched thin-M GEMV (gemv_bt) regime + candidate family (dispatch.py:373,
// 405-455). Like splitk/conv: never the deterministic decision, ported for
// completeness. gemv_bt_specs returns (vec, nwarps, ncols): trans_b entries
// carry NCOLS, row-major entries set ncols=1. `align` is the (ld|offset) the
// autotuner passes for the VEC clamp.
bool is_gemv_bt_regime(int64_t M, int64_t N, int64_t K, Dtype dtype);
int64_t largest_pow2_le(int64_t x);
std::vector<std::array<int64_t, 3>> gemv_bt_specs(int64_t M, int64_t N,
                                                  int64_t K, Dtype dtype,
                                                  int64_t align,
                                                  bool trans_b = false);

// gemv_bt TRANS_B deterministic regime (PORT POLICY #2, gen_golden.py
// _is_gemv_bt_trans_regime): transposed thin-M decode batches x[M,K] @ W[N,K]^T,
// low-precision, 2<=M<=16, 16<=N<=4096, 64<=K<=8192. Unlike the candidate
// families above this IS a deterministic decision (decide() routes it).
bool is_gemv_bt_trans_regime(int64_t M, int64_t N, int64_t K, Dtype dtype);

// Integer VEC alignment clamp (dispatch.py:934-939).
int64_t int_clamp_vec(int64_t vec, int64_t ld, int64_t off);

// GEMV pickers. gemv_pick returns (vec, nwarps); the threadgroup size is
// nwarps*32. gemv_nt_pick returns the chosen VEC key (1/2/4).
struct GemvPick {
  int64_t vec;
  int64_t nwarps;
};
// `cols` = output width; `ldb_align` = (ld | offset) the dumper passes in;
// `k_present` mirrors Python `k is not None` (always true on the float GEMV path).
GemvPick gemv_pick(int64_t cols, int64_t ldb_align, Dtype dtype, bool vec_ok,
                   bool k_present, int64_t k);
// nt_dict membership: VEC>1 keys exist iff (!HAS_TENSOR_UNIT || dtype != fp32).
int64_t gemv_nt_pick_key(int64_t k, int64_t ld, int64_t off, Dtype dtype);

}  // namespace metalblas_port

#endif  // METALBLAS_PORT_DISPATCH_H_
