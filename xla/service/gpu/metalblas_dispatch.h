
#ifndef METALBLAS_PORT_DISPATCH_H_
#define METALBLAS_PORT_DISPATCH_H_

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace metalblas_port {

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

Dtype ParseDtype(const std::string& name);
const char* DtypeName(Dtype dt);

constexpr bool kHasTensorUnit = false;
constexpr bool kHasMetal4 = true;

constexpr double kAutotuneMargin = 0.03;
constexpr double kTallNarrowMargin = 0.01;

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

Decision decide(const Inputs& in);

int64_t threadgroup_bytes(int64_t BM, int64_t BN, int64_t BK,
                          int64_t dtype_bytes);
int64_t round_swizzle_log(int64_t tiles_m, int64_t tiles_n);

std::array<int64_t, 5> pick_simd_tile(int64_t M, int64_t N, int64_t K,
                                      Dtype dtype);
std::array<int64_t, 3> pick_m5_tensor_tile(int64_t M, int64_t N, int64_t K,
                                           Dtype dtype);
std::array<int64_t, 6> pick_m5_tile(int64_t M, int64_t N, int64_t K,
                                    Dtype dtype);
std::array<int64_t, 5> pick_int_tile(int64_t M, int64_t N, int64_t K,
                                     Dtype dtype);

struct M5TensorCandidates {
  std::vector<std::array<int64_t, 3>> candidates;
  double margin;
};
M5TensorCandidates m5_tensor_tile_candidates(int64_t M, int64_t N, int64_t K,
                                             Dtype dtype);

bool is_splitk_regime(int64_t M, int64_t N, int64_t K, Dtype dtype);
bool is_conv_regime(int64_t M, int64_t N, int64_t K, Dtype dtype);
std::vector<std::array<int64_t, 4>> splitk_specs(int64_t M, int64_t N,
                                                 int64_t K);          // (BM,BN,NSG,G)
std::vector<std::array<int64_t, 3>> conv_specs(int64_t M, int64_t N,
                                               int64_t K);            // (BMW,BNO,NSG)

bool is_gemv_bt_regime(int64_t M, int64_t N, int64_t K, Dtype dtype);
int64_t largest_pow2_le(int64_t x);
std::vector<std::array<int64_t, 3>> gemv_bt_specs(int64_t M, int64_t N,
                                                  int64_t K, Dtype dtype,
                                                  int64_t align,
                                                  bool trans_b = false);

bool is_gemv_bt_trans_regime(int64_t M, int64_t N, int64_t K, Dtype dtype);

int64_t int_clamp_vec(int64_t vec, int64_t ld, int64_t off);

struct GemvPick {
  int64_t vec;
  int64_t nwarps;
};
GemvPick gemv_pick(int64_t cols, int64_t ldb_align, Dtype dtype, bool vec_ok,
                   bool k_present, int64_t k);
int64_t gemv_nt_pick_key(int64_t k, int64_t ld, int64_t off, Dtype dtype);

}  // namespace metalblas_port

#endif  // METALBLAS_PORT_DISPATCH_H_
