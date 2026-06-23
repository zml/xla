// metalblas_dispatch.cc -- see metalblas_dispatch.h for the contract.
//
// Bit-for-bit C++17 port of the pure routing/tile logic in metalblas/dispatch.py
// and the GPU-free reference decide() in port/gen_golden.py. Branch order and
// every numeric constant match the Python sources exactly.

#include "xla/service/gpu/metalblas_dispatch.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <string>

namespace metalblas_port {

namespace {

// Integer ceil-div used pervasively as (x + d - 1) / d in the Python source.
inline int64_t ceil_div(int64_t x, int64_t d) { return (x + d - 1) / d; }

inline int64_t imax(int64_t a, int64_t b) { return a > b ? a : b; }
inline int64_t imin(int64_t a, int64_t b) { return a < b ? a : b; }

// True for any dtype that is NOT float32 (the Python `is_lp` predicate / the
// "dtype is not torch.float32" tests in the float pickers).
inline bool is_lp(Dtype dt) { return dt != Dtype::kFloat32; }

}  // namespace

Dtype ParseDtype(const std::string& name) {
  if (name == "float32") return Dtype::kFloat32;
  if (name == "float16") return Dtype::kFloat16;
  if (name == "bfloat16") return Dtype::kBfloat16;
  if (name == "int8") return Dtype::kInt8;
  if (name == "uint8") return Dtype::kUint8;
  if (name == "int16") return Dtype::kInt16;
  if (name == "int32") return Dtype::kInt32;
  if (name == "int64") return Dtype::kInt64;
  if (name == "complex64") return Dtype::kComplex64;
  if (name == "complex32") return Dtype::kComplex32;
  std::abort();  // unknown dtype name
}

const char* DtypeName(Dtype dt) {
  switch (dt) {
    case Dtype::kFloat32: return "float32";
    case Dtype::kFloat16: return "float16";
    case Dtype::kBfloat16: return "bfloat16";
    case Dtype::kInt8: return "int8";
    case Dtype::kUint8: return "uint8";
    case Dtype::kInt16: return "int16";
    case Dtype::kInt32: return "int32";
    case Dtype::kInt64: return "int64";
    case Dtype::kComplex64: return "complex64";
    case Dtype::kComplex32: return "complex32";
  }
  std::abort();
}

// ===========================================================================
// _threadgroup_bytes (dispatch.py:558-562)
// ===========================================================================
int64_t threadgroup_bytes(int64_t BM, int64_t BN, int64_t BK,
                          int64_t dtype_bytes) {
  int64_t pad = imax(1, 16 / dtype_bytes);
  int64_t lda = BK + pad;
  int64_t ldb = BN + pad;
  return (BM * lda + BK * ldb) * dtype_bytes;
}

// ===========================================================================
// _round_swizzle_log (dispatch.py:780-784)
// ===========================================================================
int64_t round_swizzle_log(int64_t tiles_m, int64_t tiles_n) {
  if (tiles_m * tiles_n < 32) return 0;
  return 2;
}

// ===========================================================================
// _pick_simd_tile (dispatch.py:565-617)
// ===========================================================================
std::array<int64_t, 5> pick_simd_tile(int64_t M, int64_t N, int64_t K,
                                      Dtype dtype) {
  int64_t dtype_bytes = (dtype == Dtype::kFloat32) ? 4 : 2;
  static const int64_t kCands[][5] = {
      {128, 128, 16, 4, 4}, {128, 128, 32, 4, 4}, {64, 128, 16, 2, 4},
      {128, 64, 16, 4, 2},  {64, 64, 32, 2, 2},   {64, 64, 16, 2, 2},
      {32, 128, 16, 1, 4},  {128, 32, 16, 4, 1},  {32, 64, 16, 1, 2},
      {64, 32, 16, 2, 1},   {32, 32, 32, 1, 1},   {32, 32, 16, 1, 1},
      {16, 16, 16, 1, 1},
  };

  bool have_best = false;
  // score is up to 3 doubles; we compare lexicographically. In one call all
  // surviving candidates share the same `ops` regime so the tuple shape (number
  // of meaningful score components) is uniform -> always mutually comparable.
  bool large_ops = false;  // which score shape this call uses
  double best_score[3] = {0, 0, 0};
  std::array<int64_t, 5> best_tile{0, 0, 0, 0, 0};

  for (const auto& c : kCands) {
    int64_t BM = c[0], BN = c[1], BK = c[2], WM = c[3], WN = c[4];
    int64_t bytes_needed = threadgroup_bytes(BM, BN, BK, dtype_bytes);
    if (bytes_needed > 32 * 1024) continue;
    if (BK % 8 != 0 || BM % (8 * WM) != 0 || BN % (8 * WN) != 0) continue;
    int64_t TM = BM / (8 * WM);
    int64_t TN = BN / (8 * WN);
    if (TM * TN > 16) continue;
    if (WM * WN > 16) continue;
    int64_t tiles_m = ceil_div(M, BM);
    int64_t tiles_n = ceil_div(N, BN);
    int64_t total_tiles = tiles_m * tiles_n;
    if (total_tiles < 4 && BM * BN > 64 * 64) continue;
    double waste =
        static_cast<double>(tiles_m * BM * tiles_n * BN) /
        static_cast<double>(imax(1, M * N));
    int64_t ops = M * N * K;
    // score tuples: large -> (BM*BN, -waste, BK); small -> (-|BM*BN - 16*mx|, -waste)
    double score[3];
    bool this_large = ops > 256LL * 1024 * 1024;  // 268435456
    if (this_large) {
      score[0] = static_cast<double>(BM * BN);
      score[1] = -waste;
      score[2] = static_cast<double>(BK);
    } else {
      score[0] =
          -static_cast<double>(std::llabs(BM * BN - imax(M, N) * 16));
      score[1] = -waste;
      score[2] = 0.0;
    }
    // Lexicographic strict > on the active score shape (keeps earlier on ties).
    bool better;
    if (!have_best) {
      better = true;
      large_ops = this_large;
    } else {
      // ops depends only on M,N,K so this_large == large_ops always here.
      int ncomp = large_ops ? 3 : 2;
      better = false;
      for (int i = 0; i < ncomp; ++i) {
        if (score[i] > best_score[i]) {
          better = true;
          break;
        }
        if (score[i] < best_score[i]) {
          better = false;
          break;
        }
      }
    }
    if (better) {
      have_best = true;
      best_score[0] = score[0];
      best_score[1] = score[1];
      best_score[2] = score[2];
      best_tile = {BM, BN, BK, WM, WN};
    }
  }
  if (!have_best) return {16, 16, 16, 1, 1};
  return best_tile;
}

// ===========================================================================
// _pick_m5_tensor_tile (dispatch.py:620-705)
// ===========================================================================
std::array<int64_t, 3> pick_m5_tensor_tile(int64_t M, int64_t N, int64_t K,
                                           Dtype dtype) {
  int64_t mx = imax(M, N);
  bool lp = is_lp(dtype);
  bool m_div_32 = (M % 32 == 0);
  bool m_div_64 = (M % 64 == 0);
  bool n_div_64 = (N % 64 == 0);
  bool n_div_128 = (N % 128 == 0);

  if (M == 1 || N == 1) return {16, 128, 4};
  if (K <= 256 && M >= 1024 && N >= 1024 && m_div_32 && n_div_128)
    return {32, 128, 4};
  if (mx <= 256) return {32, 32, 4};
  if (M <= 48 && N >= 1024) return (M <= 16) ? std::array<int64_t, 3>{16, 64, 2}
                                             : std::array<int64_t, 3>{32, 64, 2};
  if (mx <= 1024) {
    int64_t n64 = ceil_div(M, 64) * ceil_div(N, 64);
    if (n64 < 120 || (K <= mx && m_div_64 && n_div_64)) return {32, 64, 2};
    // else: fall through to the branches below (no default return here).
  }
  if (K >= 2 * mx && mx >= 1792 && m_div_64 && n_div_128) {
    if (lp) {
      if (N < M && mx <= 2048) return {64, 64, 2};
      return {48, 128, 4};
    }
    return {64, 128, 4};
  }
  if (dtype == Dtype::kBfloat16 && mx >= 4096 && imin(M, N) >= 1024 &&
      !(m_div_64 && n_div_64))
    return {128, 128, 8};
  if (lp) return {64, 64, 2};
  if (m_div_64 && n_div_128) return {64, 128, 4};
  return {64, 64, 2};
}

// ===========================================================================
// _pick_m5_tile (dispatch.py:708-777). dbuf returned as 0/1 in slot 5.
// ===========================================================================
std::array<int64_t, 6> pick_m5_tile(int64_t M, int64_t N, int64_t K,
                                    Dtype dtype) {
  bool lp = is_lp(dtype);
  int64_t dtype_bytes = lp ? 2 : 4;
  double ops = static_cast<double>(M) * static_cast<double>(N) *
               static_cast<double>(K);
  double aspect =
      static_cast<double>(imax(M, N)) / static_cast<double>(imax(1, imin(M, N)));

  // X**3 thresholds (as doubles, matching the float `ops` comparisons).
  const double k128 = 128.0 * 128.0 * 128.0;     // 2097152
  const double k256 = 256.0 * 256.0 * 256.0;     // 16777216
  const double k768 = 768.0 * 768.0 * 768.0;     // 452984832
  const double k1536 = 1536.0 * 1536.0 * 1536.0; // 3623878656
  const double k3072 = 3072.0 * 3072.0 * 3072.0; // 28991029248

  std::array<int64_t, 6> primary;
  if (lp) {
    if (ops <= k128)
      primary = {32, 32, 16, 1, 1, 0};
    else if (ops <= k256)
      primary = {64, 64, 32, 2, 2, 0};
    else if (K >= 2 * imax(M, N))
      primary = {64, 128, 64, 2, 4, 0};
    else
      primary = {128, 64, 64, 4, 2, 0};
  } else {
    if (ops <= k256)
      primary = {32, 32, 16, 1, 1, 0};
    else if (ops <= k768)
      primary = {128, 128, 16, 4, 4, 0};
    else if (ops <= k1536)
      primary = {128, 64, 32, 4, 2, 0};
    else if (ops <= k3072)
      primary = {64, 64, 16, 2, 2, 0};
    else
      primary = {64, 128, 32, 2, 4, 0};
  }

  if (aspect >= 4.0 && imax(M, N) >= 1024) {
    if (M > N)
      primary = {128, 64, 32, 4, 2, 0};
    else
      primary = {64, 128, 32, 2, 4, 0};
  }

  const std::array<int64_t, 6> fallbacks[] = {
      primary,
      {128, 64, 32, 4, 2, 1},
      {64, 128, 32, 2, 4, 1},
      {128, 64, 64, 4, 2, 0},
      {64, 64, 32, 2, 2, 0},
      {64, 64, 16, 2, 2, 0},
      {32, 64, 16, 1, 2, 0},
      {32, 32, 16, 1, 1, 0},
      {16, 32, 16, 1, 1, 0},
  };
  for (const auto& cand : fallbacks) {
    int64_t BM = cand[0], BN = cand[1], BK = cand[2], WM = cand[3],
            WN = cand[4], dbuf = cand[5];
    if (BM % (16 * WM) != 0) continue;
    if (BN % (32 * WN) != 0) continue;
    if (BK % 16 != 0) continue;
    int64_t mult = dbuf ? 2 : 1;
    if (threadgroup_bytes(BM, BN, BK, dtype_bytes) * mult > 32 * 1024) continue;
    int64_t TM = BM / (16 * WM);
    int64_t TN = BN / (32 * WN);
    if (TM * TN > 8) continue;
    if (WM * WN > 16) continue;
    int64_t tiles_m = ceil_div(M, BM);
    int64_t tiles_n = ceil_div(N, BN);
    if (tiles_m * tiles_n == 0) continue;
    if (BM > 2 * M || BN > 2 * N) continue;
    return {BM, BN, BK, WM, WN, dbuf};
  }
  return {16, 32, 16, 1, 1, 0};
}

// ===========================================================================
// _pick_int_tile (dispatch.py:942-954)
// ===========================================================================
namespace {
int64_t int_bytes(Dtype dt) {
  switch (dt) {
    case Dtype::kInt8: return 1;
    case Dtype::kUint8: return 1;
    case Dtype::kInt16: return 2;
    case Dtype::kInt32: return 4;
    case Dtype::kInt64: return 8;
    default: std::abort();
  }
}
}  // namespace

std::array<int64_t, 5> pick_int_tile(int64_t M, int64_t N, int64_t K,
                                     Dtype dtype) {
  int64_t nbytes = int_bytes(dtype);
  int64_t mx = imax(M, N);
  if (mx <= 256) return {64, 64, 16, 16, 16};
  if (M <= 16 && N >= 1024) return {16, 64, 16, 16, 16};
  if (M <= 128 && N >= 1024) return {32, 64, 16, 8, 16};
  if (nbytes == 8) return {64, 64, 8, 16, 16};
  if (nbytes == 1 && M >= 512) return {128, 64, 16, 16, 16};
  return {64, 64, 16, 16, 16};
}

// ===========================================================================
// _with_primary (dispatch.py:424-429): primary first, append `extra` skipping
// any 3-tuple already present (element-wise equality).
// ===========================================================================
namespace {
std::vector<std::array<int64_t, 3>> with_primary(
    const std::array<int64_t, 3>& primary,
    const std::vector<std::array<int64_t, 3>>& extra) {
  std::vector<std::array<int64_t, 3>> cands;
  cands.push_back(primary);
  for (const auto& t : extra) {
    bool present = false;
    for (const auto& c : cands) {
      if (c[0] == t[0] && c[1] == t[1] && c[2] == t[2]) {
        present = true;
        break;
      }
    }
    if (!present) cands.push_back(t);
  }
  return cands;
}
}  // namespace

// ===========================================================================
// _m5_tensor_tile_candidates (dispatch.py:371-421)
// ===========================================================================
M5TensorCandidates m5_tensor_tile_candidates(int64_t M, int64_t N, int64_t K,
                                              Dtype dtype) {
  std::array<int64_t, 3> primary = pick_m5_tensor_tile(M, N, K, dtype);
  M5TensorCandidates out;

  if (dtype == Dtype::kFloat32) {
    out.candidates = {primary};
    out.margin = kAutotuneMargin;
    return out;
  }
  int64_t mx = imax(M, N);
  int64_t mn = imin(M, N);

  if (M == 1 || N == 1) {
    out.candidates = {primary};
    out.margin = kAutotuneMargin;
    return out;
  }
  if (K <= 256 && M >= 1024 && N >= 1024) {
    out.candidates = {primary};
    out.margin = kAutotuneMargin;
    return out;
  }
  if (mx <= 256) {
    out.candidates = {primary};
    out.margin = kAutotuneMargin;
    return out;
  }

  if (mn <= 256 && mx >= 1024) {
    std::vector<std::array<int64_t, 3>> extra = {
        {128, 32, 2}, {256, 32, 4}, {32, 128, 2}, {32, 256, 4}, {64, 64, 2},
        {64, 32, 2},  {32, 64, 2}};
    if (mn <= 48) {
      std::vector<std::array<int64_t, 3>> head = {
          {16, 64, 2}, {32, 64, 2}, {32, 128, 4}};
      head.insert(head.end(), extra.begin(), extra.end());
      extra = head;
    }
    out.candidates = with_primary(primary, extra);
    out.margin = kTallNarrowMargin;
    return out;
  }

  std::vector<std::array<int64_t, 3>> extra;
  if (mx <= 1024) {
    if (K >= 8 * mx) {
      extra = {{128, 32, 4}, {128, 32, 2}, {192, 32, 2}, {32, 64, 2},
               {64, 64, 2}};
      out.candidates = with_primary(primary, extra);
      out.margin = kTallNarrowMargin;
      return out;
    }
    extra = {{16, 64, 2}, {32, 64, 2}, {64, 64, 2}};
  } else {
    extra = {{64, 64, 2}, {48, 128, 4}, {64, 128, 4}, {128, 64, 4}};
    if (mx >= 2048 && !(M % 64 == 0 && N % 64 == 0)) {
      extra.push_back({128, 128, 8});
    }
  }

  out.candidates = with_primary(primary, extra);
  out.margin = kAutotuneMargin;
  return out;
}

// ===========================================================================
// _is_splitk_regime / _splitk_specs (dispatch.py:306-322)
// ===========================================================================
bool is_splitk_regime(int64_t M, int64_t N, int64_t K, Dtype dtype) {
  return (dtype != Dtype::kFloat32 && K >= 2048 && 64 <= imin(M, N) &&
          M * N <= 1500000 &&
          (imin(M, N) <= 256 || K >= 8 * imax(M, N)));
}

std::vector<std::array<int64_t, 4>> splitk_specs(int64_t M, int64_t N,
                                                 int64_t K) {
  (void)M;
  (void)N;
  std::vector<std::array<int64_t, 4>> specs;
  static const int64_t kTiles[][3] = {{128, 32, 2}, {64, 64, 2}, {32, 64, 2}};
  for (const auto& t : kTiles) {
    for (int64_t G : {int64_t(2), int64_t(4)}) {
      if (K % G == 0 && (K / G) % 16 == 0) {
        specs.push_back({t[0], t[1], t[2], G});
      }
    }
  }
  return specs;
}

// ===========================================================================
// _is_conv_regime / _conv_specs (dispatch.py:353-368)
// ===========================================================================
bool is_conv_regime(int64_t M, int64_t N, int64_t K, Dtype dtype) {
  return (dtype != Dtype::kFloat32 && N <= 64 && N % 32 == 0 && M >= 512 &&
          K >= 256);
}

std::vector<std::array<int64_t, 3>> conv_specs(int64_t M, int64_t N,
                                               int64_t K) {
  (void)K;
  std::vector<std::array<int64_t, 3>> specs;
  for (int64_t BMW : {int64_t(16), int64_t(32), int64_t(48), int64_t(64),
                      int64_t(96), int64_t(128)}) {
    if (M % BMW == 0) {
      for (int64_t NSG : {int64_t(2), int64_t(4)}) {
        specs.push_back({BMW, N, NSG});
      }
    }
  }
  return specs;
}

// ===========================================================================
// gemv_bt regime + specs (dispatch.py:373, 405-455). The batched thin-M GEMV
// autotuner candidate family. NEVER the deterministic decision -- the
// transposed-thin-M path resolves to the mpp_tensor tile when autotuning is off
// (decide() routes M>=2,N>=32,K>=64 there). Ported for completeness, mirroring
// the splitk/conv spec families above. (The matmul()-internal _unit_lead /
// _thin_trans_layout stride helpers are not ported: they take tensor strides,
// are not pure int pickers, and their routing effect is captured by decide().)
// ===========================================================================
constexpr int64_t kGemvBtMaxM = 16;       // _GEMV_BT_MAX_M
constexpr int64_t kGemvBtTgBudget = 192;  // _GEMV_BT_TG_BUDGET

bool is_gemv_bt_regime(int64_t M, int64_t N, int64_t K, Dtype dtype) {
  return (dtype != Dtype::kFloat32 && 2 <= M && M <= kGemvBtMaxM && 16 <= N &&
          N <= 8192 && K >= 64);
}

// gemv_bt TRANS_B deterministic regime (PORT POLICY #2 -- see
// decide_gemv_bt_trans below): transposed thin-M decode batches. N and K
// bounds are measured (2026-06-10 M4 Max sweep): N > 4096 is a wash vs the
// thin-M mpp_tensor tile, K > 8192 a loss (X re-read scales with K).
bool is_gemv_bt_trans_regime(int64_t M, int64_t N, int64_t K, Dtype dtype) {
  return (dtype != Dtype::kFloat32 && 2 <= M && M <= kGemvBtMaxM && 16 <= N &&
          N <= 4096 && 64 <= K && K <= 8192);
}

int64_t largest_pow2_le(int64_t x) {
  if (x < 1) return 1;
  int64_t p = 1;
  while ((p << 1) <= x) p <<= 1;
  return p;
}

std::vector<std::array<int64_t, 3>> gemv_bt_specs(int64_t M, int64_t N,
                                                  int64_t K, Dtype dtype,
                                                  int64_t align, bool trans_b) {
  (void)dtype;
  std::vector<std::array<int64_t, 3>> specs;
  if (trans_b) {
    // VEC vectorizes K (clamped to align); NCOLS>1 blocks output cols to reuse X.
    int64_t v = (K >= 2048) ? 8 : ((K >= 512) ? 4 : 2);
    while (v > 1 && (align % v)) v >>= 1;
    std::vector<int64_t> ncol_opts = {1};
    if (M >= 6) {
      for (int64_t nc : {int64_t(2), int64_t(4)})
        if (M * nc <= 48) ncol_opts.push_back(nc);
    }
    for (int64_t nc : ncol_opts) {
      specs.push_back({v, 8, nc});
      if (nc == 1) specs.push_back({v, 4, nc});  // nw in (8, 4) for ncols==1
    }
    return specs;
  }
  // Row-major: VEC by N (capped so M*VEC<=32 regs), NWARPS maxes the K-split.
  int64_t nat = (N >= 4096) ? 4 : ((N >= 256) ? 2 : 1);
  std::vector<int64_t> vecs;
  for (int64_t v0 : {nat, int64_t(1)}) {
    int64_t v = v0;
    while (v > 1 && (align % v)) v >>= 1;     // VecT load needs VEC | (ldb|off)
    while (v > 1 && M * v > 32) v >>= 1;       // cap accumulator registers
    bool present = false;
    for (int64_t e : vecs)
      if (e == v) {
        present = true;
        break;
      }
    if (!present) vecs.push_back(v);
  }
  std::vector<std::array<int64_t, 2>> seen;
  for (size_t vi = 0; vi < vecs.size(); ++vi) {
    int64_t v = vecs[vi];
    int64_t cap = imin(imin(int64_t(32), kGemvBtTgBudget / (M * v)),
                       imax(int64_t(1), (K + 31) / 32));
    std::vector<int64_t> wants;
    wants.push_back(cap);
    if (vi == 0) wants.push_back(imin(cap, int64_t(8)));
    for (int64_t want : wants) {
      int64_t nw = imax(int64_t(1), imin(cap, largest_pow2_le(want)));
      bool present = false;
      for (auto& s : seen)
        if (s[0] == v && s[1] == nw) {
          present = true;
          break;
        }
      if (!present) {
        seen.push_back({v, nw});
        specs.push_back({v, nw, 1});  // ncols unused (row-major) -> 1
      }
    }
  }
  return specs;
}

// ===========================================================================
// _int_clamp_vec (dispatch.py:934-939)
// ===========================================================================
int64_t int_clamp_vec(int64_t vec, int64_t ld, int64_t off) {
  int64_t a = ld | off;
  while (vec > 1 && (a % vec)) vec >>= 1;
  return vec;
}

// ===========================================================================
// _gemv_nt_pick (dispatch.py:109-123). Returns the chosen VEC key (1/2/4).
// nt_dict has VEC>1 keys iff (!HAS_TENSOR_UNIT || dtype != fp32).
// ===========================================================================
int64_t gemv_nt_pick_key(int64_t k, int64_t ld, int64_t off, Dtype dtype) {
  bool has_vec_gt1 = (!kHasTensorUnit) || (dtype != Dtype::kFloat32);
  if (!has_vec_gt1) return 1;  // "4 not in nt_dict"
  int64_t align = ld | off;
  if (kHasTensorUnit) {
    if ((align & 3) == 0 && k >= 512) return 4;
    if ((align & 1) == 0 && k >= 512) return 2;
    return 1;
  }
  if ((align & 3) == 0 && k >= 64) return 4;
  if ((align & 1) == 0 && k >= 32) return 2;
  return 1;
}

// ===========================================================================
// _gemv_pick (dispatch.py:126-183). Returns (vec, nwarps). The dumper passes
// `ldb_align` already == (ld | offset) so the clamp here uses that value.
// ===========================================================================
GemvPick gemv_pick(int64_t cols, int64_t ldb_align, Dtype dtype, bool vec_ok,
                   bool k_present, int64_t k) {
  bool k_big = true;  // default
  if (k_present) k_big = (k >= 2048);

  if (dtype == Dtype::kFloat32) {
    int64_t ng = ceil_div(cols, 32);
    if (!kHasTensorUnit) {
      if (ng >= 128) return {1, 16};
      return {1, 32};
    }
    if (16 < ng && ng <= 32) return {1, 16};
    return {1, 32};
  }

  int64_t vec, nw;
  if (!kHasTensorUnit) {
    if (vec_ok && cols > 12288) {
      vec = 8;
      nw = 8;
    } else if (vec_ok && cols >= 2560) {
      vec = 2;
      nw = 32;
    } else {
      vec = 1;
      nw = 32;
    }
  } else if (vec_ok && cols >= 4096 &&
             (k_big || (k_present && k >= 1024))) {
    vec = 8;
    nw = (k_present && k >= 8192) ? 16 : 8;
  } else if (vec_ok && cols >= 2560 && (k_big || (k_present && k >= 1024))) {
    vec = 4;
    nw = 8;
  } else if (vec_ok && k_big && cols >= 1280) {
    vec = 4;
    nw = 16;
  } else if (vec_ok && cols >= 1024) {
    vec = 2;
    nw = 16;
  } else if (vec_ok && cols > 512) {
    vec = 2;
    nw = 32;
  } else {
    vec = 1;
    nw = 32;
  }

  // Clamp VEC to the column-stride alignment (ldb_align == ld|offset).
  int64_t ldb = ldb_align;
  if (vec == 8 && (ldb & 7)) {
    if (!(ldb & 3)) {
      vec = 4;
      nw = 8;
    } else if (!(ldb & 1)) {
      vec = 2;
      nw = 32;
    } else {
      vec = 1;
      nw = 32;
    }
  } else if (vec == 4 && (ldb & 3)) {
    if (!(ldb & 1)) {
      vec = 2;
      nw = 32;
    } else {
      vec = 1;
      nw = 32;
    }
  } else if (vec == 2 && (ldb & 1)) {
    vec = 1;
    nw = 32;
  }
  return {vec, nw};
}

// ===========================================================================
// Per-path decision builders. Mirror gen_golden.py's _decide_* exactly.
// ===========================================================================
namespace {

Decision blank_decision(const std::string& path, const Inputs& in) {
  Decision d;
  d.path = path;
  d.tile = {};
  d.vec = 0;
  d.nwarps = 0;
  d.threads = {0, 0, 0};
  d.group = {0, 0, 0};
  d.swizzle_log = 0;
  d.trans_a = in.trans_a;
  d.trans_b = in.trans_b;
  d.lda = in.lda;
  d.ldb = in.ldb;
  d.candidates = {};
  d.margin = kAutotuneMargin;
  return d;
}

// --- float GEMV (_decide_gemv_float, gen_golden.py:127-186) ---
Decision decide_gemv_float(const Inputs& in) {
  int64_t M = in.m, N = in.n, K = in.k;
  if (M == 1) {
    if (in.trans_b) {
      Decision d = blank_decision("gemv_nt", in);
      int64_t fn_key = gemv_nt_pick_key(K, in.ldb, in.offset, in.dtype);
      int64_t n_groups = ceil_div(N, 4);
      d.vec = fn_key;
      d.nwarps = 4;
      d.threads = {128 * n_groups, 1, 1};
      d.group = {128, 1, 1};
      return d;
    } else {
      Decision d = blank_decision("gemv_t", in);
      int64_t align = in.ldb | in.offset;
      GemvPick p =
          gemv_pick(N, align, in.dtype, /*vec_ok=*/true, /*k_present=*/true, K);
      int64_t tg = p.nwarps * 32;
      int64_t ng = ceil_div(N, 32 * p.vec);
      d.vec = p.vec;
      d.nwarps = p.nwarps;  // == tg/32
      d.threads = {tg * ng, 1, 1};
      d.group = {tg, 1, 1};
      return d;
    }
  } else {  // N == 1
    if (in.trans_a) {
      Decision d = blank_decision("gemv_t", in);
      int64_t align = in.lda | in.offset;
      GemvPick p =
          gemv_pick(M, align, in.dtype, /*vec_ok=*/true, /*k_present=*/true, K);
      int64_t tg = p.nwarps * 32;
      int64_t ng = ceil_div(M, 32 * p.vec);
      d.vec = p.vec;
      d.nwarps = p.nwarps;
      d.threads = {tg * ng, 1, 1};
      d.group = {tg, 1, 1};
      return d;
    } else {
      Decision d = blank_decision("gemv_nt", in);
      int64_t fn_key = gemv_nt_pick_key(K, in.lda, in.offset, in.dtype);
      int64_t n_groups = ceil_div(M, 4);
      d.vec = fn_key;
      d.nwarps = 4;
      d.threads = {128 * n_groups, 1, 1};
      d.group = {128, 1, 1};
      return d;
    }
  }
}

// --- gemv_bt: deterministic thin-M low-precision GEMV (_decide_gemv_bt). PORT
// POLICY: gemv_bt is an autotuner-only candidate upstream; the port routes every
// is_gemv_bt_regime shape to it (heuristic spec[0]) so the autotuner-free consumer
// (XLA) gets the thin-M GEMV win. Diverges from metalBLAS's autotune-off baseline
// by design; (vec, nwarps) + launch dims still come from the real pure picker. ---
Decision decide_gemv_bt(const Inputs& in) {
  int64_t M = in.m, N = in.n, K = in.k;
  Decision d = blank_decision("gemv_bt", in);
  int64_t align = in.ldb | in.offset;  // B row stride | base offset (VEC clamp)
  std::vector<std::array<int64_t, 3>> specs =
      gemv_bt_specs(M, N, K, in.dtype, align, /*trans_b=*/false);
  int64_t vec = specs[0][0], nwarps = specs[0][1];  // row-major spec[0] = (vec,nwarps)
  int64_t block_n = 32 * vec;
  int64_t ng = ceil_div(N, block_n);
  d.vec = vec;
  d.nwarps = nwarps;
  d.ncols = 1;  // row-major gemv_bt has no col blocking
  d.threads = {nwarps * 32 * ng, 1, 1};
  d.group = {nwarps * 32, 1, 1};
  return d;
}

// --- gemv_bt TRANS_B: transposed thin-M decode batches (PORT POLICY #2,
// _decide_gemv_bt_trans / _is_gemv_bt_trans_regime). x[M,K] @ W[N,K]^T -- the
// batched-decode projections an LLM serving stack emits at batch_size<=16.
// Upstream only reaches gemv_bt(trans) through _autotune_trans; autotune off
// keeps these shapes on the mpp_tensor trans tile, whose tiles_n = N/BN <= 64
// threadgroup grid is starved on a 40-core GPU (82-228us measured vs ~420 GB/s
// ideal on the same weights). Deterministic heuristic from the 2026-06-10
// M4 Max sweep (CPU-oracle verified): VEC from the real _gemv_bt_specs trans_b
// vec rule (align-clamped), NWARPS=4, NCOLS = 1 for M<6 (upstream's register
// guard) else 2 for N<=2048 / 4 above. Regime bounds are data: at N > 4096 the
// thin-M tile's grid already fills the GPU (wash), at K > 8192 gemv_bt's X
// re-read (scales with K) drops below the {16,32,2} tile. The regime gate
// is_gemv_bt_trans_regime lives with the other public regime gates above. ---
Decision decide_gemv_bt_trans(const Inputs& in) {
  int64_t N = in.n, K = in.k;
  Decision d = blank_decision("gemv_bt", in);
  int64_t align = in.ldb | in.offset;  // W col stride (=K) | base offset
  int64_t vec = (K >= 2048) ? 8 : ((K >= 512) ? 4 : 2);
  while (vec > 1 && (align % vec)) vec >>= 1;
  const int64_t nwarps = 4;
  int64_t ncols = (in.m < 6) ? 1 : ((N <= 2048) ? 2 : 4);
  int64_t span = nwarps * ncols;  // columns per threadgroup
  int64_t ng = ceil_div(N, span);
  d.vec = vec;
  d.nwarps = nwarps;
  d.ncols = ncols;
  d.threads = {nwarps * 32 * ng, 1, 1};
  d.group = {nwarps * 32, 1, 1};
  return d;
}

// --- float GEMM m5_tensor (_decide_gemm_float, gen_golden.py:209-230) ---
Decision decide_gemm_float(const Inputs& in) {
  int64_t M = in.m, N = in.n, K = in.k;
  Decision d = blank_decision("gemm", in);
  std::array<int64_t, 3> t = pick_m5_tensor_tile(M, N, K, in.dtype);
  // PORT POLICY #3: thin-M decode tile. At M <= 16 the default BM=64 tile
  // wastes 48/64 rows and launches tiles_n = N/64 threadgroups (starved);
  // {16,32,2} doubles tiles_n and stops the row waste. Measured (llmd
  // 2026-06-11): gate/up [16,3072]x8192 143->133us, lm_head 1903->1853us.
  // Applied HERE, not in pick_m5_tensor_tile, so the candidates record below
  // keeps the real picker's primary (autotuner parity).
  if (2 <= M && M <= 16) t = {16, 32, 2};
  int64_t BM = t[0], BN = t[1], NSG = t[2];
  M5TensorCandidates cm = m5_tensor_tile_candidates(M, N, K, in.dtype);
  d.tile = {BM, BN, NSG};
  d.margin = cm.margin;
  d.candidates = cm.candidates;
  int64_t swz = 0;  // FIXED at 0 for m5_tensor (dispatch.py:1191)
  d.swizzle_log = swz;
  int64_t tiles_m = ceil_div(M, BM);
  int64_t tiles_n = ceil_div(N, BN);
  int64_t group0 = NSG * 32;
  int64_t tile_factor = (int64_t)1 << swz;
  int64_t tn_swz = tiles_n * tile_factor;
  int64_t tm_swz = ceil_div(tiles_m, tile_factor);
  d.group = {group0, 1, 1};
  d.threads = {group0 * tn_swz, tm_swz, 1};
  return d;
}

// --- float GEMM simd fallback (_decide_simd_float, gen_golden.py:389-406).
// Only reachable when HAS_METAL4 is false; modeled for completeness. ---
Decision decide_simd_float(const Inputs& in) {
  int64_t M = in.m, N = in.n, K = in.k;
  Decision d = blank_decision("gemm", in);
  std::array<int64_t, 5> t = pick_simd_tile(M, N, K, in.dtype);
  int64_t BM = t[0], BN = t[1], BK = t[2], WM = t[3], WN = t[4];
  d.tile = {BM, BN, BK, WM, WN};
  int64_t tiles_m = ceil_div(M, BM);
  int64_t tiles_n = ceil_div(N, BN);
  int64_t swz = round_swizzle_log(tiles_m, tiles_n);
  d.swizzle_log = swz;
  int64_t g0 = WM * WN * 32;
  int64_t tile_factor = (int64_t)1 << swz;
  int64_t tn_swz = tiles_n * tile_factor;
  int64_t tm_swz = ceil_div(tiles_m, tile_factor);
  d.group = {g0, 1, 1};
  d.threads = {g0 * tn_swz, tm_swz, 1};
  return d;
}

// --- float GEMM m5 manual (_decide_m5_float, gen_golden.py:409-426) ---
Decision decide_m5_float(const Inputs& in) {
  int64_t M = in.m, N = in.n, K = in.k;
  Decision d = blank_decision("gemm", in);
  std::array<int64_t, 6> t = pick_m5_tile(M, N, K, in.dtype);
  int64_t BM = t[0], BN = t[1], BK = t[2], WM = t[3], WN = t[4], dbuf = t[5];
  d.tile = {BM, BN, BK, WM, WN, dbuf};
  int64_t tiles_m = ceil_div(M, BM);
  int64_t tiles_n = ceil_div(N, BN);
  int64_t swz = round_swizzle_log(tiles_m, tiles_n);
  d.swizzle_log = swz;
  int64_t g0 = WM * WN * 32;
  int64_t tile_factor = (int64_t)1 << swz;
  int64_t tn_swz = tiles_n * tile_factor;
  int64_t tm_swz = ceil_div(tiles_m, tile_factor);
  d.group = {g0, 1, 1};
  d.threads = {g0 * tn_swz, tm_swz, 1};
  return d;
}

// --- complex (_decide_complex, gen_golden.py:242-277) ---
constexpr int64_t kCgemvTNwarps = 8;
constexpr int64_t kCgemvNtNwarps = 4;

Decision decide_complex(const Inputs& in) {
  int64_t M = in.m, N = in.n, K = in.k;
  Decision d = blank_decision("complex", in);
  // cgemv_t: M==1, B contiguous (ldb==N), N>=1, no transpose.
  if (M == 1 && N >= 1 && !in.trans_a && !in.trans_b && in.ldb == N) {
    d.nwarps = kCgemvTNwarps;
    int64_t ng = ceil_div(N, 32);
    int64_t g0 = kCgemvTNwarps * 32;
    d.group = {g0, 1, 1};
    d.threads = {g0 * ng, 1, 1};
    d.tile = {};
    return d;
  }
  // cgemv_nt: N==1, A contiguous (lda==K), M>=1, no transpose.
  if (N == 1 && M >= 1 && !in.trans_a && !in.trans_b && in.lda == K) {
    d.nwarps = kCgemvNtNwarps;
    int64_t ng = ceil_div(M, kCgemvNtNwarps);
    int64_t g0 = kCgemvNtNwarps * 32;
    d.group = {g0, 1, 1};
    d.threads = {g0 * ng, 1, 1};
    d.tile = {};
    return d;
  }
  // 4-real-GEMM fallback: combine launch records M*N elems, group 256.
  int64_t nC = M * N;
  d.group = {256, 1, 1};
  d.threads = {nC, 1, 1};
  d.tile = {};
  return d;
}

// --- integer (_decide_int, gen_golden.py:281-342) ---
std::array<int64_t, 2> igemv_t_cfg(Dtype dt) {  // (VEC, NWARPS)
  switch (dt) {
    case Dtype::kInt8: return {8, 8};
    case Dtype::kUint8: return {8, 8};
    case Dtype::kInt16: return {4, 8};
    case Dtype::kInt32: return {2, 8};
    case Dtype::kInt64: return {1, 8};
    default: std::abort();
  }
}
std::array<int64_t, 2> igemv_nt_cfg(Dtype dt) {  // (VEC, NWARPS)
  switch (dt) {
    case Dtype::kInt8: return {8, 4};
    case Dtype::kUint8: return {8, 4};
    case Dtype::kInt16: return {4, 4};
    case Dtype::kInt32: return {4, 8};
    case Dtype::kInt64: return {2, 4};
    default: std::abort();
  }
}

Decision decide_int(const Inputs& in) {
  int64_t M = in.m, N = in.n, K = in.k;
  Decision d = blank_decision("int", in);

  if (M == 1 && N >= 16) {
    if (!in.trans_b) {
      d.path = "gemv_t";
      auto cfg = igemv_t_cfg(in.dtype);
      int64_t vec = int_clamp_vec(cfg[0], in.ldb, in.offset);
      int64_t nw = cfg[1];
      int64_t ng = ceil_div(N, 32 * vec);
      d.vec = vec;
      d.nwarps = nw;
      d.threads = {nw * 32 * ng, 1, 1};
      d.group = {nw * 32, 1, 1};
      return d;
    } else {
      d.path = "gemv_nt";
      auto cfg = igemv_nt_cfg(in.dtype);
      int64_t vec = int_clamp_vec(cfg[0], in.ldb, in.offset);
      int64_t nw = cfg[1];
      int64_t ng = ceil_div(N, nw);
      d.vec = vec;
      d.nwarps = nw;
      d.threads = {nw * 32 * ng, 1, 1};
      d.group = {nw * 32, 1, 1};
      return d;
    }
  } else if (N == 1 && M >= 16) {
    if (in.trans_a) {
      d.path = "gemv_t";
      auto cfg = igemv_t_cfg(in.dtype);
      int64_t vec = int_clamp_vec(cfg[0], in.lda, in.offset);
      int64_t nw = cfg[1];
      int64_t ng = ceil_div(M, 32 * vec);
      d.vec = vec;
      d.nwarps = nw;
      d.threads = {nw * 32 * ng, 1, 1};
      d.group = {nw * 32, 1, 1};
      return d;
    } else {
      d.path = "gemv_nt";
      auto cfg = igemv_nt_cfg(in.dtype);
      int64_t vec = int_clamp_vec(cfg[0], in.lda, in.offset);
      int64_t nw = cfg[1];
      int64_t ng = ceil_div(M, nw);
      d.vec = vec;
      d.nwarps = nw;
      d.threads = {nw * 32 * ng, 1, 1};
      d.group = {nw * 32, 1, 1};
      return d;
    }
  }

  // general int_gemm
  std::array<int64_t, 5> t = pick_int_tile(M, N, K, in.dtype);
  int64_t BM = t[0], BN = t[1], BK = t[2], TX = t[3], TY = t[4];
  d.tile = {BM, BN, BK, TX, TY};
  int64_t tiles_m = ceil_div(M, BM);
  int64_t tiles_n = ceil_div(N, BN);
  int64_t grp = TX * TY;
  d.group = {grp, 1, 1};
  d.threads = {grp * tiles_n, tiles_m, 1};
  return d;
}

bool is_complex(Dtype dt) {
  return dt == Dtype::kComplex64 || dt == Dtype::kComplex32;
}
bool is_int(Dtype dt) {
  switch (dt) {
    case Dtype::kInt8:
    case Dtype::kUint8:
    case Dtype::kInt16:
    case Dtype::kInt32:
    case Dtype::kInt64:
      return true;
    default:
      return false;
  }
}

}  // namespace

// ===========================================================================
// decide() -- mirrors matmul()'s path selection (gen_golden.py:348-386,
// dispatch.py:1020-1156). First match wins.
// ===========================================================================
Decision decide(const Inputs& in) {
  int64_t M = in.m, N = in.n, K = in.k;

  // 1. complex (dispatch.py:1037).
  if (is_complex(in.dtype)) return decide_complex(in);

  // 2. integer (dispatch.py:1041).
  if (is_int(in.dtype)) return decide_int(in);

  // 3. float backend="auto" (dispatch.py:1127-1156).
  bool lp = is_lp(in.dtype);
  bool packed_ab = (in.lda == K && in.ldb == N);

  // Wide-N M==1 padded m5_tensor fallback (dispatch.py:1137).
  if (kHasMetal4 && M == 1 && N >= 4096 && K >= 256 && lp && !in.trans_a &&
      !in.trans_b && packed_ab) {
    return decide_gemm_float(in);
  }

  // GEMV-shaped (dispatch.py:1142-1145).
  if (M == 1 && N >= 16) return decide_gemv_float(in);
  if (N == 1 && M >= 16) return decide_gemv_float(in);

  // No Metal 4 -> simd (dispatch.py:1146-1149). Not reached when kHasMetal4.
  if (!kHasMetal4) return decide_simd_float(in);

  // gemv_bt: deterministic thin-M low-precision GEMV (PORT POLICY -- see
  // decide_gemv_bt). Row-major only.
  if (!in.trans_a && !in.trans_b && is_gemv_bt_regime(M, N, K, in.dtype)) {
    return decide_gemv_bt(in);
  }

  // gemv_bt TRANS_B: transposed thin-M decode batches (PORT POLICY #2 -- see
  // decide_gemv_bt_trans). x @ W^T only; trans_a stays on mpp_tensor (the
  // kernel's column-major X loads are scalar-slow).
  if (!in.trans_a && in.trans_b &&
      is_gemv_bt_trans_regime(M, N, K, in.dtype)) {
    return decide_gemv_bt_trans(in);
  }

  // mpp_tensor (dispatch.py:1435): M>=2 & N>=32 & K>=64 -> mpp_tensor for ANY
  // transpose/packing. The kernel reads any unit-inner-stride operand through a
  // strided tensor_inline view (TRANS_A/TRANS_B), so transposed and non-NN-packed
  // shapes ride it too (the transposed-thin-M fast path, dispatch.py:1370-1383 /
  // _gemm_trans_plan, resolves to this same mpp_tensor tile + launch).
  if (M >= 2 && N >= 32 && K >= 64) {
    return decide_gemm_float(in);
  }

  // else mpp manual (dispatch.py:1438) -- only the tiny/edge shapes reach here now.
  return decide_m5_float(in);
}

}  // namespace metalblas_port
