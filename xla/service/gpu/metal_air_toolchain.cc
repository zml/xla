/* Copyright 2026 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "xla/service/gpu/metal_air_toolchain.h"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <regex>  // NOLINT(build/c++11) — textual LLVM-15 syntax compat shim
#include <string>
#include <utility>
#include <vector>

#include "absl/cleanup/cleanup.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/subprocess.h"

namespace xla {
namespace gpu {

absl::StatusOr<std::string> RunCommand(std::vector<std::string> argv,
                                       bool capture_stdout) {
  if (argv.empty()) {
    return absl::InvalidArgumentError("Cannot run an empty command.");
  }
  tsl::SubProcess process;
  process.SetProgram(argv[0], argv);
  if (capture_stdout) {
    process.SetChannelAction(tsl::CHAN_STDOUT, tsl::ACTION_PIPE);
  }
  process.SetChannelAction(tsl::CHAN_STDERR, tsl::ACTION_PIPE);
  if (!process.Start()) {
    return absl::InternalError(
        absl::StrFormat("Failed to launch %s.", argv[0]));
  }

  std::string stdout_output;
  std::string stderr_output;
  int exit_status =
      process.Communicate(/*stdin_input=*/nullptr,
                          capture_stdout ? &stdout_output : nullptr,
                          &stderr_output);
  if (exit_status != 0) {
    return absl::InternalError(absl::StrFormat(
        "Command failed with status %d: %s\nstderr:\n%s", exit_status,
        absl::StrJoin(argv, " "), stderr_output));
  }
  return capture_stdout ? stdout_output : stderr_output;
}

absl::StatusOr<std::string> FindMetalTool(const char* tool_name) {
  if (const char* toolchain = std::getenv("METAL_TOOLCHAIN")) {
    return absl::StrCat(toolchain, "/", tool_name);
  }

  return absl::NotFoundError(
      "METAL_TOOLCHAIN is not set; cannot locate Metal toolchain tools.");
}

// air-as is an ~LLVM-15-era assembler and rejects the LLVM-17+ vector-splat
// constant syntax `<N x T> splat (T V)` (emitted e.g. for a softmax/mask -inf
// reduction-init over a vectorized loop). Expand it back to the explicit
// `<N x T> <T V, T V, ...>` form the old parser accepts. This MUST be a text
// rewrite, not an IR pass: the LLVM-23 AsmWriter prints any splat vector
// constant — even an explicit ConstantVector of identical elements — with the
// `splat` keyword, so there is no IR form that prints the old way. Conservative
// pattern (scalar element constant, no nested parens); leaves everything else
// byte-for-byte.
std::string ExpandSplatConstantsForOldAirAs(absl::string_view source) {
  static const std::regex kSplat(R"(<(\d+) x ([^<>]+?)> splat \(([^()]+)\))");
  const std::string in(source);
  std::string out;
  out.reserve(in.size());
  std::size_t last = 0;
  for (auto it = std::sregex_iterator(in.begin(), in.end(), kSplat);
       it != std::sregex_iterator(); ++it) {
    const std::smatch& m = *it;
    out.append(in, last, static_cast<std::size_t>(m.position()) - last);
    const int n = std::stoi(m[1].str());
    const std::string& inner = m[3].str();
    absl::StrAppend(&out, "<", m[1].str(), " x ", m[2].str(), "> <");
    for (int i = 0; i < n; ++i) {
      if (i) out.append(", ");
      out.append(inner);
    }
    out.append(">");
    last = static_cast<std::size_t>(m.position() + m.length());
  }
  out.append(in, last, std::string::npos);
  return out;
}

// The LLVM-23 AsmWriter prints a floating-point constant in decimal (e.g.
// `4.000000e-04`, the tanh-approximation identity threshold 0.0004) whenever the
// 6-significant-digit form round-trips back to the same value; otherwise it uses
// the hex form handled below. air-as (~LLVM-15) is stricter than LLVM-23 when
// re-parsing a constant: it rejects any *decimal* literal not exactly
// representable in its type ("floating point constant invalid for type"), e.g.
// the tanh threshold 0.0004. Rewrite every decimal FP literal to the `0x` +
// 16-hex-digit form air-as wants. CRUCIAL: that hex is always the *double* bit
// pattern, and for a `float` operand air-as additionally requires it to be the
// float-exact double (the single widened to double, low mantissa bits zero) — the
// nearest double to 0.0004 (0x...EB1C432D) is rejected, only the float-widened
// double (0x...E0000000) is accepted (verified against air-as directly). So for a
// float operand we round the decimal through float before widening, exactly like
// RewriteHexFloatsForOldAirAs; for a double operand we emit the double bits as-is.
// Precision is read from the nearest preceding `float`/`double` type keyword on
// the same line (LLVM prints the type just ahead of the operand list, so this is
// also correct for fptrunc/fpext). The literal must contain a '.' or exponent so
// plain integer operands (i32 4, getelementptr indices) are never matched; it is
// anchored to an operand-position delimiter so a number embedded in an identifier,
// a hex literal (0x.../f0x...), or a quoted version string is left untouched.
std::string RewriteDecimalFloatsForOldAirAs(absl::string_view source) {
  static const std::regex kDecimalFp(
      R"(([\s,(\[<])(-?(?:\d+\.\d*(?:[eE][-+]?\d+)?|\.\d+(?:[eE][-+]?\d+)?|\d+[eE][-+]?\d+)))");
  const std::string in(source);
  std::string out;
  out.reserve(in.size());
  std::size_t last = 0;
  for (auto it = std::sregex_iterator(in.begin(), in.end(), kDecimalFp);
       it != std::sregex_iterator(); ++it) {
    const std::smatch& m = *it;
    const std::size_t pos = static_cast<std::size_t>(m.position());
    out.append(in, last, pos - last);

    // Resolve the operand's precision from the nearest preceding `float` /
    // `double` keyword on the same line.
    const std::size_t nl = in.rfind('\n', pos);
    const std::size_t line_start = (nl == std::string::npos) ? 0 : nl + 1;
    const std::string prefix = in.substr(line_start, pos - line_start);
    const std::size_t fpos = prefix.rfind("float");
    const std::size_t dpos = prefix.rfind("double");
    const bool is_double = dpos != std::string::npos &&
                           (fpos == std::string::npos || dpos > fpos);

    const double raw = std::strtod(m[2].str().c_str(), nullptr);
    // float operand -> round through single so the double is float-exact.
    const double d = is_double ? raw : static_cast<double>(static_cast<float>(raw));
    uint64_t dbits;
    std::memcpy(&dbits, &d, sizeof(dbits));
    absl::StrAppend(&out, m[1].str());
    absl::StrAppendFormat(&out, "0x%016X", dbits);
    last = pos + static_cast<std::size_t>(m.length());
  }
  out.append(in, last, std::string::npos);
  return out;
}

// The LLVM-23 AsmWriter prints a single-precision float constant as the compact
// hex literal `f0xXXXXXXXX` (the 32 float bits), which air-as (~LLVM-15) cannot
// lex ("expected value token"). Rewrite each to the LLVM-15 form `0x` + the 16
// hex digits of the double the float widens to; air-as reads that double and
// narrows it back to the original float (a lossless round-trip).
std::string RewriteHexFloatsForOldAirAs(absl::string_view source) {
  static const std::regex kFloat(R"(\bf0x([0-9A-Fa-f]{8})\b)");
  const std::string in(source);
  std::string out;
  out.reserve(in.size());
  std::size_t last = 0;
  for (auto it = std::sregex_iterator(in.begin(), in.end(), kFloat);
       it != std::sregex_iterator(); ++it) {
    const std::smatch& m = *it;
    out.append(in, last, static_cast<std::size_t>(m.position()) - last);
    const uint32_t fbits =
        static_cast<uint32_t>(std::strtoul(m[1].str().c_str(), nullptr, 16));
    float f;
    std::memcpy(&f, &fbits, sizeof(f));
    const double d = f;
    uint64_t dbits;
    std::memcpy(&dbits, &d, sizeof(dbits));
    absl::StrAppendFormat(&out, "0x%016X", dbits);
    last = static_cast<std::size_t>(m.position() + m.length());
  }
  out.append(in, last, std::string::npos);
  return out;
}

// The LLVM-23 AsmWriter prints floating-point infinity / NaN constants with the
// keyword forms `inf` / `qnan` / `snan`, optionally signed (e.g. `+qnan`,
// `-inf`; a bf16 NaN broadcast in Qwen-27B emits `bfloat +qnan`), which air-as
// (~LLVM-15) cannot lex ("expected value token"); rewrite them to the hex bit
// pattern it expects. CRUCIAL: that pattern is TYPE-DEPENDENT — a 16-bit type
// must use its own typed-hex prefix (`bfloat 0xR7FC0`, `half 0xH7E00`); the wide
// 64-bit (double) form is accepted only for `float`/`double`, which air-as
// narrows losslessly (the typed forms above are exactly what the native `metal`
// compiler emits for the same constants, verified by compiling an MSL shader).
// So resolve the operand's type from the nearest preceding float-type keyword on
// the same line — LLVM prints the type just ahead of the operand and repeats it
// inside vector constants (`<4 x bfloat> <bfloat +qnan, ...>`), and `\bfloat\b`
// distinguishes a standalone `float` from the `float` inside `bfloat` (no word
// boundary between the `b` and `f`). Anchored to an operand-position delimiter so
// it never touches an identifier that merely contains "inf"/"nan"; a token with
// no float type on its line (so not actually a float constant) is left untouched.
std::string RewriteInfNanForOldAirAs(absl::string_view source) {
  static const std::regex kInfNan(R"(([\s,(<\[])([+-]?)(qnan|snan|nan|inf)\b)");
  static const std::regex kFpType(R"(\b(bfloat|half|float|double)\b)");
  const std::string in(source);
  std::string out;
  out.reserve(in.size());
  std::size_t last = 0;
  for (auto it = std::sregex_iterator(in.begin(), in.end(), kInfNan);
       it != std::sregex_iterator(); ++it) {
    const std::smatch& m = *it;
    const std::size_t pos = static_cast<std::size_t>(m.position());

    // Resolve the operand's float type from the nearest preceding float-type
    // keyword on the same line (last match wins).
    const std::size_t nl = in.rfind('\n', pos);
    const std::size_t line_start = (nl == std::string::npos) ? 0 : nl + 1;
    const std::string prefix = in.substr(line_start, pos - line_start);
    std::string type;
    for (auto t = std::sregex_iterator(prefix.begin(), prefix.end(), kFpType);
         t != std::sregex_iterator(); ++t) {
      type = (*t)[1].str();
    }
    if (type.empty()) continue;  // not a float constant; leave it untouched.

    const bool neg = m[2].str() == "-";
    const std::string& kw = m[3].str();
    const bool is_inf = kw == "inf";
    const bool is_snan = kw == "snan";
    std::string hex;
    if (type == "bfloat") {  // 1+8+7 bits: exp all-ones, quiet bit = mantissa MSB.
      hex = is_inf ? (neg ? "0xRFF80" : "0xR7F80")
          : is_snan ? (neg ? "0xRFFA0" : "0xR7FA0")
                    : (neg ? "0xRFFC0" : "0xR7FC0");
    } else if (type == "half") {  // 1+5+10 bits.
      hex = is_inf ? (neg ? "0xHFC00" : "0xH7C00")
          : is_snan ? (neg ? "0xHFD00" : "0xH7D00")
                    : (neg ? "0xHFE00" : "0xH7E00");
    } else {  // float / double: air-as accepts (and narrows) the double bits.
      hex = is_inf ? (neg ? "0xFFF0000000000000" : "0x7FF0000000000000")
          : is_snan ? (neg ? "0xFFF4000000000000" : "0x7FF4000000000000")
                    : (neg ? "0xFFF8000000000000" : "0x7FF8000000000000");
    }
    out.append(in, last, pos - last);
    absl::StrAppend(&out, m[1].str(), hex);
    last = pos + static_cast<std::size_t>(m.length());
  }
  out.append(in, last, std::string::npos);
  return out;
}

absl::StatusOr<std::vector<uint8_t>> CompileMetalAirToMetallib(
    absl::string_view raw_source, absl::string_view temp_name) {
  const std::string source =
      RewriteInfNanForOldAirAs(RewriteHexFloatsForOldAirAs(
          RewriteDecimalFloatsForOldAirAs(
              ExpandSplatConstantsForOldAirAs(raw_source))));
  TF_ASSIGN_OR_RETURN(std::string air_as, FindMetalTool("air-as"));
  TF_ASSIGN_OR_RETURN(std::string air_opt, FindMetalTool("air-opt"));
  TF_ASSIGN_OR_RETURN(std::string metallib, FindMetalTool("metallib"));

  tsl::Env* env = tsl::Env::Default();
  std::string base;
  if (!env->LocalTempFilename(&base)) {
    return absl::InternalError(
        absl::StrFormat("Could not create Metal AIR temp filename for %s.",
                        temp_name));
  }
  std::string air_ll_path = absl::StrCat(base, ".ll");
  std::string air_path = absl::StrCat(base, ".air");
  std::string opt_air_path = absl::StrCat(base, ".opt.air");
  std::string metallib_path = absl::StrCat(base, ".metallib");
  absl::Cleanup cleanup = [&] {
    env->DeleteFile(air_ll_path).IgnoreError();
    env->DeleteFile(air_path).IgnoreError();
    env->DeleteFile(opt_air_path).IgnoreError();
    env->DeleteFile(metallib_path).IgnoreError();
  };

  TF_RETURN_IF_ERROR(tsl::WriteStringToFile(env, air_ll_path, source));
  TF_RETURN_IF_ERROR(
      RunCommand({air_as, air_ll_path, "-o", air_path}, false).status());
  TF_RETURN_IF_ERROR(
      RunCommand({air_opt, "--O3", air_path, "-o", opt_air_path}, false)
          .status());
  TF_RETURN_IF_ERROR(
      RunCommand({metallib, opt_air_path, "-o", metallib_path}, false)
          .status());

  std::string bytes;
  TF_RETURN_IF_ERROR(tsl::ReadFileToString(env, metallib_path, &bytes));
  return std::vector<uint8_t>(bytes.begin(), bytes.end());
}

absl::StatusOr<std::vector<uint8_t>> CompileMetalSourceToMetallib(
    absl::string_view source,
    const std::vector<std::pair<std::string, std::string>>& subs) {
  TF_ASSIGN_OR_RETURN(std::string metal_c, FindMetalTool("metal"));
  TF_ASSIGN_OR_RETURN(std::string metallib_tool, FindMetalTool("metallib"));
  const std::string src =
      subs.empty() ? std::string(source) : absl::StrReplaceAll(source, subs);

  tsl::Env* env = tsl::Env::Default();
  std::string base;
  if (!env->LocalTempFilename(&base)) {
    return absl::InternalError("Could not create Metal source temp filename.");
  }
  const std::string metal_path = absl::StrCat(base, ".metal");
  const std::string air_path = absl::StrCat(base, ".air");
  const std::string metallib_path = absl::StrCat(base, ".metallib");
  absl::Cleanup cleanup = [&] {
    env->DeleteFile(metal_path).IgnoreError();
    env->DeleteFile(air_path).IgnoreError();
    env->DeleteFile(metallib_path).IgnoreError();
  };
  TF_RETURN_IF_ERROR(tsl::WriteStringToFile(env, metal_path, src));
  TF_RETURN_IF_ERROR(
      RunCommand({metal_c, "-std=metal4.0", "-c", metal_path, "-o", air_path},
                 /*capture_stdout=*/false)
          .status());
  TF_RETURN_IF_ERROR(
      RunCommand({metallib_tool, air_path, "-o", metallib_path},
                 /*capture_stdout=*/false)
          .status());
  std::string bytes;
  TF_RETURN_IF_ERROR(tsl::ReadFileToString(env, metallib_path, &bytes));
  return std::vector<uint8_t>(bytes.begin(), bytes.end());
}

absl::StatusOr<std::vector<uint8_t>> CompileMetalSourceToMetallibCached(
    absl::string_view source,
    const std::vector<std::pair<std::string, std::string>>& subs) {
  std::string key =
      subs.empty() ? std::string(source) : absl::StrReplaceAll(source, subs);
  static absl::Mutex mu(absl::kConstInit);
  static auto& cache =
      *new absl::flat_hash_map<std::string, std::vector<uint8_t>>();
  {
    absl::MutexLock lock(&mu);
    auto it = cache.find(key);
    if (it != cache.end()) return it->second;
  }
  TF_ASSIGN_OR_RETURN(std::vector<uint8_t> lib,
                      CompileMetalSourceToMetallib(key));
  absl::MutexLock lock(&mu);
  return cache.emplace(std::move(key), std::move(lib)).first->second;
}

}  // namespace gpu
}  // namespace xla
