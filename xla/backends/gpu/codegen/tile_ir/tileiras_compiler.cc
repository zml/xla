#include "xla/backends/gpu/codegen/tile_ir/tileiras_compiler.h"

#include <atomic>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <thread>
#include <variant>
#include <vector>

#include "absl/base/no_destructor.h"
#include "absl/cleanup/cleanup.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "absl/synchronization/notification.h"
#include "absl/time/time.h"

#include "absl/hash/hash.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "xla/stream_executor/cuda/subprocess_compilation.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/subprocess.h"
#include "xla/util.h"
#include "xla/xla.pb.h"

namespace xla::gpu::tile_ir {

absl::StatusOr<std::string> FindTileIrAssembler(
    absl::string_view preferred_cuda_dir) {
  return stream_executor::FindCudaExecutable("tileiras", preferred_cuda_dir);
}

namespace {

// Memoize tileiras by bytecode: same kernel appears once per HLO module.
absl::Mutex& CubinCacheMutex() {
  static absl::NoDestructor<absl::Mutex> mutex;
  return *mutex;
}

absl::flat_hash_map<std::string, std::vector<uint8_t>>& CubinCache() {
  static absl::NoDestructor<
      absl::flat_hash_map<std::string, std::vector<uint8_t>>>
      cache;
  return *cache;
}

constexpr absl::string_view kCompilerTimeoutEnvVar =
    "XLA_TILE_IR_COMPILER_TIMEOUT_SEC";

// Timeout for tileiras (cf. CUDA_TILE_COMPILER_TIMEOUT_SEC).
absl::Duration TileIrCompilerTimeout() {
  static const absl::Duration timeout = [] {
    const char* env = std::getenv(kCompilerTimeoutEnvVar.data());
    double seconds = 0;
    if (env != nullptr && absl::SimpleAtod(env, &seconds) && seconds > 0) {
      return absl::Seconds(seconds);
    }
    return absl::Minutes(2);
  }();
  return timeout;
}

}  // namespace

absl::StatusOr<std::vector<uint8_t>> CompileTileIrBytecode(
    absl::string_view bytecode, const se::DeviceDescription& device_info,
    const DebugOptions& debug_options) {
  const se::CudaComputeCapability* cc =
      device_info.gpu_compute_capability().cuda_compute_capability();
  if (cc == nullptr) {
    return absl::UnimplementedError(
        "CUDA Tile IR codegen is only available on NVIDIA GPUs.");
  }

  std::string cache_key =
      absl::StrCat("sm_", cc->major, cc->minor, "\0", bytecode);
  {
    absl::MutexLock lock(&CubinCacheMutex());
    auto it = CubinCache().find(cache_key);
    if (it != CubinCache().end()) return it->second;
  }

  absl::StatusOr<std::string> tileiras_path =
      FindTileIrAssembler(debug_options.xla_gpu_cuda_data_dir());
  if (!tileiras_path.ok()) {
    // Missing tileiras: log once; autotuner treats failures as non-viable.
    static absl::once_flag once;
    absl::call_once(once, [&] {
      LOG(ERROR) << "xla_gpu_experimental_scaled_dot_with_tile_ir is on but "
                    "`tileiras` was not found ("
                 << tileiras_path.status().message()
                 << "). Every CUDA Tile IR candidate will fail and the "
                    "autotuner will silently fall back to Triton. Put the "
                    "CUDA Tile IR assembler on $PATH or in "
                    "$XLA_FLAGS --xla_gpu_cuda_data_dir.";
    });
    return tileiras_path.status();
  }

  tsl::Env* env = tsl::Env::Default();
  std::string bytecode_path;
  std::string cubin_path;
  if (!env->LocalTempFilename(&bytecode_path) ||
      !env->LocalTempFilename(&cubin_path)) {
    return absl::InternalError("Could not get a temporary file name.");
  }
  absl::Cleanup cleanup = [&] {
    env->DeleteFile(bytecode_path).IgnoreError();
    env->DeleteFile(cubin_path).IgnoreError();
  };
  RETURN_IF_ERROR(tsl::WriteStringToFile(env, bytecode_path, bytecode));

  // tileiras wants plain sm_120; it passes sm_120a to ptxas itself.
  std::vector<std::string> args = {
      *tileiras_path, bytecode_path, "-o", cubin_path, "--gpu-name",
      absl::StrCat("sm_", cc->major, cc->minor)};

  VLOG(3) << "Running: " << absl::StrJoin(args, " ");
  tsl::SubProcess tileiras;
  tileiras.SetProgram(*tileiras_path, args);
  tileiras.SetChannelAction(tsl::CHAN_STDERR, tsl::ACTION_PIPE);
  if (!tileiras.Start()) {
    return absl::InternalError("Failed to launch tileiras.");
  }

  // Communicate() has no timeout — kill hung tileiras or the run wedges.
  absl::Notification finished;
  std::atomic<bool> timed_out{false};
  std::thread watchdog([&] {
    if (!finished.WaitForNotificationWithTimeout(TileIrCompilerTimeout())) {
      timed_out.store(true);
      tileiras.Kill(SIGKILL);
    }
  });
  std::string stderr_output;
  int exit_status = tileiras.Communicate(
      /*stdin_input=*/nullptr, /*stdout_output=*/nullptr, &stderr_output);
  finished.Notify();
  watchdog.join();

  if (timed_out.load()) {
    return absl::DeadlineExceededError(absl::StrCat(
        "tileiras exceeded ", absl::FormatDuration(TileIrCompilerTimeout()),
        " and was killed. Set ", kCompilerTimeoutEnvVar,
        " to change the limit. Partial stderr: ", stderr_output));
  }
  if (exit_status != 0) {
    return absl::InternalError(absl::StrCat(
        "tileiras exited with ", exit_status, ": ", stderr_output));
  }
  if (!stderr_output.empty()) {
    VLOG(2) << "tileiras: " << stderr_output;
  }

  std::string cubin;
  RETURN_IF_ERROR(tsl::ReadFileToString(env, cubin_path, &cubin));
  std::vector<uint8_t> result(cubin.begin(), cubin.end());

  if (const char* dump_dir = std::getenv("XLA_TILE_IR_DUMP_DIR")) {
    const std::string stem =
        absl::StrCat(dump_dir, "/tile_ir_",
                     absl::Hex(absl::HashOf(cache_key), absl::kZeroPad16));
    absl::Status bc = tsl::WriteStringToFile(env, absl::StrCat(stem, ".tilebc"),
                                             std::string(bytecode));
    absl::Status cb =
        tsl::WriteStringToFile(env, absl::StrCat(stem, ".cubin"), cubin);
    if (!bc.ok() || !cb.ok()) {
      LOG_FIRST_N(WARNING, 1) << "XLA_TILE_IR_DUMP_DIR is set but writing to "
                              << dump_dir << " failed: " << bc << " " << cb;
    }
  }

  {
    absl::MutexLock lock(&CubinCacheMutex());
    CubinCache().emplace(std::move(cache_key), result);
  }
  return result;
}

}  // namespace xla::gpu::tile_ir
