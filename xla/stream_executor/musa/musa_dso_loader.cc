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

#include "xla/stream_executor/musa/musa_dso_loader.h"

#include <dlfcn.h>
#include <sys/stat.h>

#if defined(__linux__)
#include <link.h>
#endif

#include <cerrno>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"

namespace stream_executor::musa::internal {
namespace {

class PosixMusaSymbolLoader final : public MusaSymbolLoader {
 public:
  explicit PosixMusaSymbolLoader(std::vector<std::string> candidates,
                                 bool fail_if_not_found)
      : candidates_(std::move(candidates)),
        fail_if_not_found_(fail_if_not_found) {}

  // Do not call dlclose. Driver and runtime DSOs can own process-wide state,
  // TLS destructors, callbacks, and objects retained by other vendor DSOs.
  ~PosixMusaSymbolLoader() override = default;

  absl::Status Load() override {
    std::call_once(load_once_, [this] { load_status_ = LoadOnce(); });
    return load_status_;
  }

  absl::StatusOr<void*> Resolve(absl::string_view symbol) const override {
    absl::Status status = const_cast<PosixMusaSymbolLoader*>(this)->Load();
    if (!status.ok()) return status;

    // POSIX requires clearing a stale dlerror before dlsym and checking it
    // afterwards. A null address alone is not a portable missing-symbol test.
    dlerror();
    void* address = dlsym(handle_, std::string(symbol).c_str());
    const char* error = dlerror();
    if (error != nullptr) {
      return absl::NotFoundError(absl::StrCat("MUSA symbol ", symbol,
                                              " was not found in ",
                                              loaded_path_, ": ", error));
    }
    if (address == nullptr) {
      return absl::NotFoundError(absl::StrCat("MUSA symbol ", symbol,
                                              " resolved to a null address in ",
                                              loaded_path_));
    }
    return address;
  }

  absl::string_view loaded_path() const override { return loaded_path_; }

 private:
  absl::Status LoadOnce() {
    if (candidates_.empty()) {
      return absl::InvalidArgumentError(
          "No candidate MUSA shared libraries were provided");
    }

    std::vector<std::string> failures;
    failures.reserve(candidates_.size());
    for (const std::string& candidate : candidates_) {
      bool candidate_path_exists = false;
      if (candidate.find('/') != std::string::npos) {
        struct stat candidate_stat;
        int stat_result;
        do {
          stat_result = lstat(candidate.c_str(), &candidate_stat);
        } while (stat_result != 0 && errno == EINTR);
        if (stat_result == 0) {
          candidate_path_exists = true;
        } else {
          const int stat_error = errno;
          if (stat_error != ENOENT && stat_error != ENOTDIR) {
            return absl::FailedPreconditionError(
                absl::StrCat("Could not inspect MUSA shared-library candidate ",
                             candidate, ": ", std::strerror(stat_error)));
          }
        }
      }
      dlerror();
      handle_ = dlopen(candidate.c_str(), RTLD_NOW | RTLD_LOCAL);
      const char* error = dlerror();
      if (handle_ == nullptr) {
        const char* error_message =
            error == nullptr ? "unknown dlopen error" : error;
        failures.push_back(absl::StrCat(candidate, ": ", error_message));
        if (candidate_path_exists) {
          return absl::FailedPreconditionError(absl::StrCat(
              "Could not load a MUSA shared library at existing path ",
              candidate, ": ", error_message));
        }
        continue;
      }

      loaded_path_ = candidate;
#if defined(__linux__)
      link_map* map = nullptr;
      if (dlinfo(handle_, RTLD_DI_LINKMAP, &map) == 0 && map != nullptr &&
          map->l_name != nullptr && map->l_name[0] != '\0') {
        loaded_path_ = map->l_name;
      }
#endif
      return absl::OkStatus();
    }

    const std::string message = absl::StrCat(
        "Could not load a MUSA shared library (tried ",
        absl::StrJoin(candidates_, ", "), "): ", absl::StrJoin(failures, "; "));
    if (fail_if_not_found_) {
      return absl::FailedPreconditionError(message);
    }
    return absl::NotFoundError(message);
  }

  const std::vector<std::string> candidates_;
  const bool fail_if_not_found_;
  mutable std::once_flag load_once_;
  absl::Status load_status_ = absl::UnknownError("MUSA DSO not loaded");
  void* handle_ = nullptr;  // Intentionally never passed to dlclose.
  std::string loaded_path_;
};

}  // namespace

std::vector<std::string> ExpandMusaDsoCandidates(
    const std::vector<std::string>& candidates) {
  std::vector<std::string> expanded = candidates;
  expanded.reserve(candidates.size() * 2);
  for (const std::string& candidate : candidates) {
    if (candidate.find('/') == std::string::npos) {
      expanded.push_back(absl::StrCat("/usr/local/musa/lib/", candidate));
    }
  }
  return expanded;
}

std::unique_ptr<MusaSymbolLoader> CreateMusaDsoLoader(
    std::vector<std::string> candidates, bool fail_if_not_found) {
  // A normal deployment registers the toolkit with the dynamic linker. The
  // vendor installer used on qualified S80 hosts can instead leave DSOs only
  // under /usr/local/musa/lib. Keep SONAME resolution first, then add that
  // conventional install root as a deterministic fallback. Bazel output and
  // runfiles paths are intentionally not part of the runtime ABI.
  return std::make_unique<PosixMusaSymbolLoader>(
      ExpandMusaDsoCandidates(candidates), fail_if_not_found);
}

std::unique_ptr<MusaSymbolLoader> CreateMusaDriverDsoLoader() {
  return CreateMusaDsoLoader({"libmusa.so.1.5", "libmusa.so"});
}

}  // namespace stream_executor::musa::internal
