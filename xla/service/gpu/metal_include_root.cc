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

#include "xla/service/gpu/metal_include_root.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

#if defined(__APPLE__)
#include <unistd.h>  // confstr, _CS_DARWIN_USER_CACHE_DIR
#endif

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "tsl/platform/fingerprint.h"
#include "xla/service/gpu/metal_kernels/mlx_include_tree.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
#include "xla/util/embedded_file.h"

namespace xla {
namespace gpu {
namespace {

std::vector<EmbeddedFile> SortedTree() {
  absl::Span<const EmbeddedFile> files = get_mlx_include_tree();
  std::vector<EmbeddedFile> sorted(files.begin(), files.end());
  std::sort(sorted.begin(), sorted.end(),
            [](const EmbeddedFile& a, const EmbeddedFile& b) {
              return a.path < b.path;
            });
  return sorted;
}

std::string ComputeTreeHash() {
  const std::vector<EmbeddedFile> files = SortedTree();
  tsl::Fprint128 h = {0, 0};
  for (const EmbeddedFile& f : files) {
    for (absl::string_view part : {f.path, f.contents}) {
      const tsl::Fprint128 p = tsl::Fingerprint128(part);
      h = tsl::FingerprintCat128(h, {p.low64 ^ part.size(), p.high64});
    }
  }
  return absl::StrFormat("%016x%016x", h.high64, h.low64);
}

std::string TempDir() {
  if (const char* t = std::getenv("TMPDIR"); t != nullptr && *t != '\0') {
    absl::string_view dir(t);
    while (dir.size() > 1 && dir.back() == '/') dir.remove_suffix(1);
    return std::string(dir);
  }
  return "/tmp";
}

// The per-user cache directory, not the temp directory: the OS prunes idle
// temp files by access time and would gut the tree under a long-lived process.
std::string CacheParentDir() {
#if defined(__APPLE__)
  char buf[1024];
  const size_t n = confstr(_CS_DARWIN_USER_CACHE_DIR, buf, sizeof(buf));
  if (n > 0 && n <= sizeof(buf)) {
    absl::string_view dir(buf, n - 1);  // n counts the terminating NUL.
    while (dir.size() > 1 && dir.back() == '/') dir.remove_suffix(1);
    if (!dir.empty()) return std::string(dir);
  }
#endif
  return TempDir();
}

bool TreeIsComplete(const std::string& root, tsl::Env* env) {
  for (const EmbeddedFile& f : SortedTree()) {
    if (!env->FileExists(absl::StrCat(root, "/", f.path)).ok()) return false;
  }
  return true;
}

absl::Status WriteTreeInto(const std::string& staging, tsl::Env* env) {
  for (const EmbeddedFile& f : SortedTree()) {
    const std::string path = absl::StrCat(staging, "/", f.path);
    const std::string::size_type slash = path.find_last_of('/');
    if (slash != std::string::npos) {
      TF_RETURN_IF_ERROR(env->RecursivelyCreateDir(path.substr(0, slash)));
    }
    TF_RETURN_IF_ERROR(tsl::WriteStringToFile(env, path, f.contents));
  }
  return absl::OkStatus();
}

absl::StatusOr<std::string> EnsureMaterialized() {
  tsl::Env* env = tsl::Env::Default();
  const std::string root =
      absl::StrCat(CacheParentDir(), "/xla-metal-inc-", MetalIncludeTreeHash());

  if (TreeIsComplete(root, env)) return root;

  std::string staging;
  if (!env->LocalTempFilename(&staging)) {
    return absl::InternalError(
        "Could not create a Metal include tree staging path.");
  }
  TF_RETURN_IF_ERROR(WriteTreeInto(staging, env));

  if (env->FileExists(root).ok()) {
    int64_t undeleted_files, undeleted_dirs;
    env->DeleteRecursively(root, &undeleted_files, &undeleted_dirs)
        .IgnoreError();
  }
  if (std::rename(staging.c_str(), root.c_str()) == 0) return root;
  if (TreeIsComplete(root, env)) {
    int64_t undeleted_files, undeleted_dirs;
    env->DeleteRecursively(staging, &undeleted_files, &undeleted_dirs)
        .IgnoreError();
    return root;
  }
  return staging;
}

}  // namespace

absl::string_view MetalIncludeTreeHash() {
  static const std::string* const kHash = new std::string(ComputeTreeHash());
  return *kHash;
}

absl::StatusOr<std::string> MetalIncludeRoot() {
  static absl::Mutex* const mu = new absl::Mutex();
  absl::MutexLock lock(mu);
  return EnsureMaterialized();
}

}  // namespace gpu
}  // namespace xla
