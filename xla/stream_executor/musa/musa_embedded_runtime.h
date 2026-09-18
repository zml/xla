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

#ifndef XLA_STREAM_EXECUTOR_MUSA_MUSA_EMBEDDED_RUNTIME_H_
#define XLA_STREAM_EXECUTOR_MUSA_MUSA_EMBEDDED_RUNTIME_H_

#include <array>
#include <memory>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/tsl/util/file_toc.h"

namespace stream_executor::musa::internal {

// Owns private copies of the five embedded ELF files. The SDK itself is never
// copied or modified; a private symlink supplies the helpers' relative RPATHs.
class MusaEmbeddedRuntime {
 public:
  static absl::StatusOr<std::unique_ptr<MusaEmbeddedRuntime>> Create(
      absl::string_view sdk_root, absl::string_view temporary_root,
      absl::Span<const FileToc> files);
  ~MusaEmbeddedRuntime();
  MusaEmbeddedRuntime(const MusaEmbeddedRuntime&) = delete;
  MusaEmbeddedRuntime& operator=(const MusaEmbeddedRuntime&) = delete;

  const std::string& directory() const { return directory_; }
  const std::string& sdk_root() const { return sdk_root_; }
  std::string Path(absl::string_view relative) const;
  absl::Status Verify() const;

 private:
  MusaEmbeddedRuntime(std::string directory, std::string sdk_root);
  struct ExtractedFile {
    std::string path;
    size_t size;
    std::array<unsigned char, 32> sha256;
  };
  std::string directory_;
  std::string sdk_root_;
  std::vector<ExtractedFile> files_;
};

// Thread-safe, process-lifetime extraction with normal-exit cleanup. SDK
// discovery uses MUSA_PATH, otherwise musa-sdk beside this shared object.
// XLA_MUSA_COMPILATION_TEMP_ROOT selects the extraction parent (default /tmp).
// An explicit SDK root takes precedence; changing it after extraction fails.
absl::StatusOr<const MusaEmbeddedRuntime*> GetMusaEmbeddedRuntime(
    absl::string_view sdk_root = {});

}  // namespace stream_executor::musa::internal

#endif  // XLA_STREAM_EXECUTOR_MUSA_MUSA_EMBEDDED_RUNTIME_H_
