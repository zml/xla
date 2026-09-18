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

#ifndef XLA_STREAM_EXECUTOR_MUSA_MUSA_DSO_LOADER_H_
#define XLA_STREAM_EXECUTOR_MUSA_MUSA_DSO_LOADER_H_

#include <memory>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"

namespace stream_executor::musa::internal {

// Injectable boundary between MUSA API tables and the operating-system DSO
// loader. Implementations must cache the exact result of Load(). Resolve() is
// safe to call concurrently after Load() succeeds.
class MusaSymbolLoader {
 public:
  virtual ~MusaSymbolLoader() = default;

  virtual absl::Status Load() = 0;
  virtual absl::StatusOr<void*> Resolve(absl::string_view symbol) const = 0;
  virtual absl::string_view loaded_path() const = 0;
};

// Expands DSO candidates according to the production search policy. The
// original candidates retain their order and are followed by
// /usr/local/musa/lib fallbacks for bare SONAMEs. Exposed to make this ABI
// search policy directly testable.
std::vector<std::string> ExpandMusaDsoCandidates(
    const std::vector<std::string>& candidates);

// Creates a loader that tries `candidates` in order, followed by matching
// /usr/local/musa/lib paths for bare SONAMEs. If `fail_if_not_found` is true,
// exhausting all candidates is a configuration error instead of normal DSO
// absence. An existing path entry that cannot be loaded, including a dangling
// symbolic link, is always an immediate configuration error; later fallbacks
// are not tried. A successfully opened DSO is deliberately pinned for the
// lifetime of the process: unloading a vendor driver or runtime DSO while
// another thread may be executing it is unsafe.
//
// POSIX dlopen does not portably distinguish an absent bare SONAME from a
// missing transitive dependency. Callers requiring fail-closed diagnostics
// must provide a slash-containing candidate or set fail_if_not_found.
std::unique_ptr<MusaSymbolLoader> CreateMusaDsoLoader(
    std::vector<std::string> candidates, bool fail_if_not_found = false);

// Creates the qualified MUSA driver loader. The versioned SONAME is preferred
// so that an unrelated default libmusa cannot silently change the ABI.
std::unique_ptr<MusaSymbolLoader> CreateMusaDriverDsoLoader();

}  // namespace stream_executor::musa::internal

#endif  // XLA_STREAM_EXECUTOR_MUSA_MUSA_DSO_LOADER_H_
