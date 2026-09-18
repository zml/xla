// Copyright 2026 The OpenXLA Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef XLA_STREAM_EXECUTOR_MUSA_MUBIN_SPEC_H_
#define XLA_STREAM_EXECUTOR_MUSA_MUBIN_SPEC_H_

#include <cstdint>
#include <vector>

#include "absl/types/span.h"

namespace stream_executor {

// A non-owning view of a MUSA device binary (MUBIN) in memory.
//
// MUBIN is a distinct binary format and must not be represented as a CUDA
// CUBIN. The referenced bytes must outlive the loader specification.
struct MusaMubinInMemory {
  absl::Span<const uint8_t> mubin_bytes;
};

// An owning MUSA device binary. Loader specifications expose this through a
// MusaMubinInMemory view so callers do not need to distinguish ownership.
struct OwningMusaMubinInMemory {
  std::vector<uint8_t> mubin_bytes;
};

}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_MUSA_MUBIN_SPEC_H_
