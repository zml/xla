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

#ifndef XLA_STREAM_EXECUTOR_MUSA_MUSA_OPTIONAL_LIBRARY_ABI_H_
#define XLA_STREAM_EXECUTOR_MUSA_MUSA_OPTIONAL_LIBRARY_ABI_H_

#include <string>

namespace stream_executor::musa {

inline constexpr char kMusaMuBlasLibraryAbiName[] = "mublas";
inline constexpr char kMusaMuBlasLibraryAbiVersion[] = "1";
inline constexpr char kMusaMuBlasScalLibraryAbiName[] = "mublas-scal";
inline constexpr char kMusaMuBlasScalLibraryAbiVersion[] = "1";

// Runtime identity of an optional vendor-library adapter. The ABI version is
// the compatibility key; a nonempty fingerprint can additionally constrain a
// serialized executable to one qualified implementation.
struct MusaOptionalLibraryAbi {
  std::string name;
  std::string abi_version;
  std::string fingerprint;

  bool operator==(const MusaOptionalLibraryAbi& other) const {
    return name == other.name && abi_version == other.abi_version &&
           fingerprint == other.fingerprint;
  }
};

}  // namespace stream_executor::musa

#endif  // XLA_STREAM_EXECUTOR_MUSA_MUSA_OPTIONAL_LIBRARY_ABI_H_
