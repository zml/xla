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

#ifndef XLA_SERVICE_GPU_METAL_INCLUDE_ROOT_H_
#define XLA_SERVICE_GPU_METAL_INCLUDE_ROOT_H_

#include <string>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"

namespace xla {
namespace gpu {

absl::StatusOr<std::string> MetalIncludeRoot();

// Every metallib cache key must include this: with -I the include tree
// decides the output without appearing in the source.
absl::string_view MetalIncludeTreeHash();

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_METAL_INCLUDE_ROOT_H_
