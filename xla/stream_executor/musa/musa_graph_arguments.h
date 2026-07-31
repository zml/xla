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

#ifndef XLA_STREAM_EXECUTOR_MUSA_MUSA_GRAPH_ARGUMENTS_H_
#define XLA_STREAM_EXECUTOR_MUSA_MUSA_GRAPH_ARGUMENTS_H_

#include <memory>

#include "absl/status/statusor.h"
#include "xla/stream_executor/kernel_args.h"
#include "xla/stream_executor/kernel_args_packed_vector.h"

namespace stream_executor::musa {

// MUSA 4.0.1 graph nodes retain kernelParams storage beyond the driver call.
// Clone packed argument bytes so both the pointer array and its pointees remain
// valid for graph instantiation and asynchronous replay.
absl::StatusOr<std::unique_ptr<KernelArgsPackedVector>>
CloneMusaGraphKernelArguments(const KernelArgsPackedArrayBase& arguments);

}  // namespace stream_executor::musa

#endif  // XLA_STREAM_EXECUTOR_MUSA_MUSA_GRAPH_ARGUMENTS_H_
