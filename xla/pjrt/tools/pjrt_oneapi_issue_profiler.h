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

#ifndef XLA_PJRT_TOOLS_PJRT_ONEAPI_ISSUE_PROFILER_H_
#define XLA_PJRT_TOOLS_PJRT_ONEAPI_ISSUE_PROFILER_H_

#include <memory>

namespace xla {

// Starts host tracing only when XLA_HOST_TRACE_FILE is set.
class IssueProfiler {
 public:
  static std::unique_ptr<IssueProfiler> CreateFromEnvironment();
  ~IssueProfiler();

 private:
  struct Impl;
  explicit IssueProfiler(std::unique_ptr<Impl> impl);

  std::unique_ptr<Impl> impl_;
};

}  // namespace xla

#endif  // XLA_PJRT_TOOLS_PJRT_ONEAPI_ISSUE_PROFILER_H_
