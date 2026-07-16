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

#include "xla/pjrt/tools/pjrt_oneapi_issue_profiler.h"

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <string>
#include <utility>

#include "absl/log/check.h"
#include "absl/strings/str_cat.h"
#include "xla/tsl/profiler/convert/trace_events_to_json.h"
#include "xla/tsl/profiler/convert/xplane_to_trace_events.h"
#include "tsl/profiler/lib/profiler_session.h"
#include "tsl/profiler/lib/traceme.h"
#include "tsl/profiler/protobuf/profiler_options.pb.h"
#include "tsl/profiler/protobuf/xplane.pb.h"

namespace xla {

struct IssueProfiler::Impl {
  explicit Impl(std::string output_path) : output_path(std::move(output_path)) {
    tensorflow::ProfileOptions options = tsl::ProfilerSession::DefaultOptions();
    options.set_host_tracer_level(3);
    options.set_device_tracer_level(0);
    options.set_enable_hlo_proto(false);
    session = tsl::ProfilerSession::Create(options);
    CHECK_OK(session->Status());

    const int64_t unix_us =
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::system_clock::now().time_since_epoch())
            .count();
    clock_sync = std::make_unique<tsl::profiler::TraceMe>(
        [unix_us] { return absl::StrCat("XLA_CLOCK_SYNC_UNIX_US=", unix_us); });
  }

  ~Impl() {
    clock_sync.reset();
    tensorflow::profiler::XSpace xspace;
    CHECK_OK(session->CollectData(&xspace));
    tsl::profiler::TraceContainer trace =
        tsl::profiler::ConvertXSpaceToTraceContainer(xspace);
    std::ofstream output(output_path, std::ios::binary | std::ios::trunc);
    CHECK(output) << "Could not open host trace: " << output_path;
    output << tsl::profiler::TraceContainerToJson(trace);
    CHECK(output) << "Could not write host trace: " << output_path;
  }

  std::string output_path;
  std::unique_ptr<tsl::ProfilerSession> session;
  std::unique_ptr<tsl::profiler::TraceMe> clock_sync;
};

std::unique_ptr<IssueProfiler> IssueProfiler::CreateFromEnvironment() {
  const char* output_path = std::getenv("XLA_HOST_TRACE_FILE");
  if (output_path == nullptr || *output_path == '\0') return nullptr;
  return std::unique_ptr<IssueProfiler>(
      new IssueProfiler(std::make_unique<Impl>(output_path)));
}

IssueProfiler::IssueProfiler(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}

IssueProfiler::~IssueProfiler() = default;

}  // namespace xla
