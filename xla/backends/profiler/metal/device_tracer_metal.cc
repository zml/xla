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

// Apple Metal device tracer for the XLA/TSL profiler (xprof). Registers a
// tsl::profiler::ProfilerInterface that, while a profile session is active,
// drains the per-op GPU timestamps collected by the Metal StreamExecutor
// runtime (stream_executor/metal: MetalProfiling*) into a GPU device XPlane.
//
// The GPU timestamping itself lives in the runtime (an MTLCounterSampleBuffer
// sampled at compute-encoder boundaries). This file is the thin
// ProfilerInterface + XPlane builder, so it is plain C++ — no Objective-C++.
//
// It is driven automatically by the existing ZML `--profile` path:
//   PLUGIN_Profiler_Create -> tsl::profiler::CreateProfilers(options)
//   -> our registered factory -> MetalTracer::Start/Stop/CollectData.
// Mirrors xla/backends/profiler/gpu/device_tracer_cuda.cc.

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "xla/stream_executor/metal/metal_runtime.h"
#include "xla/tsl/profiler/utils/time_utils.h"
#include "xla/tsl/profiler/utils/xplane_builder.h"
#include "xla/tsl/profiler/utils/xplane_schema.h"
#include "tsl/profiler/lib/profiler_factory.h"
#include "tsl/profiler/lib/profiler_interface.h"
#include "tsl/profiler/protobuf/profiler_options.pb.h"
#include "tsl/profiler/protobuf/xplane.pb.h"

namespace xla {
namespace profiler {
namespace {

namespace tsl_prof = tsl::profiler;
using tensorflow::ProfileOptions;
using tensorflow::profiler::XSpace;

// ProfilerInterface for the Apple Metal backend. Mirrors the CUDA GpuTracer
// (device_tracer_cuda.cc): a small Start/Stop/CollectData lifecycle. The actual
// GPU timestamping lives in the Metal runtime; here we toggle it and, on
// CollectData, turn the collected events into a GPU device XPlane.
class MetalTracer : public tsl_prof::ProfilerInterface {
 public:
  MetalTracer() = default;

  absl::Status Start() override {
    stream_executor::metal::MetalProfilingStart();
    // The XPlane line epoch: the host wall clock at Start, the SAME clock
    // (tsl::profiler::GetCurrentTimeNanos) the host TraceMe tracer uses, so GPU
    // events land on a shared timeline.
    start_walltime_ns_ = tsl_prof::GetCurrentTimeNanos();
    started_ = true;
    return absl::OkStatus();
  }

  absl::Status Stop() override {
    if (started_) stream_executor::metal::MetalProfilingStop();
    return absl::OkStatus();
  }

  absl::Status CollectData(XSpace* space) override {
    if (!started_) return absl::OkStatus();
    std::vector<stream_executor::metal::MetalProfileEvent> events =
        stream_executor::metal::MetalProfilingDrain();
    const uint64_t dropped =
        stream_executor::metal::MetalProfilingDroppedCount();

    LOG(INFO) << "MetalTracer::CollectData: " << events.size()
              << " GPU events, " << dropped << " dropped.";

    if (events.empty()) {
      space->add_warnings(
          "Metal GPU tracer collected no kernel events (GPU counter sampling "
          "unavailable, or no AIR compute kernels ran — MPSGraph matmuls are "
          "not captured).");
      return absl::OkStatus();
    }

    // One GPU device plane, named so xprof recognizes it as a device timeline.
    tsl_prof::XPlaneBuilder plane(space->add_planes());
    plane.SetName(tsl_prof::GpuPlaneName(/*device_ordinal=*/0));
    plane.SetId(0);

    // One line for the single Metal command queue. The line epoch is the host
    // wall clock at Start; events carry absolute wall ns (the runtime already
    // mapped GPU ticks onto this clock), so offsets are non-negative and align
    // with host TraceMe events. Set the line timestamp BEFORE adding events:
    // each event's offset is computed relative to the line's current epoch.
    tsl_prof::XLineBuilder line = plane.GetOrCreateLine(/*line_id=*/0);
    line.SetName("Stream #0");
    line.SetTimestampNs(start_walltime_ns_);

    auto* bytes_md = plane.GetOrCreateStatMetadata(
        tsl_prof::GetStatTypeStr(tsl_prof::StatType::kBytesAccessed));
    auto* details_md = plane.GetOrCreateStatMetadata(
        tsl_prof::GetStatTypeStr(tsl_prof::StatType::kKernelDetails));

    for (const auto& ev : events) {
      auto* event_md = plane.GetOrCreateEventMetadata(ev.name);
      tsl_prof::XEventBuilder xevent = line.AddEvent(*event_md);
      xevent.SetTimestampNs(ev.start_ns);
      xevent.SetEndTimestampNs(ev.end_ns);
      if (ev.bytes > 0) {
        xevent.AddStatValue(*bytes_md, static_cast<uint64_t>(ev.bytes));
      }
      if (!ev.details.empty()) {
        // kKernelDetails value is a ref to a StatMetadata whose name is the
        // detail string (the convention CUPTI uses; xprof parses it).
        xevent.AddStatValue(*details_md,
                            *plane.GetOrCreateStatMetadata(ev.details));
      }
    }

    if (dropped > 0) {
      space->add_warnings(absl::StrCat("Metal GPU tracer dropped ", dropped,
                                       " events (sample buffer full or invalid "
                                       "sample)."));
    }
    return absl::OkStatus();
  }

 private:
  bool started_ = false;
  uint64_t start_walltime_ns_ = 0;
};

}  // namespace

// Factory matching tsl::profiler::ProfilerFactory. Gated like the CUDA/ROCm
// tracers: opt out when device tracing is disabled or another device type is
// explicitly requested. ZML sends device_type=UNSPECIFIED, which passes.
std::unique_ptr<tsl_prof::ProfilerInterface> CreateMetalTracer(
    const ProfileOptions& options) {
  if (options.device_tracer_level() == 0) return nullptr;
  if (options.device_type() != ProfileOptions::GPU &&
      options.device_type() != ProfileOptions::UNSPECIFIED) {
    return nullptr;
  }
  return std::make_unique<MetalTracer>();
}

// Registers the factory at .dylib load. Requires alwayslink=1 on the BUILD
// target so this translation unit is not dropped (nothing references it by
// symbol). Mirrors device_tracer_cuda.cc's register_gpu_tracer_factory.
auto register_metal_tracer_factory = [] {
  tsl_prof::RegisterProfilerFactory(&CreateMetalTracer);
  return 0;
}();

}  // namespace profiler
}  // namespace xla
