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

#include <sys/resource.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "xla/future.h"
#include "xla/hlo/builder/xla_builder.h"
#include "xla/literal.h"
#include "xla/pjrt/c_api_client/pjrt_c_api_client.h"
#include "xla/pjrt/pjrt_api.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/python/profiler_utils.h"
#include "xla/service/computation_placer.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/framework/allocator.h"
#include "xla/tsl/platform/status_macros.h"
#include "xla/tsl/profiler/convert/trace_events_to_json.h"
#include "xla/tsl/profiler/convert/xplane_to_trace_events.h"
#include "xla/xla_data.pb.h"
#include "tsl/profiler/lib/profiler_session.h"
#include "tsl/profiler/lib/traceme.h"
#include "tsl/profiler/lib/traceme_encode.h"
#include "tsl/profiler/protobuf/profiler_options.pb.h"
#include "tsl/profiler/protobuf/xplane.pb.h"

namespace xla {
namespace {

using Clock = std::chrono::steady_clock;

struct Options {
  std::string plugin_path;
  std::string output_dir;
  int iterations = 10;
  int warmup_iterations = 1;
  int device_count = 1;
  int matrix_size = 256;
  int compile_iterations = 1;
  bool profile = true;
};

struct StageSample {
  std::string name;
  int64_t start_ns;
  int64_t end_ns;
};

struct MemorySample {
  std::string stage;
  int device_ordinal;
  bool available;
  std::string error;
  tsl::AllocatorStats stats;
};

struct ExecutionSample {
  int64_t execution_id;
  int iteration;
  std::string phase;
  int64_t call_start_ns;
  int64_t call_return_ns;
  int64_t completion_ns;
  std::vector<int64_t> per_device_completion_ns;
  int64_t validation_start_ns = 0;
  int64_t validation_end_ns = 0;
  std::vector<double> checksums;
  std::vector<double> max_abs_errors;
  bool validated = false;
};

struct Results {
  int64_t program_start_ns = 0;
  int64_t program_end_ns = 0;
  int64_t unix_start_ns = 0;
  int64_t unix_end_ns = 0;
  int64_t max_rss_kb = 0;
  int64_t generated_code_bytes = -1;
  std::string platform_name;
  std::string platform_version;
  std::vector<std::string> devices;
  std::vector<StageSample> stages;
  std::vector<MemorySample> memory;
  std::vector<ExecutionSample> executions;
};

int64_t NowNs() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             Clock::now().time_since_epoch())
      .count();
}

int64_t UnixNowNs() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::system_clock::now().time_since_epoch())
      .count();
}

double ToUs(int64_t nanoseconds) {
  return static_cast<double>(nanoseconds) / 1000.0;
}

std::string JsonEscape(std::string_view value) {
  std::ostringstream out;
  for (unsigned char c : value) {
    switch (c) {
      case '"':
        out << "\\\"";
        break;
      case '\\':
        out << "\\\\";
        break;
      case '\b':
        out << "\\b";
        break;
      case '\f':
        out << "\\f";
        break;
      case '\n':
        out << "\\n";
        break;
      case '\r':
        out << "\\r";
        break;
      case '\t':
        out << "\\t";
        break;
      default:
        if (c < 0x20) {
          out << "\\u" << std::hex << std::setw(4) << std::setfill('0')
              << static_cast<int>(c) << std::dec;
        } else {
          out << c;
        }
    }
  }
  return out.str();
}

std::string EnvironmentValue(const char* name) {
  const char* value = std::getenv(name);
  return value == nullptr ? "" : value;
}

absl::Status WriteTextFile(const std::string& path, std::string_view content) {
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out) {
    return absl::InternalError(absl::StrCat("Unable to open ", path));
  }
  out.write(content.data(), content.size());
  if (!out) {
    return absl::InternalError(absl::StrCat("Unable to write ", path));
  }
  return absl::OkStatus();
}

absl::StatusOr<Options> ParseOptions(int argc, char** argv) {
  Options options;
  auto parse_int = [](std::string_view text,
                      const char* name) -> absl::StatusOr<int> {
    int value = 0;
    if (!absl::SimpleAtoi(text, &value)) {
      return absl::InvalidArgumentError(
          absl::StrCat("Invalid integer for --", name, ": ", text));
    }
    return value;
  };

  for (int i = 1; i < argc; ++i) {
    std::string_view argument(argv[i]);
    if (argument == "--help") {
      std::cout
          << "Usage: pjrt_oneapi_reproducer --plugin_path=PATH "
             "--output_dir=DIR [--iterations=N] [--warmup_iterations=N] "
             "[--device_count=N] [--matrix_size=N] "
             "[--compile_iterations=N] [--profile=0|1]\n";
      std::exit(0);
    }
    if (argument.size() < 3 || argument.substr(0, 2) != "--" ||
        argument.find('=') == std::string::npos) {
      return absl::InvalidArgumentError(
          absl::StrCat("Expected --name=value, got: ", argument));
    }
    const size_t equals = argument.find('=');
    const std::string name(argument.substr(2, equals - 2));
    const std::string value(argument.substr(equals + 1));
    if (name == "plugin_path") {
      options.plugin_path = value;
    } else if (name == "output_dir") {
      options.output_dir = value;
    } else if (name == "iterations") {
      ASSIGN_OR_RETURN(options.iterations, parse_int(value, name.c_str()));
    } else if (name == "warmup_iterations") {
      ASSIGN_OR_RETURN(options.warmup_iterations,
                       parse_int(value, name.c_str()));
    } else if (name == "device_count") {
      ASSIGN_OR_RETURN(options.device_count, parse_int(value, name.c_str()));
    } else if (name == "matrix_size") {
      ASSIGN_OR_RETURN(options.matrix_size, parse_int(value, name.c_str()));
    } else if (name == "compile_iterations") {
      ASSIGN_OR_RETURN(options.compile_iterations,
                       parse_int(value, name.c_str()));
    } else if (name == "profile") {
      int profile = 0;
      ASSIGN_OR_RETURN(profile, parse_int(value, name.c_str()));
      if (profile != 0 && profile != 1) {
        return absl::InvalidArgumentError("--profile must be 0 or 1");
      }
      options.profile = profile == 1;
    } else {
      return absl::InvalidArgumentError(
          absl::StrCat("Unknown option: --", name));
    }
  }

  if (options.plugin_path.empty() || options.output_dir.empty()) {
    return absl::InvalidArgumentError(
        "--plugin_path and --output_dir are required");
  }
  if (options.iterations < 1 || options.warmup_iterations < 0 ||
      options.device_count < 1 || options.matrix_size < 1 ||
      options.compile_iterations < 1) {
    return absl::InvalidArgumentError(
        "iterations, device_count, matrix_size, and compile_iterations must be "
        "positive; warmup_iterations must be non-negative");
  }
  return options;
}

void RecordMemory(absl::Span<PjRtDevice* const> devices,
                  std::string stage, Results* results) {
  for (int i = 0; i < devices.size(); ++i) {
    MemorySample sample;
    sample.stage = stage;
    sample.device_ordinal = i;
    auto stats = devices[i]->GetAllocatorStats();
    sample.available = stats.ok();
    if (stats.ok()) {
      sample.stats = *stats;
    } else {
      sample.error = std::string(stats.status().message());
    }
    results->memory.push_back(std::move(sample));
  }
}

absl::Status WriteMetrics(const Options& options, const Results& results) {
  std::ostringstream out;
  out << std::setprecision(17);
  out << "{\n"
      << "  \"schema_version\": 1,\n"
      << "  \"success\": true,\n"
      << "  \"pid\": " << getpid() << ",\n"
      << "  \"program_start_monotonic_ns\": " << results.program_start_ns
      << ",\n"
      << "  \"program_end_monotonic_ns\": " << results.program_end_ns << ",\n"
      << "  \"program_duration_us\": "
      << ToUs(results.program_end_ns - results.program_start_ns) << ",\n"
      << "  \"program_start_unix_ns\": " << results.unix_start_ns << ",\n"
      << "  \"program_end_unix_ns\": " << results.unix_end_ns << ",\n"
      << "  \"max_host_rss_kb\": " << results.max_rss_kb << ",\n"
      << "  \"plugin_path\": \"" << JsonEscape(options.plugin_path) << "\",\n"
      << "  \"output_dir\": \"" << JsonEscape(options.output_dir) << "\",\n"
      << "  \"iterations\": " << options.iterations << ",\n"
      << "  \"warmup_iterations\": " << options.warmup_iterations << ",\n"
      << "  \"compile_iterations\": " << options.compile_iterations << ",\n"
      << "  \"device_count\": " << options.device_count << ",\n"
      << "  \"matrix_size\": " << options.matrix_size << ",\n"
      << "  \"profile_enabled\": " << (options.profile ? "true" : "false")
      << ",\n"
      << "  \"client_create_options\": {},\n"
      << "  \"allocator_baseline\": \"plugin default (BFC, preallocate=true, "
         "memory_fraction=0.75)\",\n"
      << "  \"oneapi_device_selector\": \""
      << JsonEscape(EnvironmentValue("ONEAPI_DEVICE_SELECTOR")) << "\",\n"
      << "  \"sycl_ur_use_level_zero_v2\": \""
      << JsonEscape(EnvironmentValue("SYCL_UR_USE_LEVEL_ZERO_V2"))
      << "\",\n"
      << "  \"ur_loader_use_level_zero_v2\": \""
      << JsonEscape(EnvironmentValue("UR_LOADER_USE_LEVEL_ZERO_V2"))
      << "\",\n"
      << "  \"neo_cache_dir\": \""
      << JsonEscape(EnvironmentValue("NEO_CACHE_DIR")) << "\",\n"
      << "  \"neo_cache_persistent\": \""
      << JsonEscape(EnvironmentValue("NEO_CACHE_PERSISTENT")) << "\",\n"
      << "  \"pjrt_gpu_enable_async_dispatch\": \""
      << JsonEscape(EnvironmentValue("PJRT_GPU_ENABLE_ASYNC_DISPATCH"))
      << "\",\n"
      << "  \"platform_name\": \"" << JsonEscape(results.platform_name)
      << "\",\n"
      << "  \"platform_version\": \""
      << JsonEscape(results.platform_version) << "\",\n"
      << "  \"generated_code_bytes\": " << results.generated_code_bytes
      << ",\n"
      << "  \"input_bytes_per_device\": "
      << static_cast<int64_t>(options.matrix_size) * options.matrix_size *
             sizeof(float) * 2
      << ",\n"
      << "  \"output_bytes_per_device\": "
      << static_cast<int64_t>(options.matrix_size) * options.matrix_size *
             sizeof(float)
      << ",\n";

  out << "  \"devices\": [";
  for (int i = 0; i < results.devices.size(); ++i) {
    if (i != 0) out << ", ";
    out << "\"" << JsonEscape(results.devices[i]) << "\"";
  }
  out << "],\n";

  out << "  \"stages\": [\n";
  for (int i = 0; i < results.stages.size(); ++i) {
    const auto& stage = results.stages[i];
    out << "    {\"name\": \"" << JsonEscape(stage.name)
        << "\", \"start_ns\": " << stage.start_ns
        << ", \"end_ns\": " << stage.end_ns
        << ", \"duration_us\": " << ToUs(stage.end_ns - stage.start_ns)
        << "}" << (i + 1 == results.stages.size() ? "\n" : ",\n");
  }
  out << "  ],\n";

  out << "  \"memory_samples\": [\n";
  for (int i = 0; i < results.memory.size(); ++i) {
    const auto& sample = results.memory[i];
    out << "    {\"stage\": \"" << JsonEscape(sample.stage)
        << "\", \"device_ordinal\": " << sample.device_ordinal
        << ", \"available\": " << (sample.available ? "true" : "false");
    if (sample.available) {
      out << ", \"num_allocs\": " << sample.stats.num_allocs
          << ", \"bytes_in_use\": " << sample.stats.bytes_in_use
          << ", \"peak_bytes_in_use\": " << sample.stats.peak_bytes_in_use
          << ", \"bytes_reserved\": " << sample.stats.bytes_reserved
          << ", \"peak_bytes_reserved\": "
          << sample.stats.peak_bytes_reserved
          << ", \"peak_allocated_bytes\": "
          << sample.stats.peak_allocated_bytes;
    } else {
      out << ", \"error\": \"" << JsonEscape(sample.error) << "\"";
    }
    out << "}" << (i + 1 == results.memory.size() ? "\n" : ",\n");
  }
  out << "  ],\n";

  out << "  \"executions\": [\n";
  for (int i = 0; i < results.executions.size(); ++i) {
    const auto& execution = results.executions[i];
    out << "    {\"execution_id\": " << execution.execution_id
        << ", \"iteration\": " << execution.iteration
        << ", \"phase\": \"" << JsonEscape(execution.phase)
        << "\", \"call_start_ns\": " << execution.call_start_ns
        << ", \"call_return_ns\": " << execution.call_return_ns
        << ", \"completion_ns\": " << execution.completion_ns
        << ", \"execute_api_us\": "
        << ToUs(execution.call_return_ns - execution.call_start_ns)
        << ", \"call_to_completion_us\": "
        << ToUs(execution.completion_ns - execution.call_start_ns)
        << ", \"post_return_to_completion_us\": "
        << ToUs(execution.completion_ns - execution.call_return_ns)
        << ", \"validated\": "
        << (execution.validated ? "true" : "false");
    if (execution.validated) {
      out << ", \"validation_us\": "
          << ToUs(execution.validation_end_ns - execution.validation_start_ns);
    }
    out << ", \"per_device_completion_ns\": [";
    for (int j = 0; j < execution.per_device_completion_ns.size(); ++j) {
      if (j != 0) out << ", ";
      out << execution.per_device_completion_ns[j];
    }
    out << "], \"checksums\": [";
    for (int j = 0; j < execution.checksums.size(); ++j) {
      if (j != 0) out << ", ";
      out << execution.checksums[j];
    }
    out << "], \"max_abs_errors\": [";
    for (int j = 0; j < execution.max_abs_errors.size(); ++j) {
      if (j != 0) out << ", ";
      out << execution.max_abs_errors[j];
    }
    out << "]}" << (i + 1 == results.executions.size() ? "\n" : ",\n");
  }
  out << "  ]\n}\n";

  RETURN_IF_ERROR(WriteTextFile(
      absl::StrCat(options.output_dir, "/metrics.json"), out.str()));

  std::ostringstream csv;
  csv << "execution_id,iteration,phase,execute_api_us,"
         "call_to_completion_us,post_return_to_completion_us,validated,"
         "validation_us\n";
  csv << std::setprecision(17);
  for (const auto& execution : results.executions) {
    csv << execution.execution_id << ',' << execution.iteration << ','
        << execution.phase << ','
        << ToUs(execution.call_return_ns - execution.call_start_ns) << ','
        << ToUs(execution.completion_ns - execution.call_start_ns) << ','
        << ToUs(execution.completion_ns - execution.call_return_ns) << ','
        << (execution.validated ? 1 : 0) << ',';
    if (execution.validated) {
      csv << ToUs(execution.validation_end_ns - execution.validation_start_ns);
    }
    csv << '\n';
  }
  return WriteTextFile(absl::StrCat(options.output_dir, "/metrics.csv"),
                       csv.str());
}

absl::Status Run(const Options& options, Results* results) {
  std::error_code filesystem_error;
  std::filesystem::create_directories(options.output_dir, filesystem_error);
  if (filesystem_error) {
    return absl::InternalError(absl::StrCat(
        "Unable to create output directory: ", filesystem_error.message()));
  }

  results->program_start_ns = NowNs();
  results->unix_start_ns = UnixNowNs();
  const int64_t client_total_start = NowNs();

  const int64_t plugin_load_start = NowNs();
  ASSIGN_OR_RETURN(const PJRT_Api* api,
                   pjrt::LoadPjrtPlugin("oneapi", options.plugin_path));
  const int64_t plugin_load_end = NowNs();
  results->stages.push_back(
      {"plugin_load", plugin_load_start, plugin_load_end});

  const int64_t plugin_init_start = NowNs();
  RETURN_IF_ERROR(pjrt::InitializePjrtPlugin("oneapi"));
  const int64_t plugin_init_end = NowNs();
  results->stages.push_back(
      {"plugin_initialize", plugin_init_start, plugin_init_end});

  std::unique_ptr<tsl::ProfilerSession> profiler_session;
  if (options.profile) {
    xla::RegisterProfiler(api);
    tensorflow::ProfileOptions profile_options =
        tsl::ProfilerSession::DefaultOptions();
    profile_options.set_host_tracer_level(3);
    profile_options.set_device_tracer_level(1);
    profile_options.set_enable_hlo_proto(true);
    profiler_session = tsl::ProfilerSession::Create(profile_options);
    RETURN_IF_ERROR(profiler_session->Status());
  }

  std::unique_ptr<PjRtClient> client;
  std::unique_ptr<PjRtLoadedExecutable> executable;
  std::vector<std::unique_ptr<PjRtBuffer>> lhs_buffers;
  std::vector<std::unique_ptr<PjRtBuffer>> rhs_buffers;

  {
    const int64_t start = NowNs();
    tsl::profiler::TraceMe trace("reproducer::PJRT_Client_Create");
    ASSIGN_OR_RETURN(client, GetCApiClient("oneapi"));
    const int64_t end = NowNs();
    results->stages.push_back({"client_create", start, end});
    results->stages.push_back(
        {"client_initialization_total", client_total_start, end});
  }

  results->platform_name = std::string(client->platform_name());
  results->platform_version = std::string(client->platform_version());
  if (client->addressable_device_count() < options.device_count) {
    return absl::FailedPreconditionError(absl::StrCat(
        "Requested ", options.device_count, " devices, but only ",
        client->addressable_device_count(), " are addressable"));
  }
  absl::Span<PjRtDevice* const> devices =
      client->addressable_devices().subspan(0, options.device_count);
  for (PjRtDevice* device : devices) {
    results->devices.push_back(std::string(device->DebugString()));
  }
  RecordMemory(devices, "after_client_create", results);

  XlaComputation computation;
  Shape matrix_shape;
  {
    const int64_t start = NowNs();
    tsl::profiler::TraceMe trace([&] {
      return tsl::profiler::TraceMeEncode(
          "reproducer::BuildHLO",
          {{"matrix_size", options.matrix_size},
           {"operation", "dot_broadcast_add"}});
    });
    matrix_shape = ShapeUtil::MakeShape(
        F32, {options.matrix_size, options.matrix_size});
    XlaBuilder builder("pjrt_oneapi_matmul_add");
    XlaOp lhs = Parameter(&builder, 0, matrix_shape, "lhs");
    XlaOp rhs = Parameter(&builder, 1, matrix_shape, "rhs");
    XlaOp dot = Dot(lhs, rhs);
    XlaOp one = Broadcast(ConstantR0<float>(&builder, 1.0f),
                          {options.matrix_size, options.matrix_size});
    ASSIGN_OR_RETURN(computation, builder.Build(Add(dot, one)));
    const int64_t end = NowNs();
    results->stages.push_back({"hlo_build", start, end});
  }

  CompileOptions compile_options;
  compile_options.executable_build_options.set_num_replicas(
      options.device_count);
  compile_options.executable_build_options.set_num_partitions(1);
  compile_options.executable_build_options.set_use_spmd_partitioning(false);
  DeviceAssignment assignment(options.device_count, 1);
  for (int i = 0; i < options.device_count; ++i) {
    assignment(i, 0) = devices[i]->global_device_id().value();
  }
  compile_options.executable_build_options.set_device_assignment(assignment);

  for (int compile_iteration = 0;
       compile_iteration < options.compile_iterations; ++compile_iteration) {
    const int64_t compile_id = compile_iteration + 1;
    const int64_t start = NowNs();
    {
      tsl::profiler::TraceMe trace([&] {
        return tsl::profiler::TraceMeEncode(
            "reproducer::PJRT_Compile",
            {{"compile_id", compile_id},
             {"device_count", options.device_count},
             {"matrix_size", options.matrix_size}});
      });
      ASSIGN_OR_RETURN(executable,
                       client->CompileAndLoad(computation, compile_options));
    }
    const int64_t end = NowNs();
    results->stages.push_back(
        {absl::StrCat("pjrt_compile_", compile_id), start, end});
  }
  results->generated_code_bytes = executable->SizeOfGeneratedCodeInBytes();

  const int64_t element_count =
      static_cast<int64_t>(options.matrix_size) * options.matrix_size;
  std::vector<float> host_lhs(element_count, 1.0f);
  std::vector<float> host_rhs(element_count, 1.0f);
  std::vector<std::vector<PjRtBuffer*>> argument_lists(options.device_count);
  lhs_buffers.reserve(options.device_count);
  rhs_buffers.reserve(options.device_count);

  {
    const int64_t start = NowNs();
    tsl::profiler::TraceMe trace([&] {
      return tsl::profiler::TraceMeEncode(
          "reproducer::InputBufferSetup",
          {{"device_count", options.device_count},
           {"bytes_per_device", element_count * sizeof(float) * 2}});
    });
    for (int i = 0; i < options.device_count; ++i) {
      ASSIGN_OR_RETURN(PjRtMemorySpace * memory,
                       devices[i]->default_memory_space());
      ASSIGN_OR_RETURN(
          std::unique_ptr<PjRtBuffer> lhs,
          client->BufferFromHostBuffer(
              host_lhs.data(), F32, matrix_shape.dimensions(), std::nullopt,
              PjRtClient::HostBufferSemantics::kImmutableOnlyDuringCall,
              nullptr, memory, /*device_layout=*/nullptr));
      ASSIGN_OR_RETURN(
          std::unique_ptr<PjRtBuffer> rhs,
          client->BufferFromHostBuffer(
              host_rhs.data(), F32, matrix_shape.dimensions(), std::nullopt,
              PjRtClient::HostBufferSemantics::kImmutableOnlyDuringCall,
              nullptr, memory, /*device_layout=*/nullptr));
      RETURN_IF_ERROR(lhs->GetReadyFuture().Await());
      RETURN_IF_ERROR(rhs->GetReadyFuture().Await());
      lhs_buffers.push_back(std::move(lhs));
      rhs_buffers.push_back(std::move(rhs));
      argument_lists[i] = {lhs_buffers.back().get(), rhs_buffers.back().get()};
    }
    const int64_t end = NowNs();
    results->stages.push_back({"input_buffer_setup", start, end});
  }
  RecordMemory(devices, "after_input_buffers", results);

  ExecuteOptions execute_options;
  execute_options.non_donatable_input_indices = {0, 1};
  const int total_iterations =
      options.warmup_iterations + options.iterations;
  for (int iteration = 0; iteration < total_iterations; ++iteration) {
    ExecutionSample sample;
    sample.execution_id = iteration + 1;
    sample.iteration = iteration;
    sample.phase = iteration < options.warmup_iterations ? "warmup" : "measured";
    std::optional<std::vector<Future<>>> completion_futures(std::in_place);
    absl::StatusOr<std::vector<std::vector<std::unique_ptr<PjRtBuffer>>>>
        outputs_or = absl::UnknownError("PJRT Execute not called");

    sample.call_start_ns = NowNs();
    {
      tsl::profiler::TraceMe total_trace([&] {
        return tsl::profiler::TraceMeEncode(
            "reproducer::ExecutionToHostObservedCompletion",
            {{"execution_id", sample.execution_id},
             {"iteration", sample.iteration},
             {"phase", sample.phase},
             {"device_count", options.device_count}});
      });
      {
        tsl::profiler::TraceMe api_trace([&] {
          return tsl::profiler::TraceMeEncode(
              "reproducer::PJRT_Execute_API",
              {{"execution_id", sample.execution_id},
               {"iteration", sample.iteration},
               {"phase", sample.phase}});
        });
        outputs_or = executable->Execute(absl::MakeConstSpan(argument_lists),
                                         execute_options, completion_futures);
      }
      sample.call_return_ns = NowNs();
      if (!outputs_or.ok()) return outputs_or.status();
      if (!completion_futures.has_value() ||
          completion_futures->size() != options.device_count) {
        return absl::InternalError(
            "PJRT Execute did not return one completion future per device");
      }
      {
        tsl::profiler::TraceMe wait_trace([&] {
          return tsl::profiler::TraceMeEncode(
              "reproducer::PJRT_CompletionFutureAwait",
              {{"execution_id", sample.execution_id},
               {"device_count", options.device_count}});
        });
        for (Future<>& future : *completion_futures) {
          RETURN_IF_ERROR(future.Await());
          sample.per_device_completion_ns.push_back(NowNs());
        }
      }
      sample.completion_ns = NowNs();
    }

    auto outputs = std::move(*outputs_or);
    if (outputs.size() != options.device_count) {
      return absl::InternalError("Unexpected number of device output lists");
    }

    const bool validate = iteration == total_iterations - 1;
    if (validate) {
      sample.validation_start_ns = NowNs();
      tsl::profiler::TraceMe validation_trace([&] {
        return tsl::profiler::TraceMeEncode(
            "reproducer::D2HAndValidation",
            {{"execution_id", sample.execution_id},
             {"device_count", options.device_count}});
      });
      const double expected_value = options.matrix_size + 1.0;
      const double expected_checksum = expected_value * element_count;
      for (int device = 0; device < options.device_count; ++device) {
        if (outputs[device].size() != 1) {
          return absl::InternalError("Expected one output per device");
        }
        ASSIGN_OR_RETURN(std::shared_ptr<Literal> literal,
                         outputs[device][0]->ToLiteral().Await());
        absl::Span<const float> values = literal->data<float>();
        if (values.size() != element_count) {
          return absl::InternalError("Output element count mismatch");
        }
        double checksum = 0.0;
        double max_abs_error = 0.0;
        for (float value : values) {
          checksum += value;
          max_abs_error =
              std::max(max_abs_error, std::abs(value - expected_value));
        }
        sample.checksums.push_back(checksum);
        sample.max_abs_errors.push_back(max_abs_error);
        const double checksum_tolerance =
            std::max(1e-3, std::abs(expected_checksum) * 1e-5);
        if (std::abs(checksum - expected_checksum) > checksum_tolerance ||
            max_abs_error > 1e-3) {
          return absl::InternalError(absl::StrCat(
              "Validation failed on device ", device, ": checksum=", checksum,
              " expected=", expected_checksum,
              " max_abs_error=", max_abs_error));
        }
      }
      sample.validation_end_ns = NowNs();
      sample.validated = true;
    }

    std::cout << std::setprecision(17)
              << "PJRT_METRIC {\"execution_id\":" << sample.execution_id
              << ",\"iteration\":" << sample.iteration << ",\"phase\":\""
              << sample.phase << "\",\"execute_api_us\":"
              << ToUs(sample.call_return_ns - sample.call_start_ns)
              << ",\"call_to_completion_us\":"
              << ToUs(sample.completion_ns - sample.call_start_ns)
              << ",\"post_return_to_completion_us\":"
              << ToUs(sample.completion_ns - sample.call_return_ns) << "}\n";
    results->executions.push_back(std::move(sample));
  }
  RecordMemory(devices, "after_executions", results);

  {
    const int64_t start = NowNs();
    tsl::profiler::TraceMe trace("reproducer::PJRT_Teardown");
    lhs_buffers.clear();
    rhs_buffers.clear();
    executable.reset();
    client.reset();
    const int64_t end = NowNs();
    results->stages.push_back({"pjrt_teardown", start, end});
  }

  if (options.profile) {
    tensorflow::profiler::XSpace xspace;
    const int64_t collect_start = NowNs();
    RETURN_IF_ERROR(profiler_session->CollectData(&xspace));
    const int64_t collect_end = NowNs();
    results->stages.push_back(
        {"profiler_collect", collect_start, collect_end});
    const std::string xspace_path =
        absl::StrCat(options.output_dir, "/trace.xplane.pb");
    RETURN_IF_ERROR(WriteTextFile(xspace_path, xspace.SerializeAsString()));
    tsl::profiler::TraceContainer trace_container =
        tsl::profiler::ConvertXSpaceToTraceContainer(xspace);
    std::string trace_json =
        tsl::profiler::TraceContainerToJson(trace_container);
    RETURN_IF_ERROR(WriteTextFile(
        absl::StrCat(options.output_dir, "/trace.json"), trace_json));
  }

  struct rusage usage {};
  if (getrusage(RUSAGE_SELF, &usage) == 0) {
    results->max_rss_kb = usage.ru_maxrss;
  }
  results->program_end_ns = NowNs();
  results->unix_end_ns = UnixNowNs();
  RETURN_IF_ERROR(WriteMetrics(options, *results));
  return absl::OkStatus();
}

}  // namespace
}  // namespace xla

int main(int argc, char** argv) {
  auto options = xla::ParseOptions(argc, argv);
  if (!options.ok()) {
    std::cerr << "ERROR: " << options.status() << '\n';
    return 2;
  }
  xla::Results results;
  absl::Status status = xla::Run(*options, &results);
  if (!status.ok()) {
    std::cerr << "ERROR: " << status << '\n';
    std::error_code error;
    std::filesystem::create_directories(options->output_dir, error);
    std::ofstream failure(options->output_dir + "/failure.txt",
                          std::ios::trunc);
    failure << status << '\n';
    return 1;
  }
  std::cout << "PJRT_RESULT {\"status\":\"ok\",\"metrics\":\""
            << xla::JsonEscape(options->output_dir + "/metrics.json")
            << "\",\"trace\":\""
            << xla::JsonEscape(options->profile
                                   ? options->output_dir + "/trace.json"
                                   : "")
            << "\"}\n";
  return 0;
}
