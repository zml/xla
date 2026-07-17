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

#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/literal.h"
#include "xla/pjrt/gpu/se_gpu_pjrt_client.h"
#include "xla/pjrt/maybe_owning_mlir_module.h"
#include "xla/pjrt/mlir_to_hlo.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/plugin/xla_gpu/xla_gpu_allocator_config.h"
#include "xla/pjrt/plugin/xla_gpu/xla_gpu_client_options.h"
#include "xla/tsl/profiler/convert/trace_events_to_json.h"
#include "xla/tsl/profiler/convert/xplane_to_trace_events.h"
#include "xla/xla_data.pb.h"
#include "tsl/profiler/lib/profiler_session.h"
#include "tsl/profiler/lib/traceme.h"
#include "tsl/profiler/protobuf/profiler_options.pb.h"
#include "tsl/profiler/protobuf/xplane.pb.h"

namespace xla {
namespace {

using Clock = std::chrono::steady_clock;

constexpr int64_t kElements = 1 << 20;
constexpr std::array<int64_t, 1> kDimensions = {kElements};

constexpr absl::string_view kProgram = R"mlir(
module {
  func.func @main(%arg0: tensor<1048576xf32>) -> tensor<1048576xf32> {
    %0 = stablehlo.add %arg0, %arg0 : tensor<1048576xf32>
    return %0 : tensor<1048576xf32>
  }
}
)mlir";

const std::vector<float> kInput(kElements, 21.0f);

double Milliseconds(Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start)
      .count();
}

struct StageResult {
  std::string name;
  double duration_ms = -1.0;
  std::string status = "NOT_RUN";
};

struct Metrics {
  StageResult client_create{"client_create"};
  StageResult parse_mlir{"parse_mlir"};
  StageResult compile_and_load{"compile_and_load"};
  StageResult buffer_from_host{"buffer_from_host"};
  StageResult execute_1{"execute_1"};
  StageResult execute_2{"execute_2"};
  StageResult copy_to_host{"copy_to_host"};
  StageResult validate_output{"validate_output"};
  StageResult total{"total"};
};

void FinishStage(StageResult* stage, Clock::time_point start,
                 const absl::Status& status) {
  stage->duration_ms = Milliseconds(start);
  stage->status = status.ok() ? "OK" : status.ToString();
}

std::string CsvEscape(absl::string_view value) {
  std::string escaped;
  escaped.reserve(value.size() + 2);
  escaped.push_back('"');
  for (char c : value) {
    if (c == '"') {
      escaped.append("\"\"");
    } else if (c == '\n' || c == '\r') {
      escaped.push_back(' ');
    } else {
      escaped.push_back(c);
    }
  }
  escaped.push_back('"');
  return escaped;
}

absl::Status WriteCsv(const std::string& path, const Metrics& metrics) {
  std::ofstream output(path, std::ios::trunc);
  if (!output) {
    return absl::InternalError(absl::StrCat("Could not open CSV file: ", path));
  }

  const std::array<const StageResult*, 9> stages = {
      &metrics.client_create,    &metrics.parse_mlir,
      &metrics.compile_and_load, &metrics.buffer_from_host,
      &metrics.execute_1,        &metrics.execute_2,
      &metrics.copy_to_host,     &metrics.validate_output,
      &metrics.total,
  };

  output << "stage,duration_ms,status\n";
  output << std::fixed << std::setprecision(3);
  for (const StageResult* stage : stages) {
    output << stage->name << ',' << stage->duration_ms << ','
           << CsvEscape(stage->status) << '\n';
  }

  if (!output) {
    return absl::InternalError(absl::StrCat("Could not write CSV file: ", path));
  }
  return absl::OkStatus();
}

class HostTrace {
 public:
  static absl::StatusOr<std::unique_ptr<HostTrace>> Start(
      std::string output_path) {
    tensorflow::ProfileOptions options =
        tsl::ProfilerSession::DefaultOptions();
    options.set_host_tracer_level(3);
    options.set_device_tracer_level(0);
    options.set_enable_hlo_proto(false);

    std::unique_ptr<tsl::ProfilerSession> session =
        tsl::ProfilerSession::Create(options);
    if (!session->Status().ok()) {
      return session->Status();
    }

    return std::unique_ptr<HostTrace>(
        new HostTrace(std::move(output_path), std::move(session)));
  }

  absl::Status Stop() {
    if (session_ == nullptr) {
      return absl::OkStatus();
    }

    tensorflow::profiler::XSpace xspace;
    absl::Status status = session_->CollectData(&xspace);
    session_.reset();
    if (!status.ok()) {
      return status;
    }

    tsl::profiler::TraceContainer trace =
        tsl::profiler::ConvertXSpaceToTraceContainer(xspace);

    std::ofstream output(output_path_,
                         std::ios::binary | std::ios::trunc);
    if (!output) {
      return absl::InternalError(
          absl::StrCat("Could not open trace file: ", output_path_));
    }

    output << tsl::profiler::TraceContainerToJson(trace);
    if (!output) {
      return absl::InternalError(
          absl::StrCat("Could not write trace file: ", output_path_));
    }

    return absl::OkStatus();
  }

 private:
  HostTrace(std::string output_path,
            std::unique_ptr<tsl::ProfilerSession> session)
      : output_path_(std::move(output_path)),
        session_(std::move(session)) {}

  std::string output_path_;
  std::unique_ptr<tsl::ProfilerSession> session_;
};

absl::Status ExecuteAndWait(const char* trace_name,
                            PjRtLoadedExecutable& executable,
                            PjRtBuffer& input, PjRtDevice& device,
                            std::unique_ptr<PjRtBuffer>* output) {
  tsl::profiler::TraceMe trace(trace_name);

  std::array<PjRtBuffer*, 1> arguments = {&input};
  auto result =
      executable.ExecuteSharded(arguments, &device, ExecuteOptions());
  if (!result.ok()) {
    return result.status();
  }
  if (result->size() != 1) {
    return absl::InternalError(
        absl::StrCat("Expected one output buffer, got ", result->size()));
  }

  absl::Status ready_status = result->front()->GetReadyFuture().Await();
  if (!ready_status.ok()) {
    return ready_status;
  }

  *output = std::move(result->front());
  return absl::OkStatus();
}

absl::Status ValidateOutput(const Literal& literal) {
  for (float value : literal.data<float>()) {
    if (value != 42.0f) {
      return absl::DataLossError(
          absl::StrCat("Unexpected output value: ", value));
    }
  }
  return absl::OkStatus();
}

absl::Status RunReproducer(Metrics* metrics) {
  GpuClientOptions client_options;
  client_options.allowed_devices = {0};
  client_options.allocator_config.kind =
      GpuAllocatorConfig::Kind::kPlatform;

  const Clock::time_point client_start = Clock::now();
  auto client_or = [&]() {
    tsl::profiler::TraceMe trace("CLIENT_CREATE");
    return GetStreamExecutorGpuClient(client_options);
  }();
  FinishStage(&metrics->client_create, client_start, client_or.status());
  if (!client_or.ok()) {
    return client_or.status();
  }
  std::unique_ptr<PjRtClient> client = std::move(*client_or);

  if (client->addressable_devices().empty()) {
    return absl::NotFoundError("No addressable oneAPI device was found");
  }
  PjRtDevice* const device = client->addressable_devices().front();

  auto context = std::make_unique<mlir::MLIRContext>();

  const Clock::time_point parse_start = Clock::now();
  auto module_or = [&]() {
    tsl::profiler::TraceMe trace("PARSE_MLIR");
    return ParseMlirModuleString(kProgram, *context);
  }();
  FinishStage(&metrics->parse_mlir, parse_start, module_or.status());
  if (!module_or.ok()) {
    return module_or.status();
  }

  const Clock::time_point compile_start = Clock::now();
  auto executable_or = [&]() {
    tsl::profiler::TraceMe trace("COMPILE_AND_LOAD");
    return client->CompileAndLoad(
        MaybeOwningMlirModule(std::move(context), std::move(*module_or)),
        CompileOptions());
  }();
  FinishStage(&metrics->compile_and_load, compile_start,
              executable_or.status());
  if (!executable_or.ok()) {
    return executable_or.status();
  }
  std::unique_ptr<PjRtLoadedExecutable> executable =
      std::move(*executable_or);

  const Clock::time_point buffer_start = Clock::now();
  auto input_or = [&]() -> absl::StatusOr<std::unique_ptr<PjRtBuffer>> {
    tsl::profiler::TraceMe trace("BUFFER_FROM_HOST");

    auto memory_space_or = device->default_memory_space();
    if (!memory_space_or.ok()) {
      return memory_space_or.status();
    }

    return client->BufferFromHostBuffer(
        kInput.data(), F32, kDimensions, std::nullopt,
        PjRtClient::HostBufferSemantics::kImmutableOnlyDuringCall,
        /*on_done_with_host_buffer=*/nullptr, *memory_space_or,
        /*device_layout=*/nullptr);
  }();
  FinishStage(&metrics->buffer_from_host, buffer_start, input_or.status());
  if (!input_or.ok()) {
    return input_or.status();
  }
  std::unique_ptr<PjRtBuffer> input = std::move(*input_or);

  std::unique_ptr<PjRtBuffer> output;
  {
    tsl::profiler::TraceMe execution_trace("EXECUTION");

    const Clock::time_point execute_1_start = Clock::now();
    absl::Status execute_1_status =
        ExecuteAndWait("RUN 1", *executable, *input, *device, &output);
    FinishStage(&metrics->execute_1, execute_1_start, execute_1_status);
    if (!execute_1_status.ok()) {
      return execute_1_status;
    }

    const Clock::time_point execute_2_start = Clock::now();
    absl::Status execute_2_status =
        ExecuteAndWait("RUN 2", *executable, *input, *device, &output);
    FinishStage(&metrics->execute_2, execute_2_start, execute_2_status);
    if (!execute_2_status.ok()) {
      return execute_2_status;
    }
  }

  const Clock::time_point copy_start = Clock::now();
  auto literal_or = [&]() {
    tsl::profiler::TraceMe trace("COPY_TO_HOST");
    return output->ToLiteral().Await();
  }();
  FinishStage(&metrics->copy_to_host, copy_start, literal_or.status());
  if (!literal_or.ok()) {
    return literal_or.status();
  }

  const Clock::time_point validate_start = Clock::now();
  absl::Status validation_status;
  {
    tsl::profiler::TraceMe trace("VALIDATE_OUTPUT");
    validation_status = ValidateOutput(**literal_or);
  }
  FinishStage(&metrics->validate_output, validate_start, validation_status);
  return validation_status;
}

std::string DefaultTracePath() {
  const char* environment_path = std::getenv("XLA_HOST_TRACE_FILE");
  if (environment_path != nullptr && *environment_path != '\0') {
    return environment_path;
  }
  return "pjrt_oneapi_trace.json";
}

}  // namespace
}  // namespace xla

int main(int argc, char** argv) {
  using namespace xla;

  const std::string trace_path =
      argc >= 2 ? argv[1] : DefaultTracePath();
  const std::string csv_path =
      argc >= 3 ? argv[2] : absl::StrCat(trace_path, ".csv");

  auto profiler_or = HostTrace::Start(trace_path);
  if (!profiler_or.ok()) {
    std::cerr << "Could not start profiler: "
              << profiler_or.status() << '\n';
    return 1;
  }
  std::unique_ptr<HostTrace> profiler = std::move(*profiler_or);

  Metrics metrics;
  const Clock::time_point total_start = Clock::now();

  absl::Status run_status;
  {
    tsl::profiler::TraceMe trace("PJRT_ONEAPI_REPRODUCER");
    run_status = RunReproducer(&metrics);
  }

  FinishStage(&metrics.total, total_start, run_status);

  // Stop explicitly so the trace is written even when the reproducer returns
  // the pre-fix execution error.
  const absl::Status trace_status = profiler->Stop();
  const absl::Status csv_status = WriteCsv(csv_path, metrics);

  std::cout << "TRACE_FILE=" << trace_path << '\n'
            << "CSV_FILE=" << csv_path << '\n'
            << "RUN_STATUS=" << run_status << '\n';

  if (!trace_status.ok()) {
    std::cerr << "Could not save trace: " << trace_status << '\n';
  }
  if (!csv_status.ok()) {
    std::cerr << "Could not save CSV: " << csv_status << '\n';
  }

  return run_status.ok() && trace_status.ok() && csv_status.ok() ? 0 : 1;
}
