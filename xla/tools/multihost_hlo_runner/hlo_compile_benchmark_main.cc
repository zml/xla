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

// Repeatedly compiles captured HLO modules without executing them. This keeps
// model loading and input preparation out of compiler performance experiments.

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <ostream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "xla/debug_options_flags.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/plugin/xla_gpu/xla_gpu_allocator_config.h"
#include "xla/pjrt/plugin/xla_gpu/xla_gpu_client_options.h"
#include "xla/tools/multihost_hlo_runner/create_client.h"
#include "xla/tools/multihost_hlo_runner/functional_hlo_runner.h"
#include "xla/tools/multihost_hlo_runner/hlo_input_output_format.h"
#include "xla/tsl/util/command_line_flags.h"
#include "tsl/platform/init_main.h"

namespace xla {
namespace {

constexpr absl::string_view kUsage = R"(
Repeatedly compiles one or more captured HLO modules without executing them.

Example:

  bazel run //xla/tools/multihost_hlo_runner:hlo_compile_benchmark_gpu -- \
    --repetitions=5 --parallelism=1 --output_csv=/tmp/compile.csv \
    /tmp/hlo/module_0011.before_optimizations.txt

Use --parallelism=1 to isolate each module. Set --parallelism to the number of
modules to reproduce concurrent compilation. Repetitions share one PJRT client
and process; invoke the binary repeatedly to measure fully cold processes.
)";

struct BenchmarkOptions {
  int32_t repetitions = 5;
  int32_t warmup_repetitions = 0;
  int32_t parallelism = 1;
  std::string output_csv;
  bool append_csv = false;
  std::string label;
  std::string run_id;
  std::string input_format = "text";
  float gpu_client_mem_fraction =
      xla::GpuAllocatorConfig{}.memory_fraction;
  int64_t gpu_client_initialization_timeout_sec = 300;
};

struct CompileMeasurement {
  std::string hlo_file;
  double parse_ms = 0.0;
  double compile_ms = 0.0;
  absl::Status status = absl::UnknownError("Compilation did not run");
  std::unique_ptr<PjRtLoadedExecutable> executable;
};

struct BatchMeasurement {
  int32_t iteration = 0;
  bool warmup = false;
  double wall_ms = 0.0;
  std::vector<CompileMeasurement> modules;
};

struct SummaryStats {
  size_t count;
  double minimum;
  double median;
  double mean;
  double p95;
  double maximum;
  double stddev;
};

double Milliseconds(absl::Duration duration) {
  return absl::ToDoubleMilliseconds(duration);
}

std::string CsvEscape(absl::string_view value) {
  if (value.find_first_of(",\"\n\r") == absl::string_view::npos) {
    return std::string(value);
  }
  std::string escaped = "\"";
  for (char character : value) {
    if (character == '"') {
      escaped += "\"\"";
    } else {
      escaped += character;
    }
  }
  escaped += '"';
  return escaped;
}

SummaryStats Summarize(std::vector<double> values) {
  std::sort(values.begin(), values.end());
  const double sum = std::accumulate(values.begin(), values.end(), 0.0);
  const double mean = sum / values.size();
  double squared_deviation_sum = 0.0;
  for (double value : values) {
    const double deviation = value - mean;
    squared_deviation_sum += deviation * deviation;
  }

  double median;
  if (values.size() % 2 == 0) {
    const size_t upper = values.size() / 2;
    median = (values[upper - 1] + values[upper]) / 2.0;
  } else {
    median = values[values.size() / 2];
  }
  const size_t p95_index =
      std::min(values.size() - 1,
               static_cast<size_t>(std::ceil(values.size() * 0.95)) - 1);
  return SummaryStats{
      .count = values.size(),
      .minimum = values.front(),
      .median = median,
      .mean = mean,
      .p95 = values[p95_index],
      .maximum = values.back(),
      .stddev = std::sqrt(squared_deviation_sum / values.size()),
  };
}

CompileMeasurement CompileOne(
    PjRtClient& client,
    const FunctionalHloRunner::PreprocessingOptions& preprocessing_options,
    const CompileOptions& compile_options, absl::string_view hlo_file,
    InputFormat input_format) {
  CompileMeasurement measurement;
  measurement.hlo_file = hlo_file;

  const absl::Time parse_start = absl::Now();
  absl::StatusOr<FunctionalHloRunner::HloModuleAndArguments> loaded =
      FunctionalHloRunner::LoadHloModuleAndArguments(hlo_file, input_format);
  measurement.parse_ms = Milliseconds(absl::Now() - parse_start);
  if (!loaded.ok()) {
    measurement.status = loaded.status();
    return measurement;
  }

  const absl::Time compile_start = absl::Now();
  absl::StatusOr<std::unique_ptr<PjRtLoadedExecutable>> executable =
      FunctionalHloRunner::Compile(client, loaded->hlo_module.get(),
                                   preprocessing_options, compile_options);
  measurement.compile_ms = Milliseconds(absl::Now() - compile_start);
  if (!executable.ok()) {
    measurement.status = executable.status();
    return measurement;
  }

  measurement.status = absl::OkStatus();
  measurement.executable = std::move(*executable);
  return measurement;
}

BatchMeasurement CompileBatch(
    PjRtClient& client,
    const FunctionalHloRunner::PreprocessingOptions& preprocessing_options,
    const CompileOptions& compile_options,
    const std::vector<std::string>& hlo_files, InputFormat input_format,
    int32_t parallelism, int32_t iteration, bool warmup) {
  BatchMeasurement batch;
  batch.iteration = iteration;
  batch.warmup = warmup;
  batch.modules.resize(hlo_files.size());

  const absl::Time batch_start = absl::Now();
  std::atomic<size_t> next_module = 0;
  const size_t worker_count =
      std::min(static_cast<size_t>(parallelism), hlo_files.size());
  std::vector<std::thread> workers;
  workers.reserve(worker_count);
  for (size_t worker = 0; worker < worker_count; ++worker) {
    workers.emplace_back([&] {
      while (true) {
        const size_t module_index = next_module.fetch_add(1);
        if (module_index >= hlo_files.size()) {
          return;
        }
        CompileMeasurement measurement = CompileOne(
            client, preprocessing_options, compile_options,
            hlo_files[module_index], input_format);
        batch.modules[module_index] = std::move(measurement);
      }
    });
  }
  for (std::thread& worker : workers) {
    worker.join();
  }
  batch.wall_ms = Milliseconds(absl::Now() - batch_start);
  return batch;
}

void WriteIterationRows(const BatchMeasurement& batch,
                        const BenchmarkOptions& options,
                        std::ostream& output) {
  for (const CompileMeasurement& module : batch.modules) {
    output << "module," << CsvEscape(options.label) << ','
           << CsvEscape(options.run_id) << ','
           << (batch.warmup ? "warmup" : "measurement") << ','
           << batch.iteration << ',' << CsvEscape(module.hlo_file) << ','
           << module.parse_ms << ',' << module.compile_ms << ",,,,,,,,,"
           << CsvEscape(module.status.ToString()) << '\n';
  }
  output << "batch," << CsvEscape(options.label) << ','
         << CsvEscape(options.run_id) << ','
         << (batch.warmup ? "warmup" : "measurement") << ','
         << batch.iteration << ",__batch__,,," << batch.wall_ms
         << ",,,,,,,,OK\n";
}

void WriteSummaryRow(absl::string_view name, const SummaryStats& stats,
                     const BenchmarkOptions& options, std::ostream& output) {
  output << "summary," << CsvEscape(options.label) << ','
         << CsvEscape(options.run_id) << ",measurement,," << CsvEscape(name)
         << ",,,,"
         << stats.count << ',' << stats.minimum << ',' << stats.median << ','
         << stats.mean << ',' << stats.p95 << ',' << stats.maximum << ','
         << stats.stddev << ",OK\n";
}

void PrintSummary(absl::string_view name, const SummaryStats& stats) {
  std::cout << std::fixed << std::setprecision(1) << name << ": n="
            << stats.count << " min=" << stats.minimum
            << "ms median=" << stats.median << "ms mean=" << stats.mean
            << "ms p95=" << stats.p95 << "ms max=" << stats.maximum
            << "ms stddev=" << stats.stddev << "ms\n";
}

absl::Status RunBenchmark(const BenchmarkOptions& options,
                          const std::vector<std::string>& hlo_files) {
  if (options.repetitions < 1) {
    return absl::InvalidArgumentError("--repetitions must be at least 1");
  }
  if (options.warmup_repetitions < 0) {
    return absl::InvalidArgumentError(
        "--warmup_repetitions must not be negative");
  }
  if (options.parallelism < 1) {
    return absl::InvalidArgumentError("--parallelism must be at least 1");
  }
  if (hlo_files.empty()) {
    return absl::InvalidArgumentError("At least one HLO file is required");
  }
  if (options.gpu_client_mem_fraction <= 0.0 ||
      options.gpu_client_mem_fraction >= 1.0) {
    return absl::InvalidArgumentError(
        "--gpu_client_mem_fraction must be between 0 and 1");
  }

  InputFormat input_format;
  std::string input_format_error;
  if (!AbslParseFlag(options.input_format, &input_format,
                     &input_format_error)) {
    return absl::InvalidArgumentError(input_format_error);
  }

  std::ofstream output_file;
  std::ostream* csv = &std::cout;
  bool write_csv_header = true;
  if (!options.output_csv.empty()) {
    if (options.append_csv) {
      std::ifstream existing_output(options.output_csv);
      write_csv_header =
          !existing_output.good() || existing_output.peek() == EOF;
    }
    output_file.open(options.output_csv,
                     std::ios::out | (options.append_csv ? std::ios::app
                                                        : std::ios::trunc));
    if (!output_file.is_open()) {
      return absl::InternalError(
          absl::StrCat("Could not open ", options.output_csv));
    }
    csv = &output_file;
  }
  if (write_csv_header) {
    *csv << "record,label,run_id,phase,iteration,hlo_file,parse_ms,"
            "compile_ms,batch_ms,count,min_ms,median_ms,mean_ms,p95_ms,"
            "max_ms,stddev_ms,status\n";
  }
  *csv << std::fixed << std::setprecision(3);

  GpuClientOptions gpu_options;
  gpu_options.allocator_config.memory_fraction =
      options.gpu_client_mem_fraction;
  const absl::Time client_start = absl::Now();
  ABSL_ASSIGN_OR_RETURN(
      PjRtEnvironment environment,
      GetPjRtEnvironmentForGpu(
          /*distributed_service_address=*/"", gpu_options,
          absl::Seconds(options.gpu_client_initialization_timeout_sec)));
  const double client_init_ms = Milliseconds(absl::Now() - client_start);
  if (environment.client == nullptr) {
    return absl::InternalError("GPU PJRT client initialization returned null");
  }

  FunctionalHloRunner::PreprocessingOptions preprocessing_options;
  preprocessing_options.remove_infeed_outfeed = false;
  FunctionalHloRunner::RawCompileOptions raw_compile_options;
  raw_compile_options.hlo_passes_mode =
      FunctionalHloRunner::HloPassesMode::kStandardCompile;
  raw_compile_options.spmd_mode =
      FunctionalHloRunner::SpmdMode::kNotUseSpmdPartitioning;
  raw_compile_options.debug_options = GetDebugOptionsFromFlags();
  ABSL_ASSIGN_OR_RETURN(
      CompileOptions compile_options,
      FunctionalHloRunner::CreateCompileOptions(*environment.client,
                                                raw_compile_options));

  std::cout << "GPU client initialized in " << std::fixed
            << std::setprecision(1) << client_init_ms << "ms; compiling "
            << hlo_files.size() << " module(s), repetitions="
            << options.repetitions << ", parallelism=" << options.parallelism
            << '\n';

  std::vector<BatchMeasurement> measurements;
  measurements.reserve(options.repetitions);
  const int32_t total_repetitions =
      options.warmup_repetitions + options.repetitions;
  for (int32_t repetition = 0; repetition < total_repetitions; ++repetition) {
    const bool warmup = repetition < options.warmup_repetitions;
    const int32_t visible_iteration =
        warmup ? repetition : repetition - options.warmup_repetitions;
    BatchMeasurement batch = CompileBatch(
        *environment.client, preprocessing_options, compile_options, hlo_files,
        input_format, options.parallelism, visible_iteration, warmup);
    WriteIterationRows(batch, options, *csv);

    for (const CompileMeasurement& module : batch.modules) {
      if (!module.status.ok()) {
        return absl::Status(
            module.status.code(),
            absl::StrCat("Failed to compile ", module.hlo_file, ": ",
                         module.status.message()));
      }
    }
    std::cout << (warmup ? "Warmup " : "Iteration ") << visible_iteration
              << ": batch=" << std::fixed << std::setprecision(1)
              << batch.wall_ms << "ms\n";
    if (!warmup) {
      measurements.push_back(std::move(batch));
    }
  }

  std::vector<double> batch_times;
  batch_times.reserve(measurements.size());
  for (const BatchMeasurement& batch : measurements) {
    batch_times.push_back(batch.wall_ms);
  }
  const SummaryStats batch_stats = Summarize(std::move(batch_times));
  PrintSummary("batch", batch_stats);
  WriteSummaryRow("__batch__", batch_stats, options, *csv);

  for (size_t module_index = 0; module_index < hlo_files.size(); ++module_index) {
    std::vector<double> compile_times;
    compile_times.reserve(measurements.size());
    for (const BatchMeasurement& batch : measurements) {
      compile_times.push_back(batch.modules[module_index].compile_ms);
    }
    const SummaryStats stats = Summarize(std::move(compile_times));
    PrintSummary(hlo_files[module_index], stats);
    WriteSummaryRow(hlo_files[module_index], stats, options, *csv);
  }

  if (!options.output_csv.empty()) {
    std::cout << "Wrote CSV results to " << options.output_csv << '\n';
  }
  return absl::OkStatus();
}

}  // namespace
}  // namespace xla

int main(int argc, char** argv) {
  xla::BenchmarkOptions options;
  std::vector<tsl::Flag> flags = {
      tsl::Flag("repetitions", &options.repetitions,
                "Number of measured compile batches in this process."),
      tsl::Flag("warmup_repetitions", &options.warmup_repetitions,
                "Number of unmeasured compile batches before measurements."),
      tsl::Flag("parallelism", &options.parallelism,
                "Maximum number of HLO modules compiled concurrently."),
      tsl::Flag("output_csv", &options.output_csv,
                "CSV output path."),
      tsl::Flag("append_csv", &options.append_csv,
                "Append rows to --output_csv instead of replacing it."),
      tsl::Flag("label", &options.label,
                "Label written to each CSV row, such as a commit hash."),
      tsl::Flag("run_id", &options.run_id,
                "Identifier written to each CSV row for this invocation."),
      tsl::Flag("input_format", &options.input_format,
                "Input format accepted by FunctionalHloRunner."),
      tsl::Flag("gpu_client_mem_fraction", &options.gpu_client_mem_fraction,
                "Fraction of GPU memory available to the PJRT client."),
      tsl::Flag("gpu_client_initialization_timeout_sec",
                &options.gpu_client_initialization_timeout_sec,
                "GPU PJRT client initialization timeout in seconds."),
  };
  xla::AppendDebugOptionsFlags(&flags);
  xla::ParseDebugOptionFlagsFromEnv(/*reset_envvar=*/true);

  const std::string usage =
      absl::StrCat(xla::kUsage, "\n", tsl::Flags::Usage(argv[0], flags));
  const bool parse_ok = tsl::Flags::Parse(&argc, argv, flags);
  tsl::port::InitMain(usage.c_str(), &argc, &argv);
  if (!parse_ok) {
    std::cerr << usage;
    return 1;
  }

  std::vector<std::string> hlo_files;
  hlo_files.reserve(argc - 1);
  for (int index = 1; index < argc; ++index) {
    hlo_files.emplace_back(argv[index]);
  }
  const absl::Status status = xla::RunBenchmark(options, hlo_files);
  if (!status.ok()) {
    std::cerr << status << '\n';
    return 1;
  }
  return 0;
}
