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

// Deviceless replay of the tiling cost model. JSON formatting and HLO parsing
// are outside the timed search. Use --observe only for differential validation:
// copying candidate records necessarily adds overhead to the timed region.
#include <cstdint>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "google/protobuf/text_format.h"
#include "mlir/IR/MLIRContext.h"
#include "tsl/platform/init_main.h"
#include "xla/codegen/tiling/symbolic_tile_analysis.h"
#include "xla/hlo/analysis/symbolic_expr.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/utils/hlo_traversal.h"
#include "xla/service/gpu/gpu_device_info_for_tests.h"
#include "xla/service/gpu/model/fusion_analysis_cache.h"
#include "xla/service/gpu/model/gpu_indexing_performance_model.h"
#include "xla/service/gpu/model/triton_emitter_constraints.h"
#include "xla/service/hlo_cost_analysis.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/device_description.pb.h"
#include "xla/tools/hlo_module_loader.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/util/command_line_flags.h"

namespace xla::gpu {
namespace {

struct Options {
  std::string device = "a6000";
  std::string target_config;
  std::string fusion;
  std::string computation;
  std::string producer;
  std::string consumer;
  std::string name_filter;
  int iterations = 1;
  int top_k = 1;
  int64_t min_candidates = 0;
  int64_t max_candidates = 0;
  bool pruning = true;
  bool verify_bound = false;
  bool stats = false;
  bool observe = false;
  bool multi_output = false;
  bool workspace = false;
  bool share_tile_sizes = true;
  bool use_memory_bound_workspace = false;
  bool lazy_tile_sizes = false;
  bool compact = false;
  bool verify_compact = false;
  bool verify_evaluation = false;
  bool prepared = false;
  int64_t preparation_threshold = 32;
};

std::string Quote(absl::string_view value) {
  std::string result = "\"";
  constexpr char kHex[] = "0123456789abcdef";
  for (unsigned char c : value) {
    if (c == '\\' || c == '"') {
      result.push_back('\\');
      result.push_back(c);
    } else if (c < 0x20) {
      result += "\\u00";
      result.push_back(kHex[c >> 4]);
      result.push_back(kHex[c & 15]);
    } else {
      result.push_back(c);
    }
  }
  result.push_back('"');
  return result;
}

void PrintIntegers(absl::Span<const int64_t> values) {
  std::cout << '[';
  const char* separator = "";
  for (int64_t value : values) {
    std::cout << separator << value;
    separator = ",";
  }
  std::cout << ']';
}

void PrintCandidate(const TiledRunTimeData& candidate) {
  const auto& p = candidate.block_level_parameters;
  const auto& r = candidate.runtime_data;
  std::cout << "{\"output_tile_sizes\":[";
  const char* separator = "";
  for (const auto& tile : p.output_tile_sizes) {
    std::cout << separator;
    PrintIntegers(tile);
    separator = ",";
  }
  std::cout << "],\"num_warps\":" << p.num_warps
            << ",\"num_ctas\":" << p.num_ctas
            << ",\"num_stages\":" << p.num_stages
            << ",\"global_scratch_memory_size\":"
            << p.global_scratch_memory_size
            << ",\"is_tma_allowed\":" << p.is_tma_allowed
            << ",\"is_warp_specialization_allowed\":"
            << p.is_warp_specialization_allowed
            << ",\"num_tiles_per_pid\":" << p.num_tiles_per_pid
            << ",\"waves_per_eu\":" << p.waves_per_eu
            << ",\"flops\":" << r.flops << ",\"bytes_read\":" << r.bytes_read
            << ",\"bytes_written\":" << r.bytes_written
            << ",\"l2_bytes_read\":" << r.l2_bytes_read
            << ",\"shared_memory_per_block_bytes\":"
            << r.shared_memory_per_block_bytes
            << ",\"registers_per_thread\":" << r.registers_per_thread
            << ",\"compute_utilization\":" << r.compute_utilization
            << ",\"memory_utilization\":" << r.memory_utilization
            << ",\"read_time\":" << Quote(absl::FormatDuration(r.read_time))
            << ",\"write_time\":" << Quote(absl::FormatDuration(r.write_time))
            << ",\"compute_time\":"
            << Quote(absl::FormatDuration(r.compute_time))
            << ",\"exec_time\":" << Quote(absl::FormatDuration(r.exec_time))
            << '}';
}

void PrintStats(const TilingSearchStats& s) {
  std::cout
      << "{\"symbolic_instructions\":" << s.symbolic_instructions
      << ",\"enumerated\":" << s.enumerated
      << ",\"constraint_rejections\":" << s.constraint_rejections
      << ",\"memory_rejections\":" << s.memory_rejections
      << ",\"materialized_candidates\":" << s.materialized_candidates
      << ",\"materialized_instructions\":" << s.materialized_instructions
      << ",\"accepted_candidates\":" << s.accepted_candidates
      << ",\"compact_candidates\":" << s.compact_candidates
      << ",\"compact_fallbacks\":" << s.compact_fallbacks
      << ",\"compact_rejection_reason\":" << Quote(s.compact_rejection_reason)
      << ",\"expression_preparations\":" << s.expression_preparations
      << ",\"prepared_evaluations\":" << s.prepared_evaluations
      << ",\"compact_time\":" << Quote(absl::FormatDuration(s.compact_time))
      << ",\"expression_preparation_time\":"
      << Quote(absl::FormatDuration(s.expression_preparation_time))
      << ",\"analysis_time\":" << Quote(absl::FormatDuration(s.analysis_time))
      << ",\"constraint_time\":"
      << Quote(absl::FormatDuration(s.constraint_time))
      << ",\"memory_bound_time\":"
      << Quote(absl::FormatDuration(s.memory_bound_time))
      << ",\"tile_sizes_time\":"
      << Quote(absl::FormatDuration(s.tile_sizes_time))
      << ",\"materialization_time\":"
      << Quote(absl::FormatDuration(s.materialization_time))
      << ",\"estimation_time\":"
      << Quote(absl::FormatDuration(s.estimation_time))
      << ",\"total_time\":" << Quote(absl::FormatDuration(s.total_time)) << '}';
}

absl::StatusOr<se::DeviceDescription> GetDevice(absl::string_view name) {
  if (name == "a6000") return TestGpuDeviceInfo::RTXA6000DeviceInfo();
  if (name == "a100") return TestGpuDeviceInfo::A100SXMDeviceInfo();
  if (name == "h100") return TestGpuDeviceInfo::H100SXMDeviceInfo();
  if (name == "b200") return TestGpuDeviceInfo::B200SXMDeviceInfo();
  if (name == "mi350") return TestGpuDeviceInfo::AMDMI350DeviceInfo();
  return absl::InvalidArgumentError(
      absl::StrCat("Unknown device fixture: ", name));
}

absl::StatusOr<se::DeviceDescription> LoadTargetConfig(
    const std::string& filename) {
  std::string text;
  ABSL_RETURN_IF_ERROR(
      tsl::ReadFileToString(tsl::Env::Default(), filename, &text));
  se::GpuTargetConfigProto proto;
  if (!google::protobuf::TextFormat::ParseFromString(text, &proto)) {
    return absl::InvalidArgumentError(
        "Failed to parse target configuration textproto");
  }
  if (!proto.has_gpu_device_info()) {
    return absl::InvalidArgumentError(
        "Target configuration has no gpu_device_info");
  }
  return se::DeviceDescription::FromProto(proto.gpu_device_info());
}

struct ReplayFusion {
  std::string name;
  std::unique_ptr<HloFusionAdaptor> adaptor;
};

absl::StatusOr<std::vector<ReplayFusion>> SelectFusions(
    const HloModule& module, const Options& options) {
  std::vector<ReplayFusion> selected;
  const HloInstruction* producer = nullptr;
  const HloInstruction* consumer = nullptr;
  for (const HloComputation* computation : module.MakeComputationPostOrder()) {
    if (!options.computation.empty() &&
        computation->name() == options.computation) {
      if (!computation->IsFusionComputation()) {
        return absl::InvalidArgumentError(
            "--computation must name a fusion computation");
      }
      selected.push_back({std::string(computation->name()),
                          HloFusionAdaptor::ForComputation(computation)});
    }
    for (const HloInstruction* instruction : computation->instructions()) {
      if (instruction->name() == options.producer) producer = instruction;
      if (instruction->name() == options.consumer) consumer = instruction;
      if (!options.computation.empty() || !options.producer.empty()) continue;
      if (instruction->opcode() != HloOpcode::kFusion) continue;
      if (!options.fusion.empty() && instruction->name() != options.fusion) {
        continue;
      }
      if (!absl::StrContains(instruction->name(), options.name_filter)) {
        continue;
      }
      selected.push_back({std::string(instruction->name()),
                          HloFusionAdaptor::ForInstruction(instruction)});
    }
  }
  if (!options.producer.empty()) {
    if (producer == nullptr || consumer == nullptr ||
        producer->parent() != consumer->parent() ||
        !consumer->IsUserOf(producer)) {
      return absl::InvalidArgumentError(
          "--producer and --consumer must name an existing producer-user pair");
    }
    selected.push_back({absl::StrCat(options.producer, "->", options.consumer),
                        HloFusionAdaptor::ForProducerConsumer(
                            producer, consumer, options.multi_output)});
  }
  if (selected.empty()) {
    return absl::NotFoundError(
        "No selected fusion; use --computation for an unfused computation");
  }
  return selected;
}

absl::StatusOr<int64_t> CountCandidates(const HloFusionAdaptor& fusion,
                                        mlir::MLIRContext* context,
                                        const se::DeviceDescription& device) {
  auto analysis_or_error = SymbolicTileAnalysis::AnalyzeFusion(
      fusion, context, TritonEmitterConstraints::GetBuilder(device));
  if (const auto* decision = std::get_if<FusionDecision>(&analysis_or_error)) {
    return absl::InvalidArgumentError(decision->Explain());
  }
  int64_t count = 0;
  ABSL_RETURN_IF_ERROR(
      std::get<SymbolicTileAnalysis>(analysis_or_error)
          .ForEachValidFlatTiling([&](absl::Span<const int64_t>) {
            ++count;
            return absl::OkStatus();
          }));
  return count;
}

struct ObservedCandidate {
  std::vector<int64_t> tiling;
  std::optional<TiledRunTimeData> result;
};

absl::Status Run(const std::string& filename, const Options& options) {
  ABSL_ASSIGN_OR_RETURN(std::unique_ptr<HloModule> module,
                        LoadModuleFromFile(filename));
  ABSL_ASSIGN_OR_RETURN(std::vector<ReplayFusion> fusions,
                        SelectFusions(*module, options));
  ABSL_ASSIGN_OR_RETURN(se::DeviceDescription device,
                        options.target_config.empty()
                            ? GetDevice(options.device)
                            : LoadTargetConfig(options.target_config));
  mlir::MLIRContext context;
  RegisterSymbolicExprStorage(&context);
  HloFusionAnalysisCache fusion_analysis_cache(device);
  GpuPerformanceModelWithIndexingAnalysis model(
      &device, &fusion_analysis_cache, HloCostAnalysis::DefaultShapeSize,
      &context,
      /*use_experimental_tiling=*/false,
      /*enable_same_shape_multi_output_fusion=*/false);
  bool failed = false;
  for (const ReplayFusion& fusion : fusions) {
    std::string identity = absl::StrCat(
        "\"input\":", Quote(filename), ",\"module\":", Quote(module->name()),
        ",\"fusion\":", Quote(fusion.name), ",\"device\":",
        Quote(options.target_config.empty() ? options.device : "target_config"),
        ",\"target_config\":", Quote(options.target_config));
    if (options.min_candidates > 0 || options.max_candidates > 0) {
      auto count = CountCandidates(*fusion.adaptor, &context, device);
      if (!count.ok() || *count < options.min_candidates ||
          (options.max_candidates > 0 && *count > options.max_candidates)) {
        std::cout << "{\"event\":\"filtered\"," << identity << ",\"reason\":"
                  << Quote(count.ok()
                               ? absl::StrCat("valid_candidates=", *count)
                               : count.status().ToString())
                  << "}\n";
        continue;
      }
    }
    for (int iteration = 0; iteration < options.iterations; ++iteration) {
      TilingSearchStats stats;
      TilingSearchOptions search_options;
      search_options.enable_memory_bound = options.pruning;
      search_options.use_workspace = options.workspace;
      search_options.share_tile_sizes = options.share_tile_sizes;
      search_options.use_memory_bound_workspace =
          options.use_memory_bound_workspace;
      search_options.lazy_tile_sizes = options.lazy_tile_sizes;
      search_options.use_compact = options.compact;
      search_options.verify_compact = options.verify_compact;
      search_options.verify_evaluation = options.verify_evaluation;
      search_options.use_prepared_expressions = options.prepared;
      search_options.expression_preparation_threshold =
          options.preparation_threshold;
      search_options.verify_memory_bound = options.verify_bound;
      search_options.stats = options.stats ? &stats : nullptr;
      std::vector<ObservedCandidate> observed;
      if (options.observe) {
        search_options.observe_candidate =
            [&](absl::Span<const int64_t> tiling,
                const std::optional<TiledRunTimeData>& result) {
              observed.push_back(
                  {std::vector<int64_t>(tiling.begin(), tiling.end()), result});
            };
      }
      const std::clock_t cpu_start = std::clock();
      const absl::Time start = absl::Now();
      auto result = model.TryFindTopKBestTilingsForFusion(
          *fusion.adaptor, options.top_k, search_options);
      const absl::Duration elapsed = absl::Now() - start;
      const std::clock_t cpu_end = std::clock();
      std::cout << "{\"event\":\"search\"," << identity
                << ",\"iteration\":" << iteration
                << ",\"top_k\":" << options.top_k
                << ",\"pruning\":" << options.pruning
                << ",\"workspace\":" << options.workspace
                << ",\"share_tile_sizes\":" << options.share_tile_sizes
                << ",\"use_memory_bound_workspace\":"
                << options.use_memory_bound_workspace
                << ",\"lazy_tile_sizes\":" << options.lazy_tile_sizes
                << ",\"compact\":" << options.compact
                << ",\"verify_compact\":" << options.verify_compact
                << ",\"verify_evaluation\":" << options.verify_evaluation
                << ",\"prepared\":" << options.prepared
                << ",\"preparation_threshold\":"
                << options.preparation_threshold
                << ",\"verify_bound\":" << options.verify_bound
                << ",\"observe\":" << options.observe << ",\"count_filter\":"
                << (options.min_candidates > 0 || options.max_candidates > 0)
                << ",\"wall_time\":" << Quote(absl::FormatDuration(elapsed))
                << ",\"cpu_clock_ticks\":";
      if (cpu_start == std::clock_t(-1) || cpu_end == std::clock_t(-1)) {
        std::cout << "null";
      } else {
        std::cout << cpu_end - cpu_start;
      }
      std::cout << ",\"cpu_clocks_per_second\":" << CLOCKS_PER_SEC;
      if (options.stats) {
        std::cout << ",\"stats\":";
        PrintStats(stats);
      }
      if (!result.ok()) {
        failed = true;
        std::cout << ",\"status\":\"error\",\"message\":"
                  << Quote(result.status().ToString());
      } else if (const auto* decision = std::get_if<FusionDecision>(&*result)) {
        std::cout << ",\"status\":\"forbidden\",\"message\":"
                  << Quote(decision->Explain());
      } else {
        std::cout << ",\"status\":\"ok\",\"candidates\":[";
        const char* separator = "";
        for (const TiledRunTimeData& candidate :
             std::get<absl::InlinedVector<TiledRunTimeData, 4>>(*result)) {
          std::cout << separator;
          PrintCandidate(candidate);
          separator = ",";
        }
        std::cout << ']';
      }
      std::cout << "}\n";
      for (const ObservedCandidate& candidate : observed) {
        std::cout << "{\"event\":\"candidate\"," << identity
                  << ",\"iteration\":" << iteration << ",\"tiling\":";
        PrintIntegers(candidate.tiling);
        std::cout << ",\"result\":";
        if (candidate.result.has_value()) {
          PrintCandidate(*candidate.result);
        } else {
          std::cout << "null";
        }
        std::cout << "}\n";
      }
    }
  }
  return failed ? absl::InternalError("One or more searches failed")
                : absl::OkStatus();
}

}  // namespace
}  // namespace xla::gpu

int main(int argc, char** argv) {
  xla::gpu::Options options;
  std::vector<tsl::Flag> flags = {
      tsl::Flag("device", &options.device,
                "a6000,a100,h100,b200,mi350; or use --target_config"),
      tsl::Flag("target_config", &options.target_config,
                "GpuTargetConfigProto textproto; overrides --device fixture"),
      tsl::Flag("fusion", &options.fusion, "Exact fusion instruction name"),
      tsl::Flag("computation", &options.computation, "Exact computation name"),
      tsl::Flag("producer", &options.producer, "Producer name for pair replay"),
      tsl::Flag("consumer", &options.consumer, "Consumer name for pair replay"),
      tsl::Flag("name_filter", &options.name_filter, "Fusion name substring"),
      tsl::Flag("iterations", &options.iterations,
                "Search repetitions per fusion"),
      tsl::Flag("top_k", &options.top_k, "Number of retained candidates"),
      tsl::Flag("min_candidates", &options.min_candidates,
                "Minimum valid candidates; untimed enumeration prepass"),
      tsl::Flag("max_candidates", &options.max_candidates,
                "Maximum valid candidates in selected fusion; zero=unlimited"),
      tsl::Flag("pruning", &options.pruning, "Enable memory-bound pruning"),
      tsl::Flag("verify_bound", &options.verify_bound,
                "Evaluate all candidates and verify memory bounds"),
      tsl::Flag("stats", &options.stats, "Collect stage counters and timers"),
      tsl::Flag("observe", &options.observe,
                "Copy and print each evaluated candidate"),
      tsl::Flag("multi_output", &options.multi_output,
                "Allow side output in pair adaptor"),
      tsl::Flag("share_tile_sizes", &options.share_tile_sizes,
                "Share evaluated sizes with the memory bound"),
      tsl::Flag("use_memory_bound_workspace",
                &options.use_memory_bound_workspace,
                "Reuse memory-bound metadata and candidate storage"),
      tsl::Flag("lazy_tile_sizes", &options.lazy_tile_sizes,
                "Evaluate bound inputs first; requires memory-bound workspace"),
      tsl::Flag("compact", &options.compact,
                "Use compact cost-model candidate view when supported"),
      tsl::Flag("verify_compact", &options.verify_compact,
                "Compare compact candidates with independent materialization"),
      tsl::Flag(
          "verify_evaluation", &options.verify_evaluation,
          "Compare workspace candidates with independent materialization"),
      tsl::Flag("prepared", &options.prepared,
                "Use prepared expression batches"),
      tsl::Flag(
          "preparation_threshold", &options.preparation_threshold,
          "Evaluations before preparation; one forces first-use preparation"),
      tsl::Flag("workspace", &options.workspace,
                "Reuse search-local tile evaluation storage"),
  };
  const std::string usage =
      absl::StrCat("Deviceless tiling search replay. Usage: ", argv[0],
                   " [flags] input.hlo [input2.hlo ...]\n",
                   tsl::Flags::Usage(argv[0], flags));
  if (!tsl::Flags::Parse(&argc, argv, flags)) {
    std::cerr << usage;
    return 1;
  }
  tsl::port::InitMain(usage.c_str(), &argc, &argv);
  int selectors = static_cast<int>(!options.fusion.empty()) +
                  static_cast<int>(!options.computation.empty()) +
                  static_cast<int>(!options.producer.empty());
  if (argc < 2 || options.iterations < 1 || options.top_k < 0 ||
      options.preparation_threshold < 1 || options.min_candidates < 0 ||
      options.max_candidates < 0 || selectors > 1 ||
      options.producer.empty() != options.consumer.empty() ||
      (options.multi_output && options.producer.empty()) ||
      (options.max_candidates > 0 &&
       options.min_candidates > options.max_candidates)) {
    std::cerr << "Invalid arguments.\n" << usage;
    return 1;
  }
  std::cout << std::boolalpha << std::setprecision(17);
  bool failed = false;
  for (int i = 1; i < argc; ++i) {
    absl::Status status = xla::gpu::Run(argv[i], options);
    if (!status.ok()) {
      std::cerr << argv[i] << ": " << status << '\n';
      failed = true;
    }
  }
  return failed ? 1 : 0;
}
