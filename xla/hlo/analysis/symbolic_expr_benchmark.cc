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
#include <cstdint>
#include <optional>
#include <vector>

#include "absl/types/span.h"
#include "benchmark/benchmark.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/hlo/analysis/symbolic_expr.h"

namespace xla {
namespace {

// The cases distinguish terminal-only batches, independent nonterminal results,
// and results sharing a nontrivial subexpression. Construction of the symbolic
// expressions is outside all measurements; program preparation is measured by
// BM_PrepareAndEvaluateSymbolicExprProgram.
std::vector<SymbolicExpr> MakeExpressions(mlir::MLIRContext& context,
                                          int64_t count, int64_t family) {
  SymbolicExpr v0 = CreateSymbolicVariable(0, &context);
  SymbolicExpr v1 = CreateSymbolicVariable(1, &context);
  SymbolicExpr shared =
      ((v0 + 31).floorDiv(32) * v1.min(1024)).ceilDiv(8).min(4096).max(1);
  std::vector<SymbolicExpr> expressions;
  expressions.reserve(count);
  for (int64_t index = 0; index < count; ++index) {
    if (family == 0) {
      expressions.push_back(index % 2 == 0
                                ? CreateSymbolicConstant(index, &context)
                                : (index % 4 == 1 ? v0 : v1));
    } else if (family == 1) {
      expressions.push_back(
          ((v0 + index + 1).floorDiv(index + 2) * v1).min(4096).max(1));
    } else {
      expressions.push_back((shared + index).floorDiv(index + 1));
    }
  }
  return expressions;
}

template <bool kPrepared, bool kIncludePreparation>
void RunEvaluationBenchmark(benchmark::State& state) {
  mlir::MLIRContext context;
  RegisterSymbolicExprStorage(&context);
  const std::vector<SymbolicExpr> expressions =
      MakeExpressions(context, state.range(0), state.range(1));
  std::optional<SymbolicExprProgram> persistent_program;
  std::vector<int64_t> scratch;
  if constexpr (kPrepared && !kIncludePreparation) {
    persistent_program.emplace(expressions);
    scratch.resize(persistent_program->scratch_size());
  }
  std::vector<int64_t> results(expressions.size());
  const int64_t candidates = state.range(2);
  for (auto _ : state) {
    std::optional<SymbolicExprProgram> per_iteration_program;
    if constexpr (kIncludePreparation) {
      per_iteration_program.emplace(expressions);
      scratch.resize(per_iteration_program->scratch_size());
    }
    for (int64_t candidate = 0; candidate < candidates; ++candidate) {
      const std::array<int64_t, 2> values = {1 + candidate % 1024,
                                             1 + candidate % 64};
      if constexpr (kIncludePreparation) {
        per_iteration_program->Evaluate(values, absl::MakeSpan(scratch),
                                        absl::MakeSpan(results));
      } else if constexpr (kPrepared) {
        persistent_program->Evaluate(values, absl::MakeSpan(scratch),
                                     absl::MakeSpan(results));
      } else {
        EvaluateSymbolicExprs(expressions, values, absl::MakeSpan(results));
      }
      benchmark::DoNotOptimize(results);
      benchmark::ClobberMemory();
    }
  }
  state.SetItemsProcessed(state.iterations() * candidates * expressions.size());
}

void BM_EvaluateSymbolicExprs(benchmark::State& state) {
  RunEvaluationBenchmark<false, false>(state);
}

void BM_EvaluateSymbolicExprProgram(benchmark::State& state) {
  RunEvaluationBenchmark<true, false>(state);
}

void BM_PrepareAndEvaluateSymbolicExprProgram(benchmark::State& state) {
  RunEvaluationBenchmark<true, true>(state);
}

#define SYMBOLIC_EXPR_BENCHMARK_ARGUMENTS \
  ArgsProduct({{4, 64}, {0, 1, 2}, {1, 8, 16, 32, 64, 128}})
BENCHMARK(BM_EvaluateSymbolicExprs)->SYMBOLIC_EXPR_BENCHMARK_ARGUMENTS;
BENCHMARK(BM_EvaluateSymbolicExprProgram)->SYMBOLIC_EXPR_BENCHMARK_ARGUMENTS;
BENCHMARK(BM_PrepareAndEvaluateSymbolicExprProgram)
    ->SYMBOLIC_EXPR_BENCHMARK_ARGUMENTS;
#undef SYMBOLIC_EXPR_BENCHMARK_ARGUMENTS

}  // namespace
}  // namespace xla
