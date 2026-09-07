# GLM-5.3-Flash XLA compile-time profile

Agent handoff for reducing XLA compilation time for the ZML GLM-5.3-Flash
workload. This document is intentionally source- and experiment-oriented.

## Snapshot

- Date: 2026-09-06
- XLA revision: `cb5faa79cab5747d66da025f7c9949a84ae73afa`
- Revision subject: `xla/pjrt/gpu: support memory stats for VMM allocator`
- XLA repository: `/home/benjamin/github/openxla/xla`
- ZML repository: `/home/benjamin/github/zml/monorepo`
- Model: `/var/models/zai-org/GLM-5.3-Flash/experts-repacked-fn-ep-v3/`
- Backend: Triton
- Host: AArch64, 144 CPUs, Neoverse-V2, two NUMA nodes
- GPUs: NVIDIA GB300; workload restricted to GPUs 0 and 1
- Profiled placement: CPU and memory NUMA node 0
- Approximate XLA model compilation interval: 150 seconds

XLA plugin build:

```sh
cd /home/benjamin/github/openxla/xla
USE_BAZEL_VERSION=7.7.0 bazel build \
  --disk_cache=/home/benjamin/.cache/bazel-disk \
  --enable_workspace \
  --spawn_strategy=local \
  --config=cuda_nvcc \
  --config=baseline_arm64 \
  //xla/pjrt/c:pjrt_c_api_gpu_plugin
```

The plugin is consumed by ZML through
`/home/benjamin/github/zml/zml/platforms/cuda/cuda.bzl`.

## Workload

```sh
cd /home/benjamin/github/zml/monorepo
bazel run \
  --run_under="ZML_AUTOTUNE_CACHE_DIR=/home/benjamin/xla_cache XLA_FLAGS='--xla_gpu_command_buffer_update_mode=SKIP_TEMP --xla_gpu_command_buffer_scheduling_mode=CONCURRENT --xla_gpu_enable_command_buffer=+COLLECTIVES' CUDA_VISIBLE_DEVICES=0,1 numactl --cpunodebind=0 --membind=0" \
  --@zml//platforms:cuda=true \
  --config=debug \
  //llmd -- \
  --model=/var/models/zai-org/GLM-5.3-Flash/experts-repacked-fn-ep-v3/ \
  --backend=triton \
  --batch-size=16 \
  --gpu-memory-fraction=0.95 \
  --cache-memory-fraction=0.97 \
  --listen=0.0.0.0:8768 \
  --token-batch-size=256 \
  --bench-prompt='hi' \
  --bench-concurrency=16
```

Keep the model, cache state, XLA flags, GPU visibility, and NUMA binding fixed
when comparing changes. Separate Bazel build time from process compilation time.

## Profile artifact

```text
/home/benjamin/glm53-xla-perf-debugconfig-20260906.data
```

- Complete post-fix compiler profile
- Flat `cycles:u` sampling at 99 Hz
- Size: approximately 5.1 MiB
- `libpjrt_cuda.so`: 83.48% of sampled self cycles
- Approximate compilation interval: 150 seconds
- Symbolic-tiling/cost-model symbol classification: 53.91% of all self cycles

The usable perf binary on the host is:

```text
/usr/lib/linux-tools/6.8.0-139-generic/perf
```

Report command:

```sh
PERF=/usr/lib/linux-tools/6.8.0-139-generic/perf
$PERF report \
  -i /home/benjamin/glm53-xla-perf-debugconfig-20260906.data \
  --stdio --no-children --sort dso,symbol
```

For new phase-specific profiles, use `cycles:u`, 99 Hz, and DWARF call graphs
with an 8192-byte stack dump. Use `perf report --no-inline` on this host if line
symbolization is incomplete.

## Profile result

Top relevant XLA symbols:

| Symbol | Self cycles |
|---|---:|
| `ComputeTiledInstructions` | 7.22% |
| `EvaluateSymbolicExprs` | 6.30% |
| `SymbolicTileAnalysis::ComputeTiledComputation` | 2.93% |
| `ForEachInstructionInTiledHloComputation` | 2.22% |
| `TiledHloInstruction::CreateUncheckedFromSymbolicTile` | 1.71% |
| `TiledHloInstruction::~TiledHloInstruction` | 1.47% |
| `EstimateRunTimeForTiledHloComputationImpl` | 1.38% |
| `EvaluateTileSizes` | 1.30% |
| `TryFindTopKBestTilingsForFusion` FLOP helper | 1.27% |
| `IndexingMap` move construction | 1.24% |
| `AbslHashValue(TiledHloInstruction)` | 1.20% |
| `CreateSymbolicVariable` | 1.11% |
| `IndexingMap` destruction | 1.08% |
| `EstimateNumWarps` helper | 1.08% |
| `absl::Mutex::lock` | 1.06% |
| `TryFindTopKBestTilingsForFusion` candidate callback | 1.00% |
| `ReplaceVariables` | 0.97% |
| `ReplaceDimsAndSymbols` | 0.94% |
| `TiledHloInstruction` construction | 0.94% |

PTXAS accounts for 1.13% of sampled self cycles. GPU execution, disk I/O, and
PTX assembly do not explain the compile interval. During symbolic tiling the
process used approximately 5-6 effective CPU cores despite substantially more
host capacity.

## Hot path

The search loop is in
`xla/service/gpu/model/gpu_indexing_performance_model.cc`, function
`GpuPerformanceModelWithIndexingAnalysis::TryFindTopKBestTilingsForFusion`, near
lines 875-1045.

The experimental tiling path constructs one `SymbolicTileAnalysis`, then calls
`analysis.ForEachValidFlatTiling`. Each candidate not rejected by the memory
lower bound runs:

```text
analysis.ComputeTiledComputation(flat_tiling, schedule, constraints=true)
EstimateTiledRunTimeDataImpl(..., tiled_hlo_computation, ...)
retain_candidate(...)
destroy tiled_hlo_computation
```

The relevant callback is near lines 994-1037. The memory lower-bound check is
near lines 997-1005 and only becomes active after the retained-candidate array
contains `top_k` entries. Default top-k is one. Candidate ordering affects both
time to first incumbent and subsequent pruning strength.

`ComputeTiledComputationImpl` is in
`xla/codegen/tiling/symbolic_tile_analysis.cc`, near lines 1811-1947. Each
candidate creates or fills:

- `symbolic_to_tiled_hlo_map`
- `stride_parameters_by_bounds`
- `DenseTileValueMap tile_strides_map`
- evaluated strides for every symbolic tiled HLO
- `DenseTileValueMap tile_sizes_map`
- temporary hashes for duplicate detection
- `parameters_with_offset_indexing`
- `OutputTilingInfo`
- a complete heap-owned `TiledHloInstruction` graph
- tiled roots and the resulting `TiledHloComputation`

`ComputeTiledInstructions`, near lines 2081-2115, creates an
`OrderedUniquePtrValueHashSet<TiledHloInstruction>`, reserves it, constructs one
instruction for every symbolic instruction, hashes/deduplicates the objects, and
extracts them into a vector. The graph exists only long enough to estimate one
candidate before being destroyed.

The implementation already avoids some tile-offset indexing. Lines 1867-1880
explicitly identify those maps as expensive and limit their computation in the
cost-model case. Full candidate graph materialization, expression evaluation,
deduplication, and destruction remain the dominant compiler costs.

## P0: cost-model-only candidate evaluation

Avoid constructing a complete `TiledHloComputation` for every candidate. Evaluate
the data required by `EstimateTiledRunTimeDataImpl` directly from
`SymbolicTileAnalysis` into compact indexed storage.

Candidate design:

```text
SymbolicTileAnalysis
  + TilingEvaluationWorkspace
  + EvaluateCandidate(flat_tiling, workspace)
      -> TiledRunTimeData or rejection
```

The lightweight evaluator should compute only:

- effective tile sizes and strides
- tile counts
- bytes read/written
- element and FLOP counts
- launch dimensions, warp count, and occupancy inputs
- values needed by coalescing and transaction estimates
- deduplication/equivalence information that changes cost

Only construct `TiledHloInstruction` objects if a downstream consumer actually
requires them. `TryFindTopKBestTilingsForFusion` returns runtime data rather than
the winning tiled graph, so determine whether materialization can be removed for
all candidates including the winner.

Equivalence constraints:

- Preserve instruction deduplication semantics; multiple symbolic instructions
  may map to one concrete tiled instruction.
- Preserve multi-root handling.
- Preserve current `Unimplemented` behavior for unsupported multi-output cases.
- Preserve parameter/root offset-indexing behavior where identical sizes require
  offset comparison.
- Compare accepted/rejected candidates, selected tilings, and predicted durations
  against the existing implementation.

## P1: reusable evaluation workspace

If direct symbolic cost evaluation is too invasive, reuse storage for the
duration of one `TryFindTopKBestTilingsForFusion` call:

- symbolic-to-tiled index mapping
- clamped stride-parameter buffers
- dense tile-size and tile-stride maps
- active-parameter vectors
- duplicate-detection hash storage
- tiled-instruction storage
- temporary `IndexingMap` substitution/evaluation storage

Reset logical contents without releasing capacity. Prefer integer indices and an
arena-like instruction representation over independently owned objects for the
cost-model path.

Move candidate-invariant computation outside `ForEachValidFlatTiling`. Audit:

- traversal and schedule order
- symbolic instruction metadata
- tile-parameter upper-bound layouts
- parameter activity
- FLOPs per element; a local cache already exists
- normalized/compiled forms consumed by `EvaluateSymbolicExprs`,
  `ReplaceVariables`, and `ReplaceDimsAndSymbols`

Evaluated sizes and strides vary with `flat_tiling`; cache them only by the actual
parameter tuple and bounds identifier. Multiple flat tilings can clamp to the
same effective parameter tuple for instructions with smaller upper bounds. That
provides an explicit memoization and equivalence opportunity.

## P1: pre-materialization pruning and candidate order

Produce a strong incumbent before exhaustive traversal:

- Evaluate a heuristic/default candidate first.
- Enumerate likely-good candidates first.
- Apply memory, occupancy, launch, shared-memory, register, thread, and block
  feasibility bounds from scalar tile parameters before indexing-map evaluation.
- Memoize candidates with equivalent clamped parameter tuples.
- Detect dominated candidates using cheap cost components before building the
  instruction representation.

Existing VLOG(1) output records:

```text
TryFindTopKBestTilingsForFusion symbolic analysis evaluated N tilings;
memory bound pruned M.
```

Collect `N`, `M`, and the existing scoped-timer duration per fusion. Rank fusions
by total search time, candidate count, and low `M/N`. Add timers around these
regions if aggregate profiles are insufficient:

- `ComputeTiledComputation`
- tile-size and tile-stride evaluation
- `ComputeOutputTilingInfo`
- `ComputeTiledInstructions`
- instruction hashing/deduplication
- `EstimateTiledRunTimeDataImpl`

Also record candidate count before and after constraints, maximum symbolic graph
size, constructed instruction count, and deduplication ratio.

## P2: parallel evaluation

The callback is sequential and the machine has unused CPU capacity. Parallelism
should follow representation and pruning work, not precede it.

Risks visible in the profile and source:

- `absl::Mutex::lock` already consumes 1.06% self cycles.
- MLIR storage uniquing and locking are visible below the top-symbol cutoff.
- Candidate evaluation receives an `MLIRContext`; sharing it may serialize.
- Concurrent complete-graph construction increases temporary memory demand.
- Retained-candidate ordering and floating-duration comparisons must remain
  deterministic.

Prefer parallelism across independent fusions. If parallelizing candidates, use
per-worker evaluation workspaces and investigate per-worker MLIR contexts or a
representation that does not require MLIR mutation/uniquing during evaluation.
Do not expect `xla_gpu_force_compilation_parallelism` alone to parallelize this
specific enumeration/callback loop.

## Non-targets

- PTXAS and PTX generation: 1.13% self cycles.
- GPU autotuning execution: GPUs were idle in the compiler interval.
- Disk I/O: negligible.
- General LLVM backend parallelism: not prominent in this profile.
- Command-buffer flags: not on the identified CPU hot path.

## Instrumentation patch before optimization

Add low-overhead per-fusion counters, aggregated rather than logged per candidate:

```text
fusion name
symbolic instruction count
valid flat tiling count
constraint rejection count
memory-bound rejection count
full candidate construction count
deduplicated instruction count
ComputeTiledComputation duration
EstimateTiledRunTimeDataImpl duration
total TryFindTopKBestTilingsForFusion duration
```

Use the existing scoped timer and VLOG infrastructure. Avoid unconditional logs
inside the candidate callback; candidate volume can perturb the measurement.

## Validation protocol

For each patch:

1. Clear or hold compilation cache state constant.
2. Run the exact workload and record `Compiled all models` duration.
3. Capture a complete 99 Hz `cycles:u` profile.
4. Capture one DWARF call-graph slice during symbolic tiling.
5. Record per-fusion candidate/pruning/timing counters.
6. Record peak RSS and effective CPU utilization.
7. Compare every selected tiling and predicted duration with the unmodified XLA.
8. Run focused symbolic-tiling and GPU performance-model tests.
9. If tiling decisions change intentionally, benchmark generated kernels; compile
   time alone is insufficient.

Primary success criteria:

- lower model compilation wall time
- lower cycles in `ComputeTiledInstructions`, `EvaluateSymbolicExprs`, indexing-map
  substitution, hashing, and destruction
- unchanged tiling decisions unless explicitly justified
- no material regression in peak RSS

## Suggested execution order

1. Add per-fusion aggregate instrumentation.
2. Identify the small set of fusions responsible for most candidate evaluations.
3. Implement capacity reuse/workspace lifetime across candidates.
4. Measure the upper bound obtainable from storage reuse alone.
5. Implement a lightweight evaluator for one restricted fusion class.
6. Differentially run both evaluators for every candidate and assert equal cost
   inputs/results.
7. Extend the lightweight path to the dominant fusion classes.
8. Improve candidate ordering and cheap pruning.
9. Evaluate fusion-level and then candidate-level parallelism.

