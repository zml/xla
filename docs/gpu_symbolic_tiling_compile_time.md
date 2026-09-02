# GPU symbolic tiling compile-time optimization

This document describes the compile-time investigation and optimizations for
the GPU symbolic tiling search. The motivating workload was a captured
DeepSeek-V4 CSA layer on an AMD MI300X. The same code is shared by GPU backends,
so the implementation avoids ROCm-specific behavior.

The final implementation reduced the cold compile time of the captured module
from a five-process mean of 50.241274 seconds to a five-process mean of
30.569076 seconds with autotuning disabled. This is a 39.155% reduction, or a
1.644x compile-speed improvement. The final samples ranged from 29.578966 to
31.717619 seconds.

## Executive conclusions

The investigation established the following:

* The long startup was primarily host compiler time, not ROCm kernel execution
  or final HSACO generation.
* The dominant high-level phase was `PriorityFusion`. Within its cost model,
  symbolic tiling analysis and construction of a concrete tiled graph for every
  candidate were the two main costs.
* Autotuning generated substantial aggregate work, but disabling it did not
  remove the regression. It reduced the model's compile timer from 130.967 to
  116.772 seconds; the wider original log interval through packing was about
  134 seconds. The layer programs still spent 76 to 114 seconds in
  `GpuCompiler::RunHloPasses`.
* Serializing XLA's compilation pool was not a solution. Setting
  `--xla_gpu_force_compilation_parallelism=1` increased full-model compilation
  from about 116 seconds to about 603 seconds.
* End-to-end `llmd` timing was too sensitive to concurrent module compilation,
  process placement, allocator state, caches, profiling overhead, and other
  users of the GPU. A captured-HLO test tube was required for defensible A/B
  measurements.
* The retained changes reduce redundant host work and preserve tiling order,
  constraint semantics, stable top-k ordering, and the generated-program search
  space. They are not intended to trade runtime throughput for compile time.

### Original branch-range question

The investigation began with a perceived slowdown between
`0f4a626bb5e59dba85b369b6342cf671b889375a` ("Drop the .so suffix for PjRt
plugin targets") and the then-current ROCm tip
`86dd5bb22f308d8c26626144141ea7e6fe6eb92c`. That interval contains 60 commits
and about 5,600 added lines across hermetic toolchains, TheRock configuration,
autotuning, command buffers, VMM, host execution, BLAS, FFT, DNN, streams, and
performance diagnostics.

No Git bisect was performed. A static review of that broad interval could not
separate an actual compiler regression from host contention and run-to-run
noise. The timer and `perf` evidence instead identified an existing dominant
path in `PriorityFusion`'s symbolic tiling cost model. The optimization work in
this document attacks that measured path directly; it does not claim that one
specific commit in the 60-commit interval introduced all of the observed wall
time.

## How the bottleneck was found

### Full-model capture

The first useful trace enabled XLA's scoped timers and compilation statistics
while preserving all output in one directory. A later capture also dumped the
input HLO modules so individual programs could be replayed:

```shell
trace_dir="$(mktemp -d /tmp/deepseek-v4-xla.XXXXXX)"

TF_CPP_MIN_LOG_LEVEL=0 \
XLA_FLAGS="--xla_enable_scoped_logging_timers=true \
--xla_gpu_print_compilation_stats=true \
--xla_gpu_autotune_level=0 \
--xla_dump_to=${trace_dir}/hlo \
--xla_dump_hlo_as_text=true \
--xla_dump_hlo_snapshots=true" \
ROCPROFILER_QUEUE_INTERPOSITION=false \
HIP_VISIBLE_DEVICES=1 \
bazel run //llmd \
  --@zml//platforms:cpu=false \
  --@zml//platforms:rocm=true \
  --config=release -- \
  --model=/var/models/deepseek-ai/DeepSeek-V4-Flash/ \
  --batch-size=1 \
  --token-batch-size=256 \
  --max-context-len=2048 \
  >"${trace_dir}/full.log" 2>&1
```

The full-model JIT launched twelve model-program compilations concurrently:
two sample programs and ten layer programs covering two shape groups. Four
small expert-packing programs were compiled afterwards.

The original timer and HLO-dump timestamps produced this breakdown. "Pre-config"
is the interval from compilation start through the
`before_config_assignment` dump; the next column covers configuration and
autotuning; "total" is `GpuCompiler::RunHloPasses` or the corresponding
program-completion interval. Because programs overlap, values across rows must
not be summed.

| Program | Token-batch 1: pre-config / config / total | Token-batch 256: pre-config / config / total |
| --- | ---: | ---: |
| Sample | 0.11 / 13.67 / 14.58 s | 0.40 / 15.61 / 16.03 s |
| Embedding and first layer | 79.67 / 2.58 / 82.28 s | 116.72 / 10.29 / 127.27 s |
| HCA layer | 77.79 / 4.48 / 82.30 s | 114.81 / 12.07 / 127.23 s |
| Full-attention layer | 80.34 / 1.90 / 82.27 s | 115.82 / 6.93 / 122.86 s |
| CSA layer, f32 indexing | 79.39 / 2.58 / 82.18 s | 115.81 / 14.16 / 130.02 s |
| CSA layer, s64 indexing | 77.61 / 4.37 / 82.18 s | 115.78 / 14.20 / 130.04 s |

The preserved HLO dump names map this set to modules 0000 and 0001 for sample,
0006 and 0007 for full attention, 0008 and 0010 for embedding plus first layer,
0009 and 0013 for HCA, and 0011, 0012, 0014, and 0015 for the CSA variants.

The packing programs took only 0.24 to 0.48 seconds each. Final backend code
generation took roughly 0.75 to 1.19 seconds per layer and was not on the main
critical path.

The completion timestamps formed synchronized waves:

1. All twelve model programs started at approximately 09:41:45.
2. The sample programs completed around 09:42:00 to 09:42:02.
3. The first shape group of layer programs completed together around 09:43:08.
4. The second shape group completed around 09:43:48 to 09:43:56.
5. Packing and model compilation completed around 09:44:00.

This pattern was evidence of shared worker-pool saturation rather than twelve
independent serial critical paths. PJRT supplies a compilation thread pool, and
`PriorityFusion` also schedules per-instruction work on the compiler's pool.
Concurrent outer modules and their inner priority computations therefore
compete for the same CPU resources.

### The compilation-statistics blind spot

The relevant source path is:

```text
GpuCompiler::RunHloPasses
  OptimizeHloModule
    RunFusionPasses
      FusionPipeline
        PriorityFusion::RunImpl
          CombinedGpuPerformanceModel::TryFindBestTilingForFusion
            GpuPerformanceModelWithIndexingAnalysis::TryFindTopKBestTilingsForFusion
```

The source is split across `xla/service/gpu/gpu_compiler.cc`,
`xla/service/gpu/fusion_pipeline.cc`,
`xla/backends/gpu/transforms/priority_fusion.cc`, and
`xla/service/gpu/model/gpu_indexing_performance_model.cc`.

At the time of the trace, the pre-fusion pipeline received a
`CompilationStats` object, but the fusion pipeline did not. Consequently,
"Total runtime of HLO passes" omitted most fusion time. One CSA program
reported:

* 82.18 seconds in `GpuCompiler::RunHloPasses`.
* 2.68 seconds in the printed HLO-pass statistics.
* 2.58 seconds in `ConfigAssigner`.
* Approximately 79.5 seconds in the otherwise unreported interval dominated
  by fusion.

This is why the first statistics dump appeared to implicate autotuning even
though the wall-clock gap was elsewhere.

### Autotuning and thread-pool controls

The autotuning-enabled trace contained 4,571 candidate results, about 4,048
GEMM or BLAS candidates, 4,589 candidate IR-emission calls, approximately 245
seconds of aggregate IR emission, and approximately 53 seconds of aggregate
HSACO compilation. These totals overlap heavily because candidates and modules
compile concurrently; they are CPU-work totals, not wall time.

Controlled full-model runs gave the following results:

| Experiment | Full model compile | Conclusion |
| --- | ---: | --- |
| Initial instrumented run with autotuning | 2m10.967s | Model compile timer; the wider log window through packing was about 134s |
| Autotuning disabled | 1m56.772s | Useful improvement, but the dominant delay remained |
| Autotuning disabled, compilation parallelism `0` | 1m55.775s | Default shared pool remained the best tested setting |
| Autotuning disabled, compilation parallelism `1` | 10m2.861s | Serializing the pool was severely harmful |

With parallelism `0`, the two sample programs spent 1.34 and 2.36 seconds in
`RunHloPasses`. The first layer wave spent 76.0 to 76.7 seconds per program,
and the second wave spent 112.3 to 114.4 seconds. With parallelism `1`, those
waves expanded to 6.2 to 6.5 minutes and 9.7 to 10.0 minutes respectively.

An exploratory run with
`--xla_gpu_experimental_enable_tiling_propagation=true` took 3m17s, but another
process was active on the device. That result was discarded rather than
attributed to the flag. Other uncontrolled `llmd` runs ranged from 2m07s to
4m12s, reinforcing the need for the isolated benchmark below.

## Why a dedicated benchmark was necessary

End-to-end model startup includes model loading, input preparation, PJRT client
initialization, compilation of multiple programs, and possible competition for
the GPU. That made `llmd` startup time too noisy to guide small compiler
changes.

The `//xla/tools/multihost_hlo_runner:hlo_compile_benchmark_gpu` target added by
this change compiles captured HLO modules without executing them. It supports:

* Multiple measured and warm-up repetitions.
* Serial or concurrent compilation of multiple modules.
* Separate parse, compile, and batch wall times.
* Labels and run identifiers for A/B comparisons.
* Machine-readable CSV rows and summary statistics.

Repetitions in one invocation share a process and PJRT client. Fully cold
measurements require launching the binary in a new process for every sample.

The benchmark used for this investigation was built with:

```shell
bazel build \
  --spawn_strategy=local \
  --config=rocm_ci_hermetic \
  --config=baseline_x86_64 \
  //xla/tools/multihost_hlo_runner:hlo_compile_benchmark_gpu
```

A single cold sample can be collected with:

```shell
benchmark_bin="$(bazel info bazel-bin)/xla/tools/multihost_hlo_runner/hlo_compile_benchmark_gpu"
rocm_lib="${benchmark_bin}.runfiles/local_config_rocm/rocm/rocm_dist/lib"

env \
  TF_CPP_MIN_LOG_LEVEL=1 \
  XLA_FLAGS=--xla_gpu_autotune_level=0 \
  ROCPROFILER_QUEUE_INTERPOSITION=false \
  HIP_VISIBLE_DEVICES=1 \
  LD_LIBRARY_PATH="${rocm_lib}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
  "${benchmark_bin}" \
  --repetitions=1 \
  --parallelism=1 \
  --output_csv=/tmp/hlo-compile.csv \
  --label=candidate \
  /path/to/module.before_optimizations.txt
```

The development benchmark used
`module_0015.deepseek4_csa_layer.before_optimizations.txt` from the
parallelism-0 HLO dump. It was selected because it reproduced the long
shape-256 CSA compile in one process while remaining small enough to run many
times. It is a compiler input, not an after-optimization dump, so every run
executes the complete HLO optimization and GPU backend pipeline.

The CSV schema contains one `module` row per input and iteration, one `batch`
row per iteration, and `summary` rows. It records label, run ID, phase,
iteration, input path, parse time, compile time, batch wall time, count,
minimum, median, mean, p95, maximum, standard deviation, and status. Parse time
is kept separate from compilation.

One benchmark process can run several repetitions, but that measures a warm
process after the first iteration. The five-sample baseline and first-pass
numbers in this document launched five processes. A representative shell loop
is:

```shell
results="$(mktemp /tmp/hlo-cold.XXXXXX.csv)"
rm "${results}"

for run_id in 1 2 3 4 5; do
  append_flag=--append_csv=false
  if test "${run_id}" -gt 1; then
    append_flag=--append_csv=true
  fi

  env \
    TF_CPP_MIN_LOG_LEVEL=1 \
    XLA_FLAGS=--xla_gpu_autotune_level=0 \
    ROCPROFILER_QUEUE_INTERPOSITION=false \
    HIP_VISIBLE_DEVICES=1 \
    LD_LIBRARY_PATH="${rocm_lib}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
    "${benchmark_bin}" \
    --repetitions=1 \
    --parallelism=1 \
    --output_csv="${results}" \
    "${append_flag}" \
    --label=candidate \
    --run_id="${run_id}" \
    /path/to/module_0015.deepseek4_csa_layer.before_optimizations.txt
done
```

Autotuning was disabled to isolate host compiler work. It is important to keep
the device otherwise idle even when studying host compilation, because client
initialization and compiler activity can still be affected by another process.

## What profiling showed

Scoped XLA timers identified priority fusion as the dominant high-level phase,
but did not expose enough detail inside its symbolic tiling pipeline. Linux
`perf` call graphs on the benchmark isolated the host-side work.

### Reproducible `perf` collection

The low-overhead profile used a 99 Hz userspace cycle event and a 4 KiB DWARF
stack dump:

```shell
profile_dir="$(mktemp -d /tmp/hlo-compile-perf.XXXXXX)"

perf record \
  --freq=99 \
  --event=cycles:u \
  --call-graph=dwarf,4096 \
  --output="${profile_dir}/perf.data" \
  -- \
  env \
    TF_CPP_MIN_LOG_LEVEL=1 \
    XLA_FLAGS=--xla_gpu_autotune_level=0 \
    ROCPROFILER_QUEUE_INTERPOSITION=false \
    HIP_VISIBLE_DEVICES=1 \
    LD_LIBRARY_PATH="${rocm_lib}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
    "${benchmark_bin}" \
    --repetitions=1 \
    --parallelism=1 \
    --output_csv="${profile_dir}/result.csv" \
    --label=perf \
    /path/to/module_0015.deepseek4_csa_layer.before_optimizations.txt \
  >"${profile_dir}/run.log" 2>&1

perf report \
  --stdio \
  --input="${profile_dir}/perf.data" \
  --no-children \
  --sort=comm,dso,symbol \
  --percent-limit=0.05 \
  >"${profile_dir}/self-flat.txt"

perf report \
  --stdio \
  --input="${profile_dir}/perf.data" \
  --children \
  --sort=comm,dso,symbol \
  --percent-limit=0.05 \
  >"${profile_dir}/children-flat.txt"
```

A 997 Hz experiment generated a 5.91 GB file and lost 1,449 of 355,000
samples. Its broad phase ranking remained useful, but all final comparisons use
the lower-frequency zero-loss profiles. Higher sample rates were unnecessary
for this 30-to-50-second test.

One full-model recording was interrupted before `perf record` finalized its
header. The file was 966,326,944 bytes, but its `data_size` field was zero, so
`perf report` rejected it. A preserved copy was repaired by setting only the
four-byte `data_size` header field to `file_size - data_offset` (966,325,536
bytes). The repaired file reported 103,000 samples and zero lost samples. In a
new run, allowing `perf record` to exit after the benchmark is preferable to
repairing the header.

### Original full-model profile

The pre-optimization full-model profile showed that the top-k tiling search was
the dominant inclusive call path:

| Inclusive symbol | Samples |
| --- | ---: |
| `TryFindTopKBestTilingsForFusion` | 85.42% |
| `SymbolicTileAnalysis::ComputeTiledComputation` | 68.11% |
| `ComputeTiledInstructions` | 49.70% |
| `EvaluateTileStrides` | 20.42% |
| `SymbolicTileAnalysis::AnalyzeFusion` | 15.69% |
| `TiledHloInstruction::Create` | 9.55% |
| `IndexingMap::GetDimensionBounds` | 9.48% |

The top exclusive symbols made the nature of the overhead clearer:

| Exclusive symbol | Samples |
| --- | ---: |
| `memmove` | 8.94% |
| `ComputeTiledInstructions` | 8.48% |
| `IndexingMap::GetDimensionBounds` | 7.40% |
| `mi_malloc` | 5.53% |
| `SymbolicTiledHloInstruction::symbolic_tile` | 4.86% |
| `TiledHloInstruction::Create` | 4.73% |
| `SymbolicTile::size_map` | 2.56% |
| MLIR `StorageUniquer::get` for `SymbolicExprStorage` | 2.52% |
| `SymbolicMap::Evaluate` | 2.29% |
| `SymbolicExpr::Evaluate` | 2.06% |
| Integer `SmallVector` move assignment | 1.82% |

The profile did not point to one slow mathematical operation. It showed the
same short graph and symbolic values being allocated, copied, hashed,
validated, and evaluated for every candidate.

### Profile after the first streaming pass

Once candidate materialization and several obvious copies were removed, the
relative bottleneck shifted. The high-frequency test-tube profile compiled in
36.674 seconds under profiling and attributed:

| Inclusive region | Samples |
| --- | ---: |
| `SymbolicTileAnalysis::AnalyzeFusion` | 48.15% |
| Flat-tiling enumeration and callback | 40.22% |
| `AnalyzeFromInstruction` | 38.34% |
| `ComputeTiledComputation` | 29.14% |
| `ComputeTiledInstructions` | 19.86% |
| `SetSymbolicTile` | 19.52% |
| `SymbolicTile::FromIndexingMap` | 17.32% |
| `ComputeOutputToInputIndexing` | 14.81% |
| `GetBitcastMap` | 8.97% |
| `GetOperandIndexingMaps` | 7.45% |

The leading exclusive symbols at this stage were
`ComputeTiledInstructions` (7.47%), `EvaluateSymbolicExprs` (4.52%),
`memmove` (3.61%), integer `SmallVector` assignment (3.01%), `malloc`
(2.99%), and `ComputeTiledComputation` (2.70%). This profile motivated both
analysis-scoped caches and the later concrete-node construction work.

### Final profiles

The first reliable post-cache profile used 199 Hz DWARF call graphs. It compiled
in 32.272 seconds, produced 62,000 samples in a 532.6 MB file, and lost no
samples. Its leading exclusive symbols were:

| Exclusive symbol | Samples |
| --- | ---: |
| `ComputeTiledInstructions` | 7.90% |
| `EvaluateSymbolicExprs` | 5.40% |
| `malloc` | 3.58% |
| `memmove` | 3.15% |
| Integer `SmallVector` assignment | 2.99% |
| `ComputeTiledComputation` | 2.84% |

After direct construction of final tile-value vectors, a 99 Hz profile compiled
in 32.265 seconds, produced 30,905 samples in a 142.1 MB file, and lost no
samples:

| Exclusive symbol | Samples |
| --- | ---: |
| `ComputeTiledInstructions` | 10.89% |
| `EvaluateSymbolicExprs` | 4.12% |
| `malloc` | 3.29% |
| `ComputeTiledComputation` | 2.94% |
| `memmove` | 2.62% |
| allocator `_int_malloc` | 2.14% |
| allocator `_int_free` | 2.12% |
| `TiledHloInstruction` constructor | 1.75% |
| Tiled-graph traversal | 1.60% |
| `IndexingMap` destructor | 1.53% |
| Symbolic-expression `SmallVector` assignment | 1.41% |
| Integer `SmallVector` assignment | 1.34% |

Those profiles predated the final common-layout bitcast fast path. After
rebuilding the benchmark from the exact squashed tree, the final 99 Hz profile
compiled in 31.313444 seconds, produced about 29,000 samples in a 134.5 MB
file, and lost no samples:

| Final-tree exclusive symbol | Samples |
| --- | ---: |
| `ComputeTiledInstructions` | 14.29% |
| `EvaluateSymbolicExprs` | 4.85% |
| `malloc` | 4.22% |
| `ComputeTiledComputation` | 3.32% |
| `memmove` | 3.01% |
| allocator `_int_free` | 2.54% |
| allocator `_int_malloc` | 2.12% |
| Runtime-estimation visitor | 1.75% |
| `IndexingMap` destructor | 1.43% |
| Tiled-graph traversal | 1.34% |

In the final inclusive report, `ComputeTiledComputation` accounted for 34.46%,
`ComputeTiledInstructions` 24.66%, `AnalyzeFusion` 10.47%, operand-indexing
lookup 7.75%, runtime estimation 6.83%, and `GetBitcastMap` 1.76%. The earlier
post-cache, pre-fast-path profile attributed 6.41% inclusive time to
`GetBitcastMap`; the comparison is consistent with the final speedup, while the
usual sampling and run-to-run caveats still apply.

Percentages are distributions, not absolute time. For example,
`ComputeTiledInstructions` rose as a percentage in the last profile even though
wall time did not regress, because surrounding work became cheaper.

The representative module evaluated approximately 31,000 valid tilings. Most
of them reached runtime estimation, so an optimization that only rejects more
candidates was unlikely to address the observed cost.

## Optimization layers

### Stream the tiling search

The old path materialized all possible flat tilings, converted them into
structured `Tiling` objects, filtered them, and then flattened them again during
cost-model evaluation.

`ForEachFlatTilingForInputSpace` now enumerates the Cartesian product into one
reused buffer and calls a callback for each candidate. `ForEachValidFlatTiling`
checks constraints directly on the flat parameters and immediately sends valid
candidates to the cost model. The existing APIs that return vectors remain
available and are implemented on top of the streaming traversal.

The hot path can also tell `ComputeTiledComputation` that constraints have
already been checked. This removes repeated structural conversions and
constraint evaluation without changing candidate order.

More concretely, the retained hot path is:

```text
ForEachFlatTilingForInputSpace
  reusable FlatTiling buffer
  -> FlatParametersSatisfyConstraints
  -> ComputeTiledComputation(flat parameters, constraints already checked)
  -> EstimateTiledRunTimeDataImpl
  -> retain only the current top-k candidates
```

The callback returns `absl::Status`, so an error or unsupported condition stops
enumeration immediately. Multi-output fusions that return the existing
documented `Unimplemented` status continue to be skipped exactly as before.
The old `GetFlatTilingsForInputSpace`, `GetValidTilings`, and structured
`Tiling` overload remain for callers that need materialized values.

### Reuse symbolic analysis results

Symbolic analysis repeatedly requested equivalent information while walking a
fusion. Three caches now live for exactly one top-level
`AnalyzeFromInstruction` call:

* Operand indexing maps, keyed by `HloInstruction`.
* Derived `SymbolicTile` values, keyed by indexing-map value.
* Reshape and bitcast indexing maps whose safety has already been verified.

The caches are deliberately scoped to one analysis. They cannot outlive the HLO
graph, MLIR context, or indexing maps that provide their keys. The symbolic-tile
cache delivered the largest individual improvement in this pass, approximately
three seconds on the captured module.

`OperandIndexingCache` owns the vectors returned to recursive analysis, so its
references remain valid for the whole call. `SymbolicTileCache` uses pointers to
the stable `IndexingMap` objects already owned by symbolic instructions, but its
hash and equality compare map values. Equivalent maps therefore share one
derived tile without extending the lifetime of an indexing map. The verified
reshape set is only populated after the expensive bitcast check succeeds.

Fusion membership is also computed while the symbolic graph is built and
stored as optional metadata. Concrete tiled nodes inherit it. Manually created
or test nodes retain the old adaptor lookup as a fallback.

This cache reduced the two fusion-adaptor membership implementations from about
2.25% combined in an intermediate profile to about 0.38% combined in the final
profile.

### Make symbolic expression evaluation cheaper

Tile sizes, offsets, and strides evaluate slices of one symbolic map. The new
component evaluator writes results directly into pre-sized storage and avoids
constructing a temporary symbolic map or copying parameter vectors.

`EvaluateSymbolicExprs` handles constants and variables directly and only enters
the recursive evaluator for compound expressions. The recursive implementation
operates on expression storage nodes instead of repeatedly wrapping child
expressions.

`EvaluateTileMapComponent` slices the size, offset, or stride expressions from
the original `SymbolicMap` and uses `resize_for_overwrite` before batch
evaluation. It does not build a component `SymbolicMap`, copy parameters into a
new vector, or value-initialize result elements that will immediately be
overwritten.

Small constants in the range `[-1, 128]` and variable IDs in `[0, 63]` use a
thread-local leaf cache. Cache entries include an MLIR context and a global
storage epoch. Registering or removing symbolic-expression storage advances the
epoch, preventing stale expressions from being reused after context storage is
recreated.

Stride evaluation first checks whether any parameter actually requires
clamping. Concrete computation construction groups the distinct upper-bound
vectors and clamps the flat parameters once per group, rather than once per
instruction and candidate.

`IndexingMapConstraintsCanBeIgnored` also materializes dimension bounds and
symbol bounds once per indexing map instead of allocating and copying them for
every result expression.

### Use dense candidate-local storage

Symbolic tiled instructions now have dense IDs. Candidate construction uses the
IDs for:

* Symbolic-to-concrete instruction lookup.
* Sparsely populated tile-size values behind a dense index.
* Sparsely populated tile-stride values behind a dense index.

This replaces pointer-keyed hash lookups in the inner search loop. Top-level
tile values are evaluated once and passed to concrete node construction when
available.

The `DenseTileValueMap` separates a dense vector of value indices from compact
storage for the values themselves. This is important because an empty tile
vector is a valid scalar result and must remain distinguishable from "no cached
value." Symbolic instructions, including instructions nested in regions, receive
one immutable ID when `SymbolicTileAnalysis` is constructed. Top-level
instructions also receive an ID for their distinct tile-parameter upper-bound
pattern.

`CreateUncheckedFromSymbolicTile` writes evaluated sizes and strides directly
into the final `TiledHloInstruction`. Symbolic analysis has already established
the relevant invariants, so the hot path does not repeat the public factory's
full validation or build temporary vectors merely to move them into a node.

Factory parameters use rvalue references throughout this path. The public
checked factory remains available, while `CreateUnchecked` and
`CreateUncheckedFromSymbolicTile` are reserved for analysis-owned construction
after shape, indexing-map dimensionality, and runtime-variable counts have been
established.

### Recycle short-lived tiled instructions

Every candidate creates and destroys a graph of `TiledHloInstruction` objects.
A bounded thread-local free list now recycles up to 4,096 node allocations per
worker. It only owns raw storage between destructor and constructor calls; all
members still have their normal C++ lifetimes. The fixed bound prevents an
unusually large fusion from retaining an unlimited amount of memory.

Both sized and unsized delete are implemented. A nonstandard allocation size
falls back to the global allocator, so inheritance or compiler-selected sized
deallocation cannot put incompatible storage into the free list. The cache is
thread-local because candidate graphs are constructed and destroyed by compiler
workers; no node is transferred between allocation caches.

After direct tile-value construction, a low-overhead profile showed
small-vector assignment for integer tile values falling from about 2.99% to
1.34% of samples, while `memmove` fell from about 3.15% to 2.62%.

### Reduce repeated cost-model work

The tiled graph traversal uses an inline-capacity `llvm::SmallVector` worklist
while preserving the original traversal order and algorithm. The top-k search
memoizes `FlopsPerElement` by HLO instruction for the duration of one search and
uses the cached fusion-membership bit described above.

Candidates are retained incrementally, and the public behavior remains the
same: candidates are stably ordered by estimated execution time and only the
best requested entries survive.

Incremental retention bounds candidate memory by `top_k` instead of keeping all
valid candidates until the search ends. Appending a candidate, stable-sorting
the short retained vector, and truncating it preserves the result of a global
stable sort followed by truncation, including tie order. Negative `top_k`
values are rejected explicitly.

### Fast-path common reshape bitcasts

`GetBitcastMap` recognizes shapes with untiled, non-physical, descending layouts
and directly constructs the reshape indexing map. This avoids general layout
normalization and transpose analysis for the common dense layout used heavily
by the workload. Non-matching layouts continue through the existing general
path.

The fast path requires both input and output layouts to have no tiles, no
physical shape, and a descending `minor_to_major` order. It then delegates map
construction to the existing `ComputeReshapeIndexingMap` and wraps the result
with the input tensor sizes. It does not introduce a second reshape formula.

## File and API map

| Area | Retained change |
| --- | --- |
| `xla/codegen/tiling/experimental/tiling_space_utils.*` | Callback-based flat tiling enumeration with a reusable candidate buffer |
| `xla/codegen/tiling/symbolic_tile_analysis.*` | Flat candidate APIs, analysis-scoped caches, dense IDs, grouped bounds, dense candidate values, and direct concrete construction |
| `xla/codegen/tiling/symbolic_tiled_hlo_instruction.h` | Dense instruction ID, bounds-pattern ID, and cached fusion membership |
| `xla/codegen/tiling/symbolic_tile.*` | Batch component evaluation, direct result storage, fast no-clamp stride path, and shared clamped parameters |
| `xla/hlo/analysis/symbolic_expr.*` | Direct storage-node recursion, batched terminal evaluation, and epoch-safe thread-local leaf cache |
| `xla/hlo/analysis/indexing_analysis.cc` | Common descending-layout reshape-bitcast fast path |
| `xla/codegen/tiling/tiled_hlo_instruction.*` | Rvalue factories, unchecked analysis-owned construction, direct symbolic-tile construction, optional membership metadata, and bounded node recycling |
| `xla/service/gpu/model/gpu_indexing_performance_model.cc` | Streaming top-k evaluation, bounded stable candidate retention, FLOP memoization, cached membership use, and inline worklist storage |
| `xla/tools/multihost_hlo_runner/hlo_compile_benchmark_main.cc` | Repeatable compile-only GPU HLO benchmark with CSV output |

## Results

The primary comparison used the same captured DeepSeek-V4 CSA HLO, MI300X,
GPU 1, hermetic ROCm configuration, one compiler process per sample, and
`--xla_gpu_autotune_level=0`.

| State | Cold compile measurements | Mean | Change from base |
| --- | --- | ---: | ---: |
| Branch base `86dd5bb22f` | 50.319, 50.850, 49.695, 50.766, 49.576 s | 50.241274 s | baseline |
| Initial streaming and symbolic-tiling pass | 45.117, 44.568, 44.710, 45.558, 44.429 s | 44.876282 s | -10.679% |
| Single-flight experiment, later reverted | 45.997, 45.476, 45.300, 45.090, 45.663 s | 45.505100 s | -9.467% |
| Final squashed tree | 30.835, 29.579, 30.432, 31.718, 30.281 s | 30.569076 s | -39.155% |

The final tree is 1.644x as fast to compile as the branch base and 31.881%
faster than the initial retained pass. The single-flight experiment was 1.401%
slower than that initial pass, which is why its add and revert cancel in the
squashed history.

### Development measurement ledger

The following selected one-process runs show how the optimization was
developed. They are directional measurements, not an additive attribution
model: some rows were A/B experiments, some include `perf` overhead, and some
were reverted before the next row. Repeated or five-process measurements carry
more weight than isolated smoke runs.

| Candidate or profile label | Compile time | Interpretation |
| --- | ---: | --- |
| Base, five-process mean | 50.241274 s | Stable starting point |
| Initial retained symbolic-tiling pass, five-process mean | 44.876282 s | Streaming candidates and removing structured-tiling churn |
| Single-flight, five-process mean | 45.505100 s | Rejected; slower than the initial pass |
| Fast expression evaluation smoke | 44.063309 s | Direct expression work begins to help |
| Leaf-expression cache smoke / profile | 40.921754 / 40.220918 s | Small constant and variable interning cache |
| First node-pool smoke | 40.339596 s | Directionally useful allocation recycling |
| Dense instruction IDs smoke | 37.074484 s | Replaces hot pointer hash lookups |
| Dense-ID profile | 37.848460 s | Profiling overhead included |
| Grouped stride bounds smoke / profile | 37.970143 / 37.145773 s | Clamp once per bounds pattern |
| Direct component evaluator smoke | 37.454543 s | Avoid temporary symbolic maps |
| Node-pool cap 4,096 smoke | 37.002944 s | Retained bounded pool policy |
| Dense tile-value smoke / profile | 36.829776 / 37.591959 s | Dense sparse-value representation |
| Unchecked node construction smoke / profile | 36.435607 / 36.161643 s | Avoid repeated constructor precondition checks |
| Batch expression evaluation smoke | 36.225793 s | Constants and variables handled directly |
| Storage-node recursive evaluation smoke | 35.904156 s | Avoid child wrapper churn |
| Inline checked-map smoke | 35.511402 s | Final dense map lookup path |
| Inline-capacity worklist runs | 35.312878 / 35.958981 s | Retained minimal traversal change |
| Original `std::vector` worklist comparison | 36.528833 s | A/B reference for the worklist change |
| Operand-indexing cache | 35.208671 s | Analysis-scoped HLO indexing reuse |
| Fusion-membership cache | 35.273833 s | Small wall-time change, clear profile reduction |
| Symbolic-tile cache runs | 32.039418 / 32.831845 s | Largest individual late-stage gain |
| Verified reshape cache | 32.630846 s | Avoid repeated safety checks |
| Post-cache 199 Hz profile | 32.271587 s | 62,000 samples, zero lost |
| Direct final tile-value construction | 32.157852 s | Avoid intermediate vector moves |
| Direct-value 99 Hz profile | 32.265407 s | 30,905 samples, zero lost |
| Per-search FLOP cache | 32.389223 s | Retained; within single-run noise |
| Final bitcast fast path, five-process mean | 30.569076 s | Exact squashed tree |
| Final-tree 99 Hz profile | 31.313444 s | About 29,000 samples, zero lost |

The biggest robust transitions were the initial streaming pass, the set of
dense evaluation and allocation changes that moved the workload into the
35-to-37-second range, the analysis-scoped symbolic-tile cache that moved it to
about 32 seconds, and the common-layout bitcast path that produced the final
30.569-second mean.

The latest profile still identifies concrete tiled-instruction construction
and symbolic expression evaluation as meaningful costs. The distribution is
flatter than the original profile, suggesting that future work should begin
with fresh profiles rather than assuming one remaining dominant function.

## Experiments intentionally not retained

Several ideas were measured and then removed:

* A single-flight cache around priority-fusion tiling analysis produced a
  45.505100-second five-process mean versus 44.876282 seconds without it. It
  could also couple unrelated compilations, so both the implementation and its
  revert disappear from the final squashed diff.
* A module-wide analysis-result cache had lifetime and invalidation complexity
  that was not justified by its 44.427676-second smoke result.
* Per-candidate composite memoization added lookup overhead without enough
  reuse. The size-memo smoke run was 40.469118 seconds in a stage whose simpler
  changes were already around 40 seconds.
* Merging register-fit and runtime-estimation traversals was neutral to slower
  in A/B runs. Two smoke results were 36.521761 and 35.746468 seconds, within
  the spread of the surrounding implementation and not a robust win.
* Prechecking register fit did not help because nearly all candidate tilings
  were accepted by the representative workload. Instrumentation observed about
  31,140 candidates reaching evaluation, leaving little reject-only work to
  eliminate.
* A larger worklist traversal rewrite was replaced by the smaller inline-vector
  change to preserve ordering and reduce review risk.
* Directly initializing a local tile-value vector measured 36.326819 seconds
  but did not remove the constructor move. It was superseded by
  `CreateUncheckedFromSymbolicTile`, which writes the final fields directly.
* Process-wide or module-wide caches were rejected even when a smoke timing
  looked neutral, because HLO mutation, MLIR-context lifetime, and concurrent
  compilation make invalidation part of the correctness contract.

These negative results matter because many apparently redundant operations are
cheap individually. New caches or traversal changes should demonstrate reuse
on a repeatable HLO benchmark before being added to this hot path.

## Validation

The focused suite for the final implementation is:

```shell
bazel test \
  --spawn_strategy=local \
  --config=rocm_ci_hermetic \
  --config=baseline_x86_64 \
  //xla/hlo/analysis:indexing_analysis_test \
  //xla/hlo/analysis:symbolic_expr_test \
  //xla/codegen/tiling:symbolic_tile_test \
  //xla/codegen/tiling:symbolic_tile_analysis_test \
  //xla/codegen/tiling:symbolic_tiled_hlo_instruction_test \
  //xla/codegen/tiling:tiled_hlo_instruction_test \
  //xla/service/gpu/model:gpu_indexing_performance_model_test
```

All seven targets passed after the final bitcast fast path was added. As with
any compile-time optimization, correctness tests should be paired with both a
cold-process compile benchmark and an end-to-end throughput check before
deployment.

The exact final source tree also successfully rebuilt
`//xla/tools/multihost_hlo_runner:hlo_compile_benchmark_gpu` with the hermetic
ROCm and baseline x86-64 configurations before the final five-process timing
and 99 Hz profile were collected.

### Correctness and lifetime invariants

The implementation relies on the following invariants:

* Callback enumeration visits candidates in the same order as materialized
  enumeration.
* Flat candidate constraints are checked once before the hot overload is told
  that they are satisfied.
* Analysis caches do not escape one recursive symbolic-analysis invocation.
* Indexing-map pointer keys refer to maps owned by symbolic instructions for the
  entire cache lifetime; hash and equality use map values.
* Thread-local expression leaf entries are invalidated by MLIR context and
  storage epoch.
* Dense instruction IDs are assigned once after the full symbolic graph,
  including regions, is owned by the analysis.
* An empty cached tile vector remains distinguishable from a missing cached
  value.
* Unchecked concrete-node factories are only used after symbolic analysis has
  established the checks performed by the public factory.
* Recycled nodes contain only raw storage while on the free list, and the list
  is bounded and thread-local.
* Cached fusion membership is optional, preserving adaptor lookup for manually
  constructed and test nodes.
* Incremental stable top-k retention preserves candidate order for equal
  execution-time estimates.
* The bitcast fast path accepts only simple dense descending layouts and uses
  the existing reshape-indexing-map constructor.

### Validation not yet performed

The focused tests and compile-only benchmark do not replace broader validation.
At the time of writing:

* A full `//xla/pjrt/c:pjrt_c_api_gpu_plugin` rebuild from the exact final tree
  had not been used as the validation signal for this optimization pass.
* The final tree had not been rerun through the complete DeepSeek-V4 `llmd`
  startup and steady-state tokens-per-second measurement on an otherwise idle
  machine.
* CUDA and non-x86 builds had not been exercised locally, even though the
  changed symbolic-tiling code is backend-shared.
* The benchmark covers one deliberately difficult CSA module. Other large
  fusions should be sampled before generalizing the exact percentage.

## Investigation artifact index

The following paths were workstation-local and are not checked into the
repository. They are listed to preserve the provenance of the tables above:

| Evidence | Local path |
| --- | --- |
| Initial full-model timers and statistics | `/tmp/deepseek-v4-xla.6vMhyE` |
| Full-model run without autotuning | `/tmp/deepseek-v4-xla-no-autotune.Z2vila` |
| Compilation parallelism 1 | `/tmp/deepseek-v4-xla-priority-fusion-p1.1RF2Ex` |
| Compilation parallelism 0 and captured HLO | `/tmp/deepseek-v4-xla-priority-fusion-p0.mbZToh` |
| Repaired pre-optimization full-model call graph | `/tmp/deepseek-v4-perf-callgraph.AXJY6Z/perf.repaired.data` |
| Five-process branch baseline | `/tmp/deepseek-csa-cold-baseline.nLAzZb.csv` |
| Five-process initial retained pass | `/tmp/deepseek-csa-cold-symbolic-tiling.itKtXX.csv` |
| Five-process single-flight experiment | `/tmp/deepseek-csa-cold-singleflight-5.csv` |
| High-frequency transition profile | `/tmp/deepseek-csa-inline-worklist-perf.X9isUf` |
| Zero-loss post-cache profile | `/tmp/deepseek-csa-post-opt-perf.MFK6bI` |
| Zero-loss direct-value profile | `/tmp/deepseek-csa-direct-values-perf.xCZmRE` |
| Exact final-tree five-process sample | `/tmp/deepseek-csa-final-5.nw0P4F/result.csv` |
| Exact final-tree 99 Hz profile | `/tmp/deepseek-csa-final-perf.WZ0kMb` |

Additional one-run CSV files and profiles use the
`/tmp/deepseek-csa-<experiment>` prefix corresponding to labels in the
development ledger.
