# Faster XLA compilation with measured executable quality

## Assessment

Prioritize eliminating repeated candidate-evaluation work, then improve search and parallelism. Use offline GPU measurements to find better configurations and retain them for subsequent compilations.

The source and existing profile support the direction in [FASTER.md](/home/benjamin/github/openxla/xla/FASTER.md), with several refinements:

- Candidate graph construction, expression evaluation, hashing, and destruction dominate the reported compiler work. `mi_malloc` adds 5.38% of sampled cycles outside the plugin’s symbol classification.
- The memory-pruning check evaluates tile sizes that surviving candidates evaluate again.
- Priority fusion already schedules work concurrently and caches fusion estimates. Measure contention, duplicate misses, and uneven task sizes before adding parallelism.
- Cost estimates influence **fusion formation as well as tiling selection**. Checking only winning tiles would miss executable changes.
- Increasing top-k alone does not ensure broader tuning: [the block-level tuner](/home/benjamin/github/openxla/xla/xla/backends/gpu/autotuner/block_level_emitter.cc:114) returns only the existing configuration when one is present.

The reported 53.91% of sampled cycles identifies an opportunity; it does not establish an equivalent wall-time reduction. “Best possible” will mean the fastest validated executable among the configurations and transformations tested, with the baseline always included.

## Experiments, in order

### 1. Establish a replayable baseline and verify pruning

- Freeze the XLA/ZML revisions, loaded plugin checksum, effective compilation options, model, GPU placement, and NUMA binding.
- Capture pre-optimization HLO for whole-module comparisons and representative fusion inputs for fast search-only replay.
- Add optional aggregate counters for enumeration, constraint rejection, memory pruning, materialization, expression evaluation, deduplication, allocation, and elapsed time. Include caller/pass identity and fusion fingerprints.
- Rank fusions separately by compilation cost and GPU execution cost. Initially cover those accounting for 80% of each.
- Add an exhaustive reference replay that disables memory pruning and records every candidate’s outcome. Check that the current bound never exceeds the corresponding full estimate and that pruning preserves top-k results.

**Decision:** repair any unsound bound before strengthening pruning. Keep detailed tracing and exhaustive comparison outside timed production runs.

### 2. Reuse evaluated values and storage

This is the first implementation experiment: relatively contained, with several independently measurable benefits.

- Introduce a workspace owned by one tiling search. Reuse tile buffers, instruction-index mappings, operand-read accumulators, and deduplication storage.
- Compute tile sizes once per candidate and share them between pruning and full evaluation.
- Hoist instruction metadata, shape sizes, traversal metadata, and FLOPs into indexed arrays.
- Keep size evaluation keyed by the original parameters. Only stride evaluation uses the relevant clamped parameters and bounds identity.
- Measure concurrent misses in the existing fusion cache. If duplicate searches are significant, let simultaneous requests for the same existing cache key share one computation.

**Gate:** identical candidate outcomes, scores, ordered top-k results, and resulting fusion decisions.

### 3. Precompute expression evaluation

The current evaluator recursively visits expression trees for each result. Test an immutable, indexed expression program built once per analysis.

- Share common subexpressions within each parameter environment.
- Evaluate into reusable integer arrays without constructing MLIR objects.
- Preserve signed division, modulo, clamping, and error semantics.
- Compare against the existing evaluator on captured expressions and boundary cases.
- Measure preprocessing overhead as well as candidate savings; retain the existing path for searches where preprocessing does not pay back.

This can reduce both direct evaluation work and the cost of subsequent lightweight evaluation.

### 4. Replace temporary graphs with a compact cost-model representation

Build on the workspace rather than designing a second independent cost model.

- Add a compact candidate view containing indexed operands, concrete sizes/strides, roots, region relationships, and deduplication identities.
- Adapt the existing cost calculations to consume that view.
- Start with single-root fusions without nested regions or runtime-dependent indexing; extend according to measured workload coverage.
- Preserve exact deduplication semantics, including offset comparisons where required. Equal tile sizes or equal predicted costs do not establish equivalent kernels.
- Keep the current materialization path as the fallback for unsupported cases, including existing multi-output limitations.

**Interface:** internal workspace/view types and evaluator adapters; preserve the existing `TryFindTopKBestTilingsForFusion` return contract and emitter-facing representation.

Run both implementations on every candidate in replay tests. The search returns configuration and runtime data, so its winning candidate also need not retain a full graph.

### 5. Improve ordering and pruning without losing candidates

- Evaluate the baseline/default candidate first to establish an incumbent, then continue the complete search.
- Assign candidates their original enumeration ordinal and use it to resolve equal scores deterministically.
- Move existing rejection checks earlier when their inputs are available.
- Add stronger bounds only after proving conservativeness against the actual cost formula and testing against exhaustive replay.
- Memoize expression results by their actual dependencies. Collapse whole candidates only when their relevant semantics are equivalent.

Treat occupancy estimates, register heuristics, and apparent dominance as separate measured experiments unless they reproduce an existing rejection exactly.

### 6. Revisit parallelism after reducing per-candidate work

- Measure the existing fusion task queue, lock waits, cache misses, and slowest searches.
- First improve scheduling of independent searches while preserving serial graph mutation.
- If a few large searches dominate, evaluate candidates in bounded batches with worker-local workspaces and immutable metadata.
- Compare 1, 2, 4, 8, and 16 workers within NUMA node 0. Merge results deterministically and avoid nested thread-pool oversubscription.

**Decision:** retain parallelism only when compilation wall time improves without a material memory increase.


## Validation and acceptance

- **Compilation:** alternate baseline and experimental builds for at least five runs each, using the exact documented workload and equivalent cache snapshots. Record compilation wall time, CPU time, peak RSS, and per-fusion measurements. Measure cold autotuning separately.
- **Executable performance:** benchmark changed kernels and full GLM prefill/decode. Include short and long prompts, concurrency 1 and 16, fixed inputs, and warmup. Track throughput, time to first token, and decode latency.
- **Quality gate:** require correctness and no reproducible performance regression. Aim to resolve differences of approximately 1%; repeat inconclusive measurements rather than treating noise as improvement.
- **Determinism:** compare candidate outcomes, scores, ties, selected configurations, optimized HLO, and backend configurations. Any intentional difference must have a measured justification.
- **Focused Bazel tests:** run `//xla/codegen/tiling:symbolic_tile_analysis_test`, `//xla/codegen/tiling:symbolic_tile_test`, `//xla/service/gpu/model:gpu_indexing_performance_model_test`, and `//xla/service/gpu/model:triton_emitter_constraints_test`. Add `//xla/hlo/analysis:symbolic_expr_test`, `//xla/backends/gpu/transforms:priority_fusion_test`, and `//xla/backends/gpu/autotuner:block_level_emitter_test` as those subsystems change.

## Defaults

GLM on the two GB300 GPUs is the primary acceptance workload; implementations remain general XLA code. Extra tuning happens offline. Numerical precision and optimization quality remain unchanged unless an explicitly measured experiment justifies a change. Each experiment is independently reviewable and retained only after its gate passes.

