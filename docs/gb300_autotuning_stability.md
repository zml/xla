# GB300 compile-time and autotuning stability investigation

Status: experimental implementation, measured on 2026-09-04. This document
records the problem, evidence, experiments, current patch, and remaining
uncertainties. It intentionally distinguishes HLO/compiler evidence from the
incomplete end-to-end `llmd` performance check.

## Executive summary

Two symbolic-tiling changes materially reduce DeepSeek-V4 compilation time but
also change the host completion order of independently compiled modules. The
input HLO, all deterministic HLO stages before autotuning, candidate groups,
candidate sets, and order inside every candidate group remain semantically
identical. Fresh autotuning nevertheless observes the groups in a very
different global order. Many GB300 candidate timings are separated by only one
or two 32 ns timer quanta, so a single observation frequently changes the
selected kernel.

The resulting cache, rather than a pre-autotuning HLO or code-generation
semantic change, explains the initial throughput difference. Loading the same
complete autotune cache into the source and optimized plugins makes their
throughput approximately agree; changing the cache while holding the source
fixed changes throughput.

The current experimental patch combines two targeted mechanisms:

1. A per-GPU candidate transaction holds the existing GPU writer mutex across
   warmup, synchronization, and measurement.
2. Every candidate receives one correctness-checked profile. If the two
   fastest candidates are within `max(64 ns, 1%)`, they receive two additional
   profiles in round-robin order and are ranked by the median of three. The
   scratch-memory preference window is capped by the same noise interval.

The final HLO replay retains the compile-time improvement: **16,848.705 ms to
11,779.532 ms**, or **30.1% faster**, relative to the first-optimization-only
reference capture. The bounded adaptive policy reduced, but did not eliminate,
fresh-selection churn: roughly 35 changed selections in an ordinary repeat
became 22--32 (mean 28) out of 202 against one fixed adaptive reference.

The final paired `llmd` check was stopped after four of five pairs at the
request of the operator. Its observed medians were 33.63 tok/s for the source
baseline and 33.88 tok/s for the adaptive optimized build, but this is an
incomplete, noisy experiment and **is not a claim of throughput equivalence or
improvement**.

## Scope and naming

The primary workload in this investigation is DeepSeek-V4-Flash on physical
CUDA device 3. The historical GLM-5.3-Flash data is used only as supporting
mechanism evidence.

The local branch lineage is:

| Name in this report | Commit | Meaning |
| --- | --- | --- |
| source baseline | `a1faac5a9c` | local CUDA base before both compile optimizations |
| compile optimization 1 | `e0edcbb4d0` | accelerate symbolic tiling compilation |
| compile optimization 2 | `e4b58a8c1e` | reuse/prune dominated symbolic tilings before materialization |
| current experiment | branch tip after the commits above | candidate transaction plus bounded adaptive measurement |

The original equivalent optimization commits also exist as `5b1a2489c7` and
`545a8ff3e2` in the other worktree. The locally rebuilt commits have different
IDs but the same optimization intent. A pre-existing plugin attributed to an
older source state was separately preserved at `86dd5bb22f`; it was not used
as an implicit substitute for the `a1faac5a9c` source baseline.

Before the branch's author/committer metadata was normalized, the two local
optimization trees were recorded as `221071acd9` and `53778c81dc`. Rewriting
their commit metadata changed their IDs to the values in the table without
changing either source tree. Retained build artifacts and logs keep their old
commit-qualified filenames as historical provenance.

The detailed HLO comparison uses the directory name `HLO-opt1-fresh` for the
reference and therefore measures the incremental effect of optimization 2 plus
the experimental autotuner changes. The initial `A0`/`C0` `llmd` comparison is
source-before-both versus optimized-after-both. These are different
comparisons and should not be merged into one percentage.

No cross-run autotune cache was loaded in the fresh HLO measurements. The
cache-crossover experiment deliberately loaded a complete cache as its
independent variable. Bazel builds used the disk cache at
`/home/hugo/.cache/bazel-disk`; Bazel build time is not included in the XLA
compile timings.

One control issue was found outside this XLA branch. ZML's
`ZML_AUTOTUNE_CACHE_DIR` had targeted the deprecated, inactive
`xla_gpu_experimental_autotuner_cache_dir` field. The separate ZML commit
`85b329dd` maps it to `xla_gpu_per_fusion_autotune_cache_dir` and a validation
run produced per-fusion cache files. Fresh experiments still explicitly used
an empty/no-load cache; this wiring fix was not used to mix decisions from an
earlier process into a fresh measurement.

## Machine and dependency snapshot

The retained machine manifest is under
`/tmp/gb300-xla-autotune.7FpgFs/machine`. Its relevant snapshot is:

| Item | Value |
| --- | --- |
| Host | `gb300-1`, Linux 6.8.0-124-generic-64k, aarch64 |
| CPU | 144 ARM Neoverse-V2 cores, two sockets |
| GPU | physical GPU 3, NVIDIA GB300, UUID `GPU-9c1ccb87-8e30-8d7d-ea9b-eab15af99f0b` |
| GPU memory | 284,208 MiB |
| Driver | 595.71.05 |
| Captured compiler runtime | CUDA 13.3; detailed reference dump reports cuDNN 9.24 |
| Controlled paired `llmd` build | cuDNN 9.22 for both sides |

At the initial snapshot GPU 3 had no allocated memory and 0% utilization. Its
temperature was 39 C, power 176.29 W, SM clock 120 MHz, memory clock 3996 MHz,
and persistence mode was disabled. These values establish provenance; they do
not prove identical clocks or temperature at every individual kernel launch.

Two important packaged-plugin hashes are:

| Build | SHA-256 |
| --- | --- |
| `a1faac5a9c`, CUDA 13.3, cuDNN 9.22 | `74c25baae1a307626d0966427bb2c8dd7a3b0b933401635ff9db2592352efcc2` |
| optimization-2 tree, artifact named `53778c81dc`, before the adaptive patch, CUDA 13.3, cuDNN 9.22 | `f61c8fbe4a8f11e3deca6c9e9c6960ef7dcd21b283b59e07adc22fc03f4d7af8` |

For the final paired run, the adaptive plugin source artifact hashed to
`12ed080bfe0d91fe6a00d03951f5be1f11dd96ed527fda9f24f83ab757b5d8e0`.
The packaged copy actually loaded from the `llmd` runfiles hashed to
`b05f3a32a765ed8a8e1b1c01ae035373449dbaceac0bcf1e5ee13c2ecc766040`.
The difference is expected from packaging, and both values were captured to
avoid relying on a filename for provenance.

## The original problem

The first DeepSeek fresh-cache measurements showed the following:

| Case | tok/s samples | Median | Mean | Min--max | MAD | Median TTFT |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| A0: source baseline, fresh cache | 33.67, 33.94, 33.97, 33.67, 33.76 | 33.76 | 33.802 | 33.67--33.97 | 0.09 | 746.82 ms |
| C0: both optimizations, fresh cache | 31.83, 32.34, 32.36, 32.26, 32.18 | 32.26 | 32.194 | 31.83--32.36 | 0.08 | 748.35 ms |

The median decode difference is -4.44%. That observation alone does not show
that either optimization generated slower code: the two compilers performed
fresh, noisy online measurements and wrote different choices.

The cache/source crossover resolves that ambiguity:

| Source and loaded cache | tok/s samples | Median | Mean | Min--max | MAD |
| --- | --- | ---: | ---: | ---: | ---: |
| source baseline + cache A | 33.70, 33.34, 34.04, 33.72, 33.11 | 33.70 | 33.582 | 33.11--34.04 | 0.34 |
| optimized source + cache A | 33.90, 33.91, 33.33, 32.84, 33.52 | 33.52 | 33.500 | 32.84--33.91 | 0.38 |
| source baseline + cache C | 33.89, 33.40, 33.09, 33.88, 33.02 | 33.40 | 33.456 | 33.02--33.89 | 0.38 |
| optimized source + cache C | 33.78, 33.80, 33.85, 33.42, 33.86 | 33.80 | 33.742 | 33.42--33.86 | 0.05 |

The within-cache source differences are small relative to process-to-process
spread and change sign between the caches. The decisive pattern is that a
fixed complete cache largely decouples runtime behavior from which of the two
compiler sources loads it. Thus the initial A0/C0 gap is mediated by fresh
autotune choices, not by a difference already present in the input HLO.

Small hybrid-cache probes also showed that substituting selected entries could
restore the high-throughput region:

| Hybrid/control cache | Single observed tok/s |
| --- | ---: |
| first two changed entries substituted | 33.43 |
| last two changed entries substituted | 33.84 |
| all four substituted | 33.98 |
| changed entry 1 only | 33.85 |
| changed entry 2 only | 33.90 |
| changed entry 3 only | 33.97 |
| changed entry 4 only | 33.91 |
| unmodified control | 33.38 |

These are useful localization evidence, not defensible per-entry effect sizes:
each hybrid was measured once and the control itself was noisy. The run stopped
before a replicated cache delta-debugging matrix could identify a minimal
causal entry set.

Several exploratory `llmd` probes helped expose the size of ordinary runtime
variance and prevented a single favorable run from being promoted to a
conclusion:

| Probe | Observed decode tok/s | Interpretation |
| --- | ---: | --- |
| D0, first profile-transaction build | 32.13 | transaction alone did not restore the initial A0 median |
| E0, isolated profiler | 32.25 | same conclusion under another isolation variant |
| F0, fresh run with config dump | 33.83 | fresh choices can also land in the high-throughput region |
| G0, fresh no-dump processes | 33.76, 31.82, 33.75, 33.24, 32.32 | median 33.24; visibly multimodal/variable |
| J cache control | 33.20 | control for cache/unload investigation |
| J0 immediate unload | 33.31 | unloading immediately was not a material recovery |

Dumping, unloading, and transaction isolation were therefore not accepted as
standalone explanations for the throughput modes. The persistent selected
configuration remained the variable best supported by the crossover and HLO
evidence.

## HLO and autotuner comparison

The detailed comparison uses:

```text
reference: /tmp/gb300-xla-autotune.7FpgFs/runs/HLO-opt1-fresh.FEhh3J
candidate: /tmp/gb300-xla-autotune.7FpgFs/runs/HLO-opt2-adaptive2-full.kalECV
HLO report: /tmp/gb300-xla-autotune.7FpgFs/analysis/baseline-vs-adaptive2-full-hlo.json
autotune report: /tmp/gb300-xla-autotune.7FpgFs/analysis/baseline-vs-adaptive2-full-autotune.json
```

### Before kernel selection

| Comparison point | Result |
| --- | --- |
| input HLO | 12/12 byte-equivalent after normalization |
| every captured stage through `0014.autotuner.after_fusion-wrapper.before_conv-fp8-fallback` | 12/12 exact |
| `before_config_assignment` | 12/12 exact |
| candidate-before artifacts | 562/576 exact; 14/576 parameter-renames only, so 576/576 semantically equal |
| semantic autotune instruction groups | 60 versus 60 |
| candidate-set changes | 0/60 |
| candidate order inside a group | identical in 60/60 groups |
| candidate slots | 1,247 |
| matched successful measurements | 1,019 |

This is why the HLOs are identical through the autotuner wrapper: the two
symbolic-tiling optimizations change the cost and completion time of host-side
analysis, not the functional graph or the deterministic construction of a
given candidate set. The wrapper is still upstream of the nondeterministic
measurement result. Its output can therefore be identical even when the later
backend-config assignment is not.

The first semantic divergence occurs when measured results are committed as a
selected backend configuration. Different selected configurations then feed
different code-generation, copy insertion, scheduling, buffer-assignment, and
thunk decisions. Later differences are consequences of kernel selection, not
evidence that an earlier HLO pass diverged.

### Measurement and selection

The candidate sets were stable, but their global emission/measurement order
was not:

| Global group-order metric | Result |
| --- | ---: |
| same absolute position | 5/60 |
| pairwise inversions | 861/1,770 |
| longest common subsequence | 26 |

Across matched candidate measurements, adaptive/reference duration ratios
were: minimum 0.6224, p10 0.969, median 1.000, p90 1.094, maximum 1.5497. The
greatest common divisor of observed timing increments was 32 ns. This is a
large spread around a median of exactly one and many decisions are made only
one or two quanta apart.

Among 202 selected configuration keys, no key was missing and 32 choices
changed. The transitions were:

| Transition | Count |
| --- | ---: |
| block to block | 3 |
| block to native | 13 |
| GEMM to GEMM | 6 |
| native to block | 3 |
| Triton to GEMM | 3 |
| Triton to Triton | 4 |

After autotuning, 478/576 compared candidate artifacts remained exact. The 98
non-exact artifacts correspond to the 32 unique selected-choice changes.

### Downstream HLO and executable stages

| Stage | Exact modules |
| --- | ---: |
| first captured pre-scheduling/copy-insertion stage after selection | 1/12 |
| final optimized HLO | 1/12 |
| buffer assignment | 7/12 |
| live ranges | 7/12 |
| memory report | 7/12 |
| thunk metadata | 7/12 |
| thunk sequence | 1/12 |

The priority-fusion diagnostic dump was textually exact for 0/12 modules
because its trace records were emitted in a different order. Inspection of a
representative mismatch found the same `update_priority` records and costs in
a different sequence. The actual HLO after priority fusion was exact for
12/12 modules. Diagnostic trace order must therefore not be confused with HLO
semantics.

## Exact mechanism

The sequence without this patch is conceptually:

```text
module A host work finishes -> A candidate warmup -> release GPU mutex
module B host work finishes -> B candidate warmup/measurement
module A reacquires mutex    -> A candidate measurement
```

Candidate compilation runs concurrently on host worker pools. A
`ConfigRunner` and its `profiler_m_` belong to a particular autotuner/config
assignment instance; that mutex does not globally serialize all modules using
the same physical GPU. `GpuProfiler::Profile` calls `Execute` once for warmup
and once for the timed run. Each old `Execute` acquired the shared GPU writer
mutex independently, so exclusivity applied to one launch, not the warmup and
measurement pair.

The second symbolic-tiling commit changes which module finishes enough host
work first. It therefore changes which waiter arrives at the device mutex
first. Mutexes guarantee exclusion but not a canonical acquisition order.
Different global order changes the immediately preceding workload, clock and
power history, cache/TLB state, and other short-lived device conditions seen by
the one measured launch. A 32 ns-quantized observation then promotes a
different member of a near tie into the persistent autotune cache.

This explains both important observations:

* The HLO and candidate lists can be exactly the same while fresh selections
  differ.
* Optimization 2 specifically can perturb emission order without changing
  order inside an individual candidate group: it changes asynchronous module
  completion and mutex arrival, not deterministic group construction.

### Why the lock did not already wrap warmup and measurement

The exclusive lock was implemented in the generic executable run path. That
path sees only one `Execute` invocation at a time and cannot know whether it is
the warmup half or measurement half of a higher-level profiling transaction.
`GpuProfiler::Profile`, which does know that boundary, did not own the lock.

Simply acquiring `GetGpuMutex()` in `GpuProfiler::Profile` while leaving the
nested `Execute` behavior unchanged would recursively acquire a non-recursive
writer mutex and deadlock. The current implementation therefore:

1. acquires the per-executor GPU writer mutex at `GpuProfiler::Profile` entry;
2. keeps it through warmup, synchronization, and timed execution;
3. marks nested runs with `gpu_lock_already_held`;
4. makes both direct `GpuExecutable` and PJRT execution paths skip their nested
   acquisition only when that marker is present.

The marker is narrowly scoped to an explicit parent transaction. Ordinary
executions continue to take reader or writer locks as before, and independent
physical devices continue to use different mutexes.

### Why the correct lock scope is not sufficient

The transaction lock prevents another in-process XLA execution from appearing
between one candidate's warmup and measured run. It does not impose a stable
order on transactions originating from independently compiled modules. Their
host completion times still determine mutex arrival order, and the GPU state
before the transaction can still differ. It also cannot remove event-timer
quantization, clock/thermal drift, or unrelated external GPU use.

The experiment confirmed this distinction. Candidate-level transaction
isolation compiled in 13,229.727, 12,721.534, and 12,854.084 ms, but repeat
selection churn remained 37/202 and 34/202 against the first run. The global
emission order was not restored even though local warmup/measurement adjacency
was corrected.

### Why repeated execution of every candidate is not sufficient

Multiple measurements reduce independent random error, but these observations
are neither fully independent nor free:

* Back-to-back samples of one candidate share clocks, caches, temperature, and
  predecessor state, so their error is correlated.
* Measuring candidate A three times and then B three times adds temporal bias;
  round-robin A/B/A/B ordering is fairer for finalists.
* Repetition does not make global module arrival deterministic.
* Candidates separated by one 32 ns quantum have overlapping distributions;
  even a median can cross the selection boundary.
* Repeated isolated microkernels can converge to a steady state that differs
  from the kernel's state in the full model.
* Thousands of additional launches, synchronizations, and checks serialize on
  the tuning GPU and can erase compilation gains.

The broad median-of-three experiment illustrates both sides. It reduced repeat
churn to 19/202 and 21/202, but its compile batches were 12,487.373,
13,237.651, and 13,032.436 ms and it still did not make selection deterministic.
An associated five-process `llmd` diagnostic was also highly variable:
33.67, 31.83, 30.99, 33.26, and 30.84 tok/s (median 31.83). That policy was
reverted rather than treated as a solution.

## Experiment chronology and results

### Historical GLM evidence

The colleague's GLM-5.3-Flash capture changed compile time from 72.907 s to
46.283 s and reportedly changed throughput from about 1078 to 1055 tok/s
(-2.1%). This is not a DeepSeek performance target.

Its HLO and autotune comparison nevertheless showed the same mechanism:

* 21/21 matched input HLOs were identical.
* Candidate sets and within-group order were identical for 64/64 groups.
* Only 16/64 groups retained the same global position; there were 176/2,016
  pairwise inversions and the global longest common subsequence was 45.
* 39/64 selected winners changed; 30 of those pairs were tied or within 32 ns
  in at least one capture.
* 29/39 new winners were slower under the old measurements; 12 were at least
  1 us slower and 7 at least 2 us slower.
* Loading the earlier complete cache with the optimized compiler restored the
  earlier reported throughput.

Fresh selection churn also existed before these optimizations: 67/316 choices
changed across comparable pre-optimization runs. The optimization exposed and
amplified an existing sensitivity; it did not create all measurement noise.

### DeepSeek compile replay

| Variant | Batch time(s) | Interpretation |
| --- | --- | --- |
| first-optimization reference, full dump | 16,848.705 ms | detailed HLO reference |
| optimization 2, initial fresh run | 12,451.540 ms | 26.1% faster than reference |
| optimization 2 repeats | 12,309.094; 12,956.042 ms | ordinary optimized range |
| candidate transaction lock | 13,229.727; 12,721.534; 12,854.084 ms | fixes local interleaving, not selection order |
| serialize modules (`parallelism=1`) | 71,801.567; 68,215.728; 68,955.255 ms | stabilizes global order somewhat, destroys compile benefit |
| all candidates median-of-three | 12,487.373; 13,237.651; 13,032.436 ms | less churn, unnecessary extra profiles |
| unbounded adaptive finalists | 30,220.056; 26,680.621; 27,931.251; 29,527.106; 14,797.047 ms | rejected: too many near-tie finalists serialized GPU work |
| bounded adaptive, no dump | 13,703.658; 11,969.171; 14,286.107; 12,080.491; 12,986.554 ms | median 12,986.554 ms |
| bounded adaptive, full dump | 11,779.532 ms | final detailed comparison, 30.1% faster than reference |

The serial-module experiment is particularly diagnostic. It reduced changes
to 31/202 and 20/202; global order comparisons had only 41 and 23 inversions,
with longest common subsequences of 44 and 48. This confirms that host
scheduling controls global profiling order. However, 68--72 s batch times are
incompatible with retaining the compilation optimization, and selection churn
still remained.

The first adaptive implementation admitted every candidate inside the 64 ns /
1% band. On short kernels this could be a large finalist set, so repeated
profiles serialized enough GPU work to produce 26--30 s batches. Limiting the
policy to the best two initial candidates was the key compile-time bound.

### Selection stability of the final policy

An ordinary optimization-2 repeat changed 35/202 selected configurations. Four
bounded-adaptive repeats compared with one fixed adaptive reference changed
22, 32, 28, and 30 choices, for a mean of 28/202. This is a measurable but
modest improvement, not determinism. It is consistent with the policy's
purpose: spend extra samples only where the first observation already says the
decision is ambiguous.

### Per-autotuner noise and drift

The aggregate churn above hides two distinct questions: whether a candidate's
recorded duration moves, and whether that movement crosses a decision boundary.
The retained logs were therefore analyzed at both levels.

For candidate timing, a semantic group is the normalized instruction plus its
ordered candidate configuration sequence. This produces 50 unique groups and
60 occurrences per complete run because some semantic groups appear in more
than one module. For a candidate repeated within one run, the within-run median
is used. Across processes, robust coefficient of variation is
`1.4826 * MAD / median`. The cross-run span is `(max - min) / median`. A group
occurrence is ambiguous when the observed top-two margin is no larger than
`max(64 ns, 1%)`.

Persisted selection stability is calculated independently from all 202 exact
autotune-result HLO keys. Pairwise agreement is the most comparable metric when
cohorts contain different numbers of runs. “Unanimous fastest group” describes
the raw minimum-duration candidate in the logged group and intentionally does
not reproduce scratch preference.

| Policy | Runs | Mean pairwise selection agreement | Pairwise churn, min/median/max of 202 | Unanimous persisted keys | Candidate robust CV, median/p90 | Candidate span, median/p90 | Ambiguous group occurrences | Unanimous fastest groups |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ordinary optimized traces | 3 | 84.16% | 29 / 32 / 35 | 159/202 (78.7%) | 0.525% / 5.816% | 3.709% / 17.113% | 59.4% | 14/50 |
| candidate transaction lock | 3 | 83.50% | 29 / 34 / 37 | 156/202 (77.2%) | 0.183% / 2.117% | 1.766% / 11.915% | 71.7% | 10/50 |
| serial modules | 3 | 86.96% | 20 / 28 / 31 | 167/202 (82.7%) | 0.696% / 4.965% | 2.969% / 15.000% | 66.7% | 15/50 |
| all candidates median-of-three | 3 | 89.77% | 19 / 21 / 22 | 174/202 (86.1%) | 0.055% / 1.578% | 1.714% / 10.702% | 71.1% | 23/50 |
| unbounded adaptive, three parseable logs | 3 | 83.00% | 34 / 34 / 35 | 157/202 (77.7%) | 0.696% / 7.489% | 3.759% / 15.401% | 48.3% | 12/50 |
| bounded adaptive | 5 selections / 4 complete logs | 86.24% | 22 / 28 / 34 | 153/202 (75.7%) | 0.631% / 5.922% | 2.974% / 14.739% | 62.5% | 14/50 |

The first bounded run's persisted result file contains all 202 keys, but its
combined parallel log contains only 59 groups and one altered signature. It is
included in the exact selection statistics and excluded from timing
dispersion. The four remaining bounded logs each contain the same 50 semantic
groups and 60 occurrences.

Several conclusions follow from these numbers:

1. The candidate transaction removes a real source of duration noise. Relative
   to ordinary traces, median robust CV falls by 65% and p90 falls by 64%; the
   median span is approximately halved. Selection agreement nevertheless does
   not improve. Local warmup/measurement interleaving is therefore real but is
   not the dominant remaining cause of winner flips.
2. Serial module scheduling improves selection agreement while its aggregate
   candidate dispersion does not improve. That is direct evidence for a
   separate systematic ordering component: making group arrival order more
   repeatable helps which side of a near tie is observed, even when individual
   durations still drift.
3. Measuring every candidate three times gives the best timing and choice
   stability in this dataset. It reduces median robust CV to 0.055% and raises
   pairwise choice agreement to 89.77%, but still changes 19--22 of 202 keys.
   Repetition reduces noise; it cannot resolve quantized or systematically
   biased ties.
4. The bounded policy improves ordinary pairwise agreement from 84.16% to
   86.24% without the unbounded policy's compile-time explosion. It does not
   reduce dispersion across all candidates because only the two initial
   finalists receive more samples; the log contains their final medians mixed
   with single observations for every non-finalist.

All cohorts retain a **32 ns timing quantum**. For bounded adaptive, the median
top-two margin over all logged occurrences is exactly 0 ns after aggregation,
and 62.5% of occurrences are inside the 64 ns / 1% ambiguity window. Forty of
50 groups are ambiguous in at least half their occurrences, 23/50 in at least
three quarters, and 8/50 in every occurrence. Close decisions are therefore
the normal case for much of this workload, not a rare tail.

#### Noise by kernel-duration class

The drift is concentrated in short kernels. The following table uses the
median of each per-group statistic over the four complete bounded runs:

| Pooled best duration | Groups | Median fastest modal share | Median ambiguity frequency | Median candidate robust CV | Median candidate span | Median top-two margin |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| below 10 us | 33 | 66.7% | 75.0% | 3.39% | 10.27% | 16 ns |
| 10--50 us | 10 | 75.0% | 37.5% | 0.85% | 3.22% | 712 ns |
| at least 50 us | 7 | 75.0% | 25.0% | 0.47% | 1.26% | 1,024 ns |

The 16 ns table median does not imply sub-32 ns measurements; it is the median
of an even set containing group medians at 0 and 32 ns. Individual recorded
durations remain quantized to 32 ns.

The same split appears by group family. The 24 logged GEMM groups have a median
per-group candidate robust CV of 0.67% and median span of 2.05%. The 20 input
reduction groups have 3.47% robust CV, 10.27% span, and are ambiguous in a
median 75% of their occurrences. The remaining six loop/reduction groups are
too few for a population claim, but several are among the noisiest cases.

#### Most unstable logged groups under bounded adaptive

The IDs below are anonymous SHA-256 prefixes of the normalized instruction and
candidate sequence. “P90 candidate CV” is across candidates within that group;
“fastest regret” compares the candidate declared fastest in an occurrence with
the best pooled-median candidate. It is a diagnostic instability score, not a
measured model-runtime regression.

| Group ID | Group label | Candidates | Pooled best | Modal fastest share | Ambiguous occurrences | P90 candidate CV | Median candidate span | P95 observed-fastest regret |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `5bf39302c096` | GEMM fusion | 33 | 8.480 us | 50% | 50% | 12.64% | 10.38% | 10.95% |
| `abeefd2692f0` | loop add fusion | 3 | 7.648 us | 50% | 50% | 9.79% | 22.50% | 0.42% |
| `5c4f64e0bf9d` | input reduction | 3 | 7.136 us | 50% | 50% | 9.16% | 13.19% | 0.22% |
| `e99a6d739c72` | GEMM fusion | 17 | 6.784 us | 50% | 100% | 8.92% | 13.36% | 7.26% |
| `5de84d060cb4` | input reduction | 3 | 6.816 us | 50% | 75% | 8.03% | 10.33% | 4.19% |
| `dba4fc041564` | input reduction | 3 | 7.488 us | 50% | 50% | 7.96% | 12.82% | 13.80% |

This also shows why duration noisiness and winner noisiness must be reported
separately. A group can have large absolute drift while all candidates move
together and the fastest identity stays fixed; another can have low individual
dispersion but flip because its candidate distributions overlap at the timer
quantum.

#### Persisted-choice stability by autotuner key

Across the five bounded-adaptive result files, 153/202 keys use one config in
all runs and 49 are unstable. Of the unstable keys, 32 use two configurations,
15 use three, and 2 use four. Twenty-nine of the 49 cross emitter families
(for example native versus block-level); 20 switch only parameters inside the
same family.

By root operation, 22/44 reduction keys, 10/12 dot keys, and 7/28 conversion
keys are unstable. The dot population is small, but its 83.3% instability is
important because dots are more likely than tiny reductions to carry material
runtime cost. The two four-choice keys are both dots:

| Autotuner ID | Choices over five runs | Modal share | Choice family |
| --- | ---: | ---: | --- |
| `4b1459007e6b` | 4 | 40% | Triton-only tile/config changes |
| `6cf14d931fee` | 4 | 40% | Triton-only tile/config changes |

This exposes the principal limitation of `max_finalists=2`: across processes,
different initial samples admit different pairs, so four different candidates
can still win even though only two are remeasured in any one process.

The full sanitized data is checked in beside this report:

* `gb300_autotuning_variation_summary.json` contains cohort-level metrics and
  every pairwise selection comparison.
* `gb300_autotuning_variation_by_group.csv` contains all 50 bounded-adaptive
  timing groups.
* `gb300_autotuning_selection_stability.csv` contains all 202 persisted keys
  and their anonymous per-run config IDs.

The raw adaptive log stores only the final `ConfigProfile.duration`: for a
remeasured finalist this is the median, while for a non-finalist it is the one
initial sample. It does not retain the three constituent timings. Consequently
this analysis measures cross-process drift in the value used by selection; it
cannot estimate within-transaction sample correlation. A follow-up patch
should dump every sample, its round, transaction/group order, and whether it
was used as a finalist before changing the policy again.

### Final incomplete `llmd` check

Four baseline/adaptive pairs completed with the same cuDNN 9.22 dependency
before the fifth pair was stopped:

| Build | tok/s samples | Median | Mean | Min--max | Median TTFT |
| --- | --- | ---: | ---: | ---: | ---: |
| source baseline | 33.39, 33.87, 33.88, 32.70 | 33.63 | 33.460 | 32.70--33.88 | 746.30 ms |
| optimized + bounded adaptive | 33.91, 33.89, 32.58, 33.87 | 33.88 | 33.563 | 32.58--33.91 | 747.24 ms |

Pair 5's baseline process was interrupted and its adaptive partner did not run.
The four-pair distributions overlap and contain a low outlier on each side.
No runtime-performance conclusion should be drawn from this stopped test.

## Current implementation

### Candidate profiling transaction

`GpuProfiler::Profile` now owns the per-executor GPU writer lock over the whole
warmup/measure transaction. A `GpuExecutableRunOptions` bit communicates that
the parent already owns the lock. Both execution entry paths honor the bit,
preventing recursive lock acquisition.

This is the correct architectural level for local transaction isolation:
`Profile` knows the semantic boundary, while `Execute` remains usable outside
autotuning. Candidate compilation remains parallel and locks remain keyed by
executor/device rather than one process-wide mutex across GPUs.

### Bounded adaptive finalist measurement

The active defaults are:

```text
enabled = true
total_samples = 3
max_finalists = 2
absolute_window = 64 ns
relative_window = 1%
```

The algorithm is:

1. Profile every successfully compiled candidate once through the existing
   correctness path.
2. Rank successful candidates stably by initial duration.
3. Define `window = max(64 ns, fastest_initial * 0.01)`.
4. Retain at most the first two candidates no slower than
   `fastest_initial + window`.
5. If there are two finalists, obtain two additional samples for each in
   round-robin order.
6. Recheck input redzones after repeated executions when correctness checking
   is enabled.
7. Replace each finalist's duration with its median of three.
8. Preserve failures and stable initial candidate-order tie breaking.
9. Limit scratch-memory preference to
   `min(2 us, max(64 ns, selected_duration * 1%))`.

A clear winner receives no extra profiles. The two-candidate limit is a
deliberate compile-time/robustness tradeoff discovered by the unbounded
experiment.

### Tests and build validation

Architecture-independent scripted-profiler tests cover:

* a clear loser is not remeasured;
* near contenders are remeasured in round-robin order;
* a median rejects a timing outlier;
* the scratch-memory selector respects the adaptive noise cap.

The following passed:

```shell
bazel test \
  //xla/backends/autotuner:config_runner_test \
  //xla/backends/autotuner:config_selector_test \
  --disk_cache=/home/hugo/.cache/bazel-disk \
  --test_output=errors
```

The CUDA HLO compile benchmark and PJRT GPU plugin also built successfully.
The plugin was built against cuDNN 9.24 and again against the controlled cuDNN
9.22 dependency used for the paired `llmd` comparison. `git diff --check`
passes.

## Reproduction commands

### Build the CUDA benchmark/plugin

All local Bazel rebuilds should keep the home-directory disk cache:

```shell
bazel build \
  --disk_cache=/home/hugo/.cache/bazel-disk \
  --spawn_strategy=local \
  --config=cuda_nvcc \
  --config=baseline_arm64 \
  //xla/tools/multihost_hlo_runner:hlo_compile_benchmark_gpu \
  //xla/pjrt/c:pjrt_c_api_gpu_plugin
```

The original GB300 plugin packaging command used `--enable_workspace` as well.
Use the same dependency lock and cuDNN package on both sides of a runtime A/B
test.

### Replay the 12 captured DeepSeek inputs

The exact input paths are recorded as the 12 `module` rows in each retained
`compile.csv`. A fresh replay has this shape:

```shell
CUDA_VISIBLE_DEVICES=3 \
XLA_FLAGS="--xla_gpu_dump_autotune_results_to=/tmp/unique-run/autotune-results.textproto --xla_gpu_dump_autotune_logs_to=/tmp/unique-run/autotune-logs.textproto" \
"$(bazel info bazel-bin)/xla/tools/multihost_hlo_runner/hlo_compile_benchmark_gpu" \
  --repetitions=1 \
  --warmup_repetitions=0 \
  --parallelism=12 \
  --output_csv=/tmp/unique-run/compile.csv \
  --label=candidate \
  <the-12-before_optimizations-HLO-paths>
```

Use a new output directory and an empty/no-load autotune cache for every fresh
comparison. Do not compare a full-dump batch time with a no-dump batch time as
if instrumentation were free.

### Run the finite DeepSeek benchmark

```shell
CUDA_VISIBLE_DEVICES=3 \
bazel run \
  --disk_cache=/home/hugo/.cache/bazel-disk \
  --config=release \
  --@zml//platforms:cpu=false \
  --@zml//platforms:cuda=true \
  //llmd -- \
  --model=/var/models/deepseek-ai/DeepSeek-V4-Flash/ \
  --batch-size=1 \
  --token-batch-size=256 \
  --max-context-len=2048 \
  --listen=127.0.0.1:0 \
  --bench-prompt='Write a detailed technical essay about GPU compiler autotuning. Continue until the response is forcibly stopped by the token limit.' \
  --bench-max-tokens=256
```

The benchmark performs one unreported warmup request and one measured request.
Its TTFT is therefore a warm-prefix/cache metric; decode tok/s is the primary
runtime signal here.

### Compare retained dumps

```shell
python3 /tmp/gb300-xla-autotune.7FpgFs/analysis/compare_hlo_dumps.py \
  /tmp/gb300-xla-autotune.7FpgFs/runs/HLO-opt1-fresh.FEhh3J \
  /tmp/gb300-xla-autotune.7FpgFs/runs/HLO-opt2-adaptive2-full.kalECV

python3 /tmp/gb300-xla-autotune.7FpgFs/analysis/compare_autotune_textproto.py \
  /tmp/gb300-xla-autotune.7FpgFs/runs/HLO-opt1-fresh.FEhh3J/autotune-logs.textproto \
  /tmp/gb300-xla-autotune.7FpgFs/runs/HLO-opt2-adaptive2-full.kalECV/autotune-logs.textproto
```

The scripts are retained in `/tmp`, not committed to this branch. Their JSON
outputs named earlier are the source of the counts in this report.

## Limitations and unresolved work

1. The policy improves selection stability but does not guarantee identical
   fresh caches. The first sample determines the finalist set; a genuinely
   competitive candidate measured just outside the band is not reconsidered.
2. The `max_finalists=2` bound preserves compile time but can omit a third
   statistically indistinguishable candidate.
3. The 64 ns and 1% thresholds are calibrated from these GB300 traces, not a
   cross-architecture study. They should become an experimental flag before a
   general default is proposed.
4. There is no incumbent/cache-refresh comparison in this patch. Ordinary
   valid cache hits should continue to avoid retuning.
5. The final five-pair `llmd` validation is incomplete. It must be rerun before
   claiming the requested no-loss throughput property.
6. The hybrid cache entries were not replicated enough to identify a minimal
   causal kernel set. Runtime GPU profiling of confirmed culprits was therefore
   not justified.
7. Some earlier parallel dump attempts produced a damaged combined autotune
   log because multiple per-module `Autotuner` instances append to one path
   without a process-global file lock. That is a diagnostic-output integrity
   issue, not the cause of kernel-selection churn. The final full dump parsed
   successfully.
8. External processes, device clocks, and thermal state remain outside the
   in-process candidate transaction lock.

## Recommendation

Keep both symbolic-tiling compilation optimizations: the HLO evidence shows no
semantic divergence before selection, and the final replay retains a 30.1%
incremental batch-time reduction versus the first-optimization reference.

Keep the candidate-level transaction as the correct fix for warmup/measurement
interleaving, but do not describe it as global determinism. Keep bounded,
round-robin median remeasurement as an opt-in experiment until the interrupted
`llmd` A/B test is completed and thresholds are validated beyond this GB300.
The rollback should independently disable adaptive remeasurement while leaving
the compile optimizations and transaction lock intact.

The next policy experiment should not merely increase `max_finalists` for every
group: the unbounded run already shows that this can erase the compile benefit.
First retain every raw sample and transaction-order field in the dump. Then
separate two cases under a fixed total profiling budget:

* For sub-10-us kernels whose candidates remain inside the calibrated noise
  interval, stop trying to infer a strict ordering from indistinguishable event
  measurements and apply a deterministic canonical tie rule.
* For high-impact dot/GEMM groups, allow a slightly wider initial reservoir and
  use round-robin successive elimination. The two four-choice dot keys show
  that a fixed top-two admission decision is too sensitive to the first sample.

Any such policy still needs cache-held runtime validation. A stable choice is
not necessarily the fastest choice in the full model, especially for a small
kernel called many times.

For deployment, the strongest immediate protection is still to use a complete,
validated per-fusion autotune cache generated under controlled conditions. The
goal of adaptive fresh tuning is to make creation of that cache less sensitive
to near-tie noise; it does not make a fresh online decision equivalent to a
trusted cache.
