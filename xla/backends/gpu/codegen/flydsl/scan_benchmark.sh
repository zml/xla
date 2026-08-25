#!/usr/bin/env bash

set -euo pipefail

if [[ -n "${RUNFILES_DIR:-}" ]]; then
  runfiles_root="${RUNFILES_DIR}"
elif [[ -n "${RUNFILES_MANIFEST_FILE:-}" ]]; then
  runfiles_root="${RUNFILES_MANIFEST_FILE%.runfiles_manifest}.runfiles"
else
  runfiles_root="$0.runfiles"
fi

workspace_name="${TEST_WORKSPACE:-xla}"
workspace_root="${runfiles_root}/${workspace_name}"
runner="${workspace_root}/xla/tools/multihost_hlo_runner/hlo_runner_main_gpu"
benchmark_root="${workspace_root}/xla/backends/gpu/codegen/flydsl"
rocm_lib="${runfiles_root}/local_config_rocm/rocm/rocm_dist/lib"

if [[ ! -x "${runner}" || ! -d "${rocm_lib}" ]]; then
  echo "Could not locate benchmark runfiles." >&2
  exit 1
fi

export LD_LIBRARY_PATH="${rocm_lib}:${rocm_lib}/rocm_sysdeps/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

repeats="${FLY_SCAN_BENCHMARK_REPEATS:-31}"
warmups="${FLY_SCAN_BENCHMARK_WARMUPS:-10}"
output_dir="${FLY_SCAN_BENCHMARK_OUTPUT_DIR:-/tmp/fly_scan_benchmark}"
mkdir -p "${output_dir}"

run_case() {
  local name="$1"
  local enable_fly="$2"
  local hlo="$3"
  local log="${output_dir}/${name}.log"

  XLA_FLAGS="${XLA_FLAGS:-} \
--xla_gpu_enable_flydsl_fusion=${enable_fly} \
--xla_gpu_experimental_enable_fusion_autotuner=false \
--xla_gpu_autotune_level=0" \
    HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}" \
    "${runner}" \
      --device_type=gpu \
      --run=true \
      --num_replicas=1 \
      --num_partitions=1 \
      --num_repeats="${repeats}" \
      --profile_execution=true \
      --output_mode=not_return_outputs \
      --hlo_argument_mode=use_zeros_as_input \
      "${hlo}" >"${log}" 2>&1

  awk -v name="${name}" -v warmups="${warmups}" '
    /## Execution time/ {
      split($5, repeat_field, "=")
      split($6, duration_field, "=")
      repeat = repeat_field[2] + 0
      sub(/ns$/, "", duration_field[2])
      if (repeat >= warmups) {
        values[++count] = duration_field[2] + 0
        sum += values[count]
        if (count == 1 || values[count] < minimum) minimum = values[count]
      }
    }
    END {
      if (count == 0) exit 1
      for (i = 1; i <= count; ++i) {
        for (j = i + 1; j <= count; ++j) {
          if (values[j] < values[i]) {
            temporary = values[i]
            values[i] = values[j]
            values[j] = temporary
          }
        }
      }
      if (count % 2) {
        median = values[(count + 1) / 2]
      } else {
        median = (values[count / 2] + values[count / 2 + 1]) / 2
      }
      printf "%s median=%.3f us mean=%.3f us min=%.3f us samples=%d\n", \
             name, median / 1000.0, sum / count / 1000.0, \
             minimum / 1000.0, count
    }
  ' "${log}"
}

fly_result=$(run_case \
  fly \
  true \
  "${benchmark_root}/fly_fusion_bf16_scan_256_preconfigured_benchmark.hlo")
rocprim_result=$(run_case \
  rocprim \
  false \
  "${benchmark_root}/fly_fusion_bf16_scan_256_benchmark.hlo")

printf '%s\n%s\n' "${fly_result}" "${rocprim_result}"

fly_median=$(printf '%s\n' "${fly_result}" | sed -E 's/.*median=([0-9.]+) us.*/\1/')
rocprim_median=$(printf '%s\n' "${rocprim_result}" | sed -E 's/.*median=([0-9.]+) us.*/\1/')
awk -v fly="${fly_median}" -v rocprim="${rocprim_median}" \
  'BEGIN { printf "rocPRIM / Fly median speedup: %.2fx\n", rocprim / fly }'
