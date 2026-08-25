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
autotuner="${workspace_root}/xla/backends/gpu/autotuner/autotuner_main"
benchmark_root="${workspace_root}/xla/backends/gpu/codegen/flydsl"
rocm_lib="${runfiles_root}/local_config_rocm/rocm/rocm_dist/lib"

if [[ ! -x "${autotuner}" || ! -d "${rocm_lib}" ]]; then
  echo "Could not locate benchmark runfiles." >&2
  exit 1
fi

export LD_LIBRARY_PATH="${rocm_lib}:${rocm_lib}/rocm_sysdeps/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export HIPBLASLT_PRELOAD_KERNELS="${HIPBLASLT_PRELOAD_KERNELS:-0}"
export HIPBLASLT_ROCROLLER_NO_CUSTOM_KERNEL="${HIPBLASLT_ROCROLLER_NO_CUSTOM_KERNEL:-1}"

declare -A hlo_files=(
  [softmax]="fly_fusion_bf16_softmax_benchmark.hlo"
  [softmax_transformer]="fly_fusion_bf16_transformer_softmax_benchmark.hlo"
  [softmax_bf16_tail]="fly_fusion_bf16_softmax_tail_benchmark.hlo"
  [softmax_f16_tail]="fly_fusion_f16_softmax_tail_benchmark.hlo"
  [softmax_f32_tail]="fly_fusion_f32_softmax_tail_benchmark.hlo"
  [bf16_reduce]="fly_fusion_bf16_row_reduction_benchmark.hlo"
  [bf16_narrowing_reduce]="fly_fusion_bf16_narrowing_row_reduction_benchmark.hlo"
  [bf16_ragged_narrowing_reduce]="fly_fusion_bf16_ragged_narrowing_row_reduction_benchmark.hlo"
  [bf16_squared_difference_reduce]="fly_fusion_bf16_squared_difference_reduction_benchmark.hlo"
  [rms_norm_1024]="fly_fusion_bf16_rms_norm_1024_benchmark.hlo"
  [residual_rms_norm_1024]="fly_fusion_bf16_residual_rms_norm_1024_benchmark.hlo"
  [f32_reduce]="fly_fusion_row_reduction_benchmark.hlo"
  [elementwise]="fly_fusion_elementwise_hbm_benchmark.hlo"
  [elementwise_f16]="fly_fusion_f16_elementwise_hbm_benchmark.hlo"
  [elementwise_f32]="fly_fusion_f32_elementwise_hbm_benchmark.hlo"
  [ragged_elementwise]="fly_fusion_ragged_elementwise_hbm_benchmark.hlo"
  [multi_output_elementwise]="fly_fusion_multi_output_elementwise_hbm_benchmark.hlo"
  [sigmoid]="fly_fusion_bf16_sigmoid_benchmark.hlo"
  [transpose]="fly_fusion_bf16_transpose_benchmark.hlo"
  [transpose_1024]="fly_fusion_bf16_transpose_1024_benchmark.hlo"
  [transpose_4096]="fly_fusion_bf16_transpose_4096_benchmark.hlo"
  [transpose_4096x16384]="fly_fusion_bf16_transpose_4096x16384_benchmark.hlo"
  [transpose_transformer_qkv]="fly_fusion_bf16_transformer_qkv_transpose_benchmark.hlo"
  [transpose_transformer_qkv_multi_output]="fly_fusion_bf16_transformer_qkv_multi_output_transpose_benchmark.hlo"
  [transpose_transformer_context]="fly_fusion_bf16_transformer_context_transpose_benchmark.hlo"
)

if [[ "$#" -eq 0 ]]; then
  set -- softmax bf16_reduce f32_reduce elementwise transpose
fi

if [[ -n "${FLY_FUSION_BACKENDS:-}" ]]; then
  backends="${FLY_FUSION_BACKENDS}"
else
  backends="fly_fusion,block_level_emitter"
  if [[ "${FLY_FUSION_INCLUDE_NATIVE:-0}" == "1" ]]; then
    backends="${backends},native_emitter"
  fi
fi

output_dir="${FLY_FUSION_BENCHMARK_OUTPUT_DIR:-/tmp/fly_fusion_benchmark}"
mkdir -p "${output_dir}"

for benchmark in "$@"; do
  hlo_name="${hlo_files[${benchmark}]:-}"
  if [[ -z "${hlo_name}" ]]; then
    echo "Unknown benchmark '${benchmark}'." >&2
  echo "Available benchmarks: softmax softmax_transformer softmax_bf16_tail softmax_f16_tail softmax_f32_tail bf16_reduce bf16_narrowing_reduce bf16_ragged_narrowing_reduce bf16_squared_difference_reduce rms_norm_1024 residual_rms_norm_1024 f32_reduce elementwise elementwise_f16 elementwise_f32 ragged_elementwise multi_output_elementwise sigmoid transpose transpose_1024 transpose_4096 transpose_4096x16384 transpose_transformer_qkv transpose_transformer_qkv_multi_output transpose_transformer_context" >&2
    exit 1
  fi
  hlo="${benchmark_root}/${hlo_name}"
  selected="${output_dir}/${benchmark}.autotuned.hlo"
  log="${output_dir}/${benchmark}.autotune.textproto"
  if [[ ! -f "${hlo}" ]]; then
    echo "Missing benchmark HLO: ${hlo}" >&2
    exit 1
  fi
  rm -f "${log}"

  # Triton still gates tuple formation behind an unsupported flag; enable it
  # here so multi-output benchmarks have a real Triton candidate to compare
  # against. Fly formation itself does not depend on this flag.
  XLA_FLAGS="${XLA_FLAGS:-} \
--xla_gpu_autotune_level=3 \
--xla_gpu_autotune_num_repetitions=${FLY_FUSION_BENCHMARK_REPETITIONS:-5} \
--xla_gpu_enable_flydsl_fusion=true \
--xla_gpu_fusion_autotune_top_k_configs=${FLY_FUSION_AUTOTUNE_TOP_K:-8} \
--xla_gpu_experimental_all_fusions_with_triton=true \
--xla_gpu_unsupported_enable_triton_multi_output_fusion=true \
--xla_gpu_experimental_autotune_backends=${backends} \
--xla_gpu_dump_autotune_logs_to=${log}" \
    HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}" \
    "${autotuner}" --hlo_files="${hlo}" >"${selected}"

  echo "${benchmark}:"
  grep -m2 -E '  Backend:|  Config:' "${selected}" | sed 's/^/  selected /'
  awk '
    /nanos:/ { nanos = $2 }
    /name:/ {
      name = $2
      gsub(/"/, "", name)
      printf "  candidate %-22s %10.3f us\n", name, nanos / 1000.0
      if (!(name in best) || nanos < best[name]) best[name] = nanos
    }
    END {
      if ("FLY_FUSION" in best && "BLOCK_LEVEL_EMITTER" in best) {
        printf "  best Fly / Triton:       %10.3f / %.3f us (%.2fx)\n", \
               best["FLY_FUSION"] / 1000.0, \
               best["BLOCK_LEVEL_EMITTER"] / 1000.0, \
               best["BLOCK_LEVEL_EMITTER"] / best["FLY_FUSION"]
      }
    }
  ' "${log}"
done
