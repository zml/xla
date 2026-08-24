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
autotuner="${runfiles_root}/${workspace_name}/xla/backends/gpu/autotuner/autotuner_main"
benchmark_modes=0
for mode in FLY_GEMM_BENCHMARK_SCALED FLY_GEMM_BENCHMARK_SCALED_FP8 \
            FLY_GEMM_BENCHMARK_BLOCK_SCALED_FP8 \
            FLY_GEMM_BENCHMARK_NARROWING \
            FLY_GEMM_BENCHMARK_F32_OUTPUT \
            FLY_GEMM_BENCHMARK_F32 \
            FLY_GEMM_BENCHMARK_F16 \
            FLY_GEMM_BENCHMARK_F16_BATCHED \
            FLY_GEMM_BENCHMARK_F16_BATCHED_EPILOGUE \
            FLY_GEMM_BENCHMARK_FP8 \
            FLY_GEMM_BENCHMARK_BATCHED_FP8 \
            FLY_GEMM_BENCHMARK_BATCHED_SCALED_FP8 \
            FLY_GEMM_BENCHMARK_GLOBAL_SPLIT_K \
            FLY_GEMM_BENCHMARK_INT4 \
            FLY_GEMM_BENCHMARK_SCALAR FLY_GEMM_BENCHMARK_VECTOR \
            FLY_GEMM_BENCHMARK_CHAIN FLY_GEMM_BENCHMARK_CONVERTED_INPUTS \
            FLY_GEMM_BENCHMARK_BITCAST_INPUTS \
            FLY_GEMM_BENCHMARK_SLICE_INPUTS \
            FLY_GEMM_BENCHMARK_DYNAMIC_SLICE_INPUTS \
            FLY_GEMM_BENCHMARK_CONCAT_INPUTS \
            FLY_GEMM_BENCHMARK_RELU FLY_GEMM_BENCHMARK_BATCHED; do
  if [[ "${!mode:-0}" == "1" ]]; then
    benchmark_modes=$((benchmark_modes + 1))
  fi
done
if [[ "${benchmark_modes}" -gt 1 ]]; then
  echo "GEMM benchmark modes are mutually exclusive." >&2
  exit 1
elif [[ "${FLY_GEMM_BENCHMARK_SCALED:-0}" == "1" ]]; then
  template="${runfiles_root}/${workspace_name}/xla/backends/gpu/codegen/flydsl/scaled_gemm_benchmark.hlo.tpl"
  benchmark_name="bf16_scaled_gemm"
elif [[ "${FLY_GEMM_BENCHMARK_SCALED_FP8:-0}" == "1" ]]; then
  template="${runfiles_root}/${workspace_name}/xla/backends/gpu/codegen/flydsl/scaled_fp8_gemm_benchmark.hlo.tpl"
  benchmark_name="fnuz_fp8_scaled_gemm"
elif [[ "${FLY_GEMM_BENCHMARK_BLOCK_SCALED_FP8:-0}" == "1" ]]; then
  template="${runfiles_root}/${workspace_name}/xla/backends/gpu/codegen/flydsl/block_scaled_fp8_gemm_benchmark.hlo.tpl"
  benchmark_name="fnuz_fp8_block_scaled_gemm"
elif [[ "${FLY_GEMM_BENCHMARK_NARROWING:-0}" == "1" ]]; then
  template="${runfiles_root}/${workspace_name}/xla/backends/gpu/codegen/flydsl/narrowing_gemm_benchmark.hlo.tpl"
  benchmark_name="f32_dot_bf16_epilogue"
elif [[ "${FLY_GEMM_BENCHMARK_F32_OUTPUT:-0}" == "1" ]]; then
  template="${runfiles_root}/${workspace_name}/xla/backends/gpu/codegen/flydsl/f32_output_gemm_benchmark.hlo.tpl"
  benchmark_name="bf16_f32_output_gemm"
elif [[ "${FLY_GEMM_BENCHMARK_F32:-0}" == "1" ]]; then
  template="${runfiles_root}/${workspace_name}/xla/backends/gpu/codegen/flydsl/f32_gemm_benchmark.hlo.tpl"
  benchmark_name="f32_gemm"
elif [[ "${FLY_GEMM_BENCHMARK_F16:-0}" == "1" ]]; then
  template="${runfiles_root}/${workspace_name}/xla/backends/gpu/codegen/flydsl/f16_gemm_benchmark.hlo.tpl"
  benchmark_name="f16_gemm"
elif [[ "${FLY_GEMM_BENCHMARK_F16_BATCHED:-0}" == "1" ]]; then
  template="${runfiles_root}/${workspace_name}/xla/backends/gpu/codegen/flydsl/f16_batched_gemm_benchmark.hlo.tpl"
  benchmark_name="f16_batched_gemm"
elif [[ "${FLY_GEMM_BENCHMARK_F16_BATCHED_EPILOGUE:-0}" == "1" ]]; then
  template="${runfiles_root}/${workspace_name}/xla/backends/gpu/codegen/flydsl/f16_batched_epilogue_gemm_benchmark.hlo.tpl"
  benchmark_name="f16_batched_epilogue_gemm"
elif [[ "${FLY_GEMM_BENCHMARK_BATCHED_FP8:-0}" == "1" ]]; then
  template="${runfiles_root}/${workspace_name}/xla/backends/gpu/codegen/flydsl/batched_fp8_gemm_benchmark.hlo.tpl"
  benchmark_name="fnuz_fp8_batched_gemm"
elif [[ "${FLY_GEMM_BENCHMARK_BATCHED_SCALED_FP8:-0}" == "1" ]]; then
  template="${runfiles_root}/${workspace_name}/xla/backends/gpu/codegen/flydsl/batched_scaled_fp8_gemm_benchmark.hlo.tpl"
  benchmark_name="fnuz_fp8_batched_scaled_gemm"
elif [[ "${FLY_GEMM_BENCHMARK_GLOBAL_SPLIT_K:-0}" == "1" ]]; then
  template="${runfiles_root}/${workspace_name}/xla/backends/gpu/codegen/flydsl/global_split_k_gemm_benchmark.hlo.tpl"
  benchmark_name="bf16_global_split_k_gemm"
elif [[ "${FLY_GEMM_BENCHMARK_FP8:-0}" == "1" ]]; then
  template="${runfiles_root}/${workspace_name}/xla/backends/gpu/codegen/flydsl/fp8_gemm_benchmark.hlo.tpl"
  benchmark_name="fnuz_fp8_gemm"
elif [[ "${FLY_GEMM_BENCHMARK_INT4:-0}" == "1" ]]; then
  template="${runfiles_root}/${workspace_name}/xla/backends/gpu/codegen/flydsl/int4_gemm_benchmark.hlo.tpl"
  benchmark_name="s4_dequantized_gemm"
elif [[ "${FLY_GEMM_BENCHMARK_SCALAR:-0}" == "1" ]]; then
  template="${runfiles_root}/${workspace_name}/xla/backends/gpu/codegen/flydsl/scalar_gemm_benchmark.hlo.tpl"
  benchmark_name="f32_dot_scalar_epilogue"
elif [[ "${FLY_GEMM_BENCHMARK_VECTOR:-0}" == "1" ]]; then
  template="${runfiles_root}/${workspace_name}/xla/backends/gpu/codegen/flydsl/vector_gemm_benchmark.hlo.tpl"
  benchmark_name="f32_dot_column_bias_epilogue"
elif [[ "${FLY_GEMM_BENCHMARK_CHAIN:-0}" == "1" ]]; then
  template="${runfiles_root}/${workspace_name}/xla/backends/gpu/codegen/flydsl/chain_gemm_benchmark.hlo.tpl"
  benchmark_name="f32_dot_epilogue_chain"
elif [[ "${FLY_GEMM_BENCHMARK_CONVERTED_INPUTS:-0}" == "1" ]]; then
  template="${runfiles_root}/${workspace_name}/xla/backends/gpu/codegen/flydsl/converted_input_gemm_benchmark.hlo.tpl"
  benchmark_name="f32_inputs_bf16_gemm"
elif [[ "${FLY_GEMM_BENCHMARK_BITCAST_INPUTS:-0}" == "1" ]]; then
  template="${runfiles_root}/${workspace_name}/xla/backends/gpu/codegen/flydsl/bitcast_input_gemm_benchmark.hlo.tpl"
  benchmark_name="f32_bitcast_inputs_bf16_gemm"
elif [[ "${FLY_GEMM_BENCHMARK_SLICE_INPUTS:-0}" == "1" ]]; then
  template="${runfiles_root}/${workspace_name}/xla/backends/gpu/codegen/flydsl/slice_input_gemm_benchmark.hlo.tpl"
  benchmark_name="f32_slice_inputs_bf16_gemm"
elif [[ "${FLY_GEMM_BENCHMARK_DYNAMIC_SLICE_INPUTS:-0}" == "1" ]]; then
  template="${runfiles_root}/${workspace_name}/xla/backends/gpu/codegen/flydsl/dynamic_slice_input_gemm_benchmark.hlo.tpl"
  benchmark_name="bf16_dynamic_slice_inputs_gemm"
elif [[ "${FLY_GEMM_BENCHMARK_CONCAT_INPUTS:-0}" == "1" ]]; then
  template="${runfiles_root}/${workspace_name}/xla/backends/gpu/codegen/flydsl/concat_input_gemm_benchmark.hlo.tpl"
  benchmark_name="f32_concat_inputs_bf16_gemm"
elif [[ "${FLY_GEMM_BENCHMARK_RELU:-0}" == "1" ]]; then
  template="${runfiles_root}/${workspace_name}/xla/backends/gpu/codegen/flydsl/relu_gemm_benchmark.hlo.tpl"
  benchmark_name="f32_dot_bias_relu_epilogue"
elif [[ "${FLY_GEMM_BENCHMARK_BATCHED:-0}" == "1" ]]; then
  template="${runfiles_root}/${workspace_name}/xla/backends/gpu/codegen/flydsl/batched_gemm_benchmark.hlo.tpl"
  benchmark_name="bf16_batched_gemm"
else
  template="${runfiles_root}/${workspace_name}/xla/backends/gpu/codegen/flydsl/gemm_benchmark.hlo.tpl"
  benchmark_name="bf16_gemm"
fi
rocm_lib="${runfiles_root}/local_config_rocm/rocm/rocm_dist/lib"

if [[ ! -x "${autotuner}" || ! -f "${template}" || ! -d "${rocm_lib}" ]]; then
  echo "Could not locate benchmark runfiles." >&2
  exit 1
fi

export LD_LIBRARY_PATH="${rocm_lib}:${rocm_lib}/rocm_sysdeps/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

# TheRock multi-architecture redists keep each architecture's hipBLASLt
# database and code objects in a subdirectory. Bazel packages only the
# configured architectures, so select it automatically when it is unique.
if [[ -z "${HIPBLASLT_TENSILE_LIBPATH:-}" ]]; then
  hipblaslt_arch_dirs=("${rocm_lib}"/hipblaslt/library/gfx*)
  if [[ "${#hipblaslt_arch_dirs[@]}" -eq 1 &&
        -d "${hipblaslt_arch_dirs[0]}" ]]; then
    export HIPBLASLT_TENSILE_LIBPATH="${hipblaslt_arch_dirs[0]}"
  fi
fi

output_dir="${FLY_GEMM_BENCHMARK_OUTPUT_DIR:-/tmp/fly_gemm_benchmark}"
mkdir -p "${output_dir}"
autotune_backends="${FLY_GEMM_BENCHMARK_BACKENDS:-triton,fly,fly_fission,hipblaslt_fission}"

rhs_layout="${FLY_GEMM_BENCHMARK_RHS_LAYOUT:-1,0}"
if [[ "${rhs_layout}" != "1,0" && "${rhs_layout}" != "0,1" ]]; then
  echo "Invalid FLY_GEMM_BENCHMARK_RHS_LAYOUT='${rhs_layout}'; expected 1,0 or 0,1." >&2
  exit 1
fi

if [[ "$#" -eq 0 ]]; then
  if [[ "${FLY_GEMM_BENCHMARK_BATCHED:-0}" == "1" ||
        "${FLY_GEMM_BENCHMARK_F16_BATCHED:-0}" == "1" ||
        "${FLY_GEMM_BENCHMARK_F16_BATCHED_EPILOGUE:-0}" == "1" ||
        "${FLY_GEMM_BENCHMARK_BATCHED_FP8:-0}" == "1" ||
        "${FLY_GEMM_BENCHMARK_BATCHED_SCALED_FP8:-0}" == "1" ||
        "${FLY_GEMM_BENCHMARK_GLOBAL_SPLIT_K:-0}" == "1" ]]; then
    if [[ "${FLY_GEMM_BENCHMARK_GLOBAL_SPLIT_K:-0}" == "1" ]]; then
      set -- 2x256x1024x512 4x256x1024x1024
    else
      set -- 8x256x256x256 8x512x512x512 8x1024x1024x1024
    fi
  else
    set -- 256x256x256 512x512x512 1024x1024x1024
  fi
fi

for shape in "$@"; do
  batch=1
  if [[ "${FLY_GEMM_BENCHMARK_BATCHED:-0}" == "1" ||
        "${FLY_GEMM_BENCHMARK_F16_BATCHED:-0}" == "1" ||
        "${FLY_GEMM_BENCHMARK_F16_BATCHED_EPILOGUE:-0}" == "1" ||
        "${FLY_GEMM_BENCHMARK_BATCHED_FP8:-0}" == "1" ||
        "${FLY_GEMM_BENCHMARK_BATCHED_SCALED_FP8:-0}" == "1" ||
        "${FLY_GEMM_BENCHMARK_GLOBAL_SPLIT_K:-0}" == "1" ]]; then
    IFS=x read -r batch m n k extra <<<"${shape}"
    expected_shape="BxMxNxK"
  else
    IFS=x read -r m n k extra <<<"${shape}"
    expected_shape="MxNxK"
  fi
  if [[ -n "${extra:-}" || -z "${batch}" || -z "${m}" || -z "${n}" ||
        -z "${k}" || ! "${batch}" =~ ^[0-9]+$ ||
        ! "${m}" =~ ^[0-9]+$ || ! "${n}" =~ ^[0-9]+$ ||
        ! "${k}" =~ ^[0-9]+$ ]]; then
    echo "Invalid shape '${shape}'; expected ${expected_shape}." >&2
    exit 1
  fi
  if [[ "${FLY_GEMM_BENCHMARK_BLOCK_SCALED_FP8:-0}" == "1" ]] &&
      ((k % 32 != 0)); then
    echo "Block-scaled FP8 benchmark requires K divisible by 32; got '${shape}'." >&2
    exit 1
  fi

  hlo="${output_dir}/${benchmark_name}_${shape}.hlo"
  selected="${output_dir}/${benchmark_name}_${shape}.autotuned.hlo"
  log="${output_dir}/${benchmark_name}_${shape}.autotune.textproto"

  padded_m=$((m + 32))
  padded_n=$((n + 32))
  padded_k=$((k + 64))
  end_m=$((m + 16))
  end_n=$((n + 16))
  end_k=$((k + 32))
  half_m=$((m / 2))
  half_n=$((n / 2))
  scale_k=$((k / 32))
  if [[ "${rhs_layout}" == "1,0" ]]; then
    batch_rhs_layout="2,1,0"
  else
    batch_rhs_layout="1,2,0"
  fi
  if [[ "${FLY_GEMM_BENCHMARK_CONCAT_INPUTS:-0}" == "1" ]] &&
      ((m % 2 != 0 || n % 2 != 0)); then
    echo "Concat-input benchmark requires even M and N; got '${shape}'." >&2
    exit 1
  fi
  sed -e "s/__B__/${batch}/g" -e "s/__M__/${m}/g" \
    -e "s/__N__/${n}/g" -e "s/__K__/${k}/g" \
    -e "s/__KB__/${scale_k}/g" \
    -e "s/__MH__/${half_m}/g" -e "s/__NH__/${half_n}/g" \
    -e "s/__MP__/${padded_m}/g" -e "s/__NP__/${padded_n}/g" \
    -e "s/__KP__/${padded_k}/g" -e "s/__MEND__/${end_m}/g" \
    -e "s/__NEND__/${end_n}/g" -e "s/__KEND__/${end_k}/g" \
    -e "s/__RHS_LAYOUT__/${rhs_layout}/g" \
    -e "s/__BATCH_RHS_LAYOUT__/${batch_rhs_layout}/g" \
    "${template}" >"${hlo}"
  rm -f "${log}"

  XLA_FLAGS="${XLA_FLAGS:-} \
--xla_gpu_autotune_level=${FLY_GEMM_BENCHMARK_AUTOTUNE_LEVEL:-3} \
--xla_gpu_autotune_num_repetitions=${FLY_GEMM_BENCHMARK_REPETITIONS:-5} \
--xla_gpu_enable_flydsl_gemm=true \
--xla_gpu_experimental_autotune_backends=${autotune_backends} \
--xla_gpu_dump_autotune_logs_to=${log}" \
    HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}" \
    HIPBLASLT_PRELOAD_KERNELS="${HIPBLASLT_PRELOAD_KERNELS:-0}" \
    HIPBLASLT_ROCROLLER_NO_CUSTOM_KERNEL="${HIPBLASLT_ROCROLLER_NO_CUSTOM_KERNEL:-1}" \
    "${autotuner}" --hlo_files="${hlo}" >"${selected}"

  echo "${shape}:"
  grep -m2 -E '  Backend:|  Config:' "${selected}" | sed 's/^/  selected /'
  awk '
    function record_result() {
      if (backend != "" && nanos != "" &&
          (!(backend in best) || nanos < best[backend])) {
        best[backend] = nanos
      }
    }
    /^  results \{/ {
      record_result()
      backend = ""
      nanos = ""
      next
    }
    /nanos:/ { nanos = $2 }
    /^[[:space:]]+triton \{/ { backend = "TRITON" }
    /^[[:space:]]+gemm \{/ { backend = "HIPBLASLT" }
    /name: "FLY"/ { backend = "FLY" }
    /name: "FLY_FISSION"/ { backend = "FLY_FISSION" }
    END {
      record_result()
      for (i = 1; i <= 4; ++i) {
        name = (i == 1 ? "FLY" :
                (i == 2 ? "FLY_FISSION" :
                 (i == 3 ? "TRITON" : "HIPBLASLT")))
        if (name in best) {
          printf "  best %-10s %10.3f us\n", name, best[name] / 1000.0
        }
      }
      if ("FLY" in best && "TRITON" in best) {
        printf "  Fly / Triton speedup: %.2fx\n", best["TRITON"] / best["FLY"]
      }
    }
  ' "${log}"
  echo "  selected config: ${selected}"
  echo "  candidate timings: ${log}"
done
