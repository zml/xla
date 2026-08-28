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
direct_template="${benchmark_root}/paged_attention_preconfigured_benchmark.hlo.tpl"
segmented_template="${benchmark_root}/paged_attention_segmented_preconfigured_benchmark.hlo.tpl"
segmented_fused_template="${benchmark_root}/paged_attention_segmented_fused_preconfigured_benchmark.hlo.tpl"
rocm_lib="${runfiles_root}/local_config_rocm/rocm/rocm_dist/lib"

if [[ ! -x "${autotuner}" || ! -f "${direct_template}" ||
      ! -f "${segmented_template}" || ! -f "${segmented_fused_template}" ||
      ! -d "${rocm_lib}" ]]; then
  echo "Could not locate paged-attention benchmark runfiles." >&2
  exit 1
fi

export LD_LIBRARY_PATH="${rocm_lib}:${rocm_lib}/rocm_sysdeps/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

if [[ "$#" -eq 0 ]]; then
  set -- 1x32x8x128x16 64x16x4x128x16
fi

dtype="${FLY_PAGED_ATTENTION_BENCHMARK_DTYPE:-bf16}"
if [[ "${dtype}" != "bf16" && "${dtype}" != "f16" ]]; then
  echo "FLY_PAGED_ATTENTION_BENCHMARK_DTYPE must be bf16 or f16." >&2
  exit 1
fi

output_dir="${FLY_PAGED_ATTENTION_BENCHMARK_OUTPUT_DIR:-/tmp/fly_paged_attention_benchmark}"
mkdir -p "${output_dir}"

for shape in "$@"; do
  IFS=x read -r batch query_heads kv_heads max_context page_size extra <<<"${shape}"
  if [[ -n "${extra:-}" || -z "${batch}" || -z "${query_heads}" ||
        -z "${kv_heads}" || -z "${max_context}" || -z "${page_size}" ||
        ! "${batch}" =~ ^[0-9]+$ || ! "${query_heads}" =~ ^[0-9]+$ ||
        ! "${kv_heads}" =~ ^[0-9]+$ || ! "${max_context}" =~ ^[0-9]+$ ||
        ! "${page_size}" =~ ^[0-9]+$ ]]; then
    echo "Invalid shape '${shape}'; expected BxHqxHkvxKVxPAGE." >&2
    exit 1
  fi
  if ((batch < 1 || kv_heads < 1 || query_heads % kv_heads != 0 ||
       query_heads / kv_heads > 16)); then
    echo "Shape '${shape}' requires a positive GQA group in [1,16]." >&2
    exit 1
  fi
  if ((max_context < 1 || max_context > 262144 ||
       max_context % page_size != 0)) ||
      [[ "${page_size}" != "16" && "${page_size}" != "32" &&
         "${page_size}" != "64" ]]; then
    echo "Shape '${shape}' requires KV<=262144 divisible by PAGE in {16,32,64}." >&2
    exit 1
  fi

  pages=$((max_context / page_size))
  blocks=$((batch * pages))
  used_k_values="${max_context}"
  for ((sequence = 1; sequence < batch; ++sequence)); do
    used_k_values+=",${max_context}"
  done
  segment_tokens=${FLY_PAGED_ATTENTION_SEGMENT_TOKENS:-}
  if [[ -n "${segment_tokens}" ]]; then
    if [[ ! "${segment_tokens}" =~ ^[0-9]+$ ]] ||
       ((segment_tokens < 16 || segment_tokens % 16 != 0)); then
      echo "FLY_PAGED_ATTENTION_SEGMENT_TOKENS must be a positive multiple of 16." >&2
      exit 1
    fi
  else
    segment_tokens=128
    if ((max_context > 131072)); then
      segment_tokens=2304
    elif ((max_context == 131072)); then
      segment_tokens=1152
    elif ((max_context >= 65536)); then
      segment_tokens=576
    elif ((max_context > 8192)); then
      segment_tokens=512
    fi
  fi
  segments=$(((max_context + segment_tokens - 1) / segment_tokens))
  if ((max_context <= 256)); then
    segments=1
  fi
  if ((segments > 256)); then
    echo "Shape '${shape}' with segment size ${segment_tokens} produces ${segments} segments; the segmented reducer supports at most 256." >&2
    exit 1
  fi
  template=${direct_template}
  autotune_backends=fly_fusion,triton
  if ((segments > 1)); then
    template=${segmented_template}
  fi
  # Exercise the public-call rewrite for the long-context GQA4 shapes where
  # Fly combines the segmented producer and reducer into one device kernel.
  if [[ "${FLY_PAGED_ATTENTION_FORCE_SEPARATE:-0}" != "1" ]] &&
     ((max_context >= 65536 && query_heads / kv_heads == 4 &&
       segments <= 128 &&
       (max_context >= 131072 || batch * kv_heads <= 2))); then
    template=${segmented_fused_template}
    # Triton's compiler cannot implement the inter-CTA completion protocol in
    # this custom tuple ABI. The separate producer/reducer path remains the
    # like-for-like Triton baseline selected with FORCE_SEPARATE below.
    autotune_backends=fly_fusion
  fi
  hlo="${output_dir}/paged_attention_${dtype}_${shape}.hlo"
  selected="${output_dir}/paged_attention_${dtype}_${shape}.autotuned.hlo"
  log="${output_dir}/paged_attention_${dtype}_${shape}.autotune.textproto"

  sed -e "s/__B__/${batch}/g" -e "s/__HQ__/${query_heads}/g" \
    -e "s/__HKV__/${kv_heads}/g" -e "s/__BLOCKS__/${blocks}/g" \
    -e "s/__PAGE__/${page_size}/g" -e "s/__PAGES__/${pages}/g" \
    -e "s/__SEGMENTS__/${segments}/g" \
    -e "s/__USED_K_VALUES__/${used_k_values}/g" \
    -e "s/__TYPE__/${dtype}/g" \
    "${template}" >"${hlo}"
  rm -f "${log}"

  XLA_FLAGS="${XLA_FLAGS:-} \
--xla_gpu_autotune_level=3 \
--xla_gpu_autotune_num_repetitions=${FLY_PAGED_ATTENTION_BENCHMARK_REPETITIONS:-100} \
--xla_gpu_enable_flydsl_fusion=true \
--xla_gpu_experimental_enable_fusion_autotuner=true \
--xla_gpu_experimental_autotune_backends=${autotune_backends} \
--xla_gpu_force_compilation_parallelism=1 \
--xla_gpu_dump_autotune_logs_to=${log}" \
    HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}" \
    "${autotuner}" --hlo_files="${hlo}" >"${selected}"

  echo "${dtype} ${shape}:"
  grep -E '  Backend:|  Config:' "${selected}" | sed 's/^/  selected /'
  awk '
    BEGIN {
      split("0 1 2 4", fly_occupancy)
      triton_config[1] = "stages=2 waves_per_eu=2"
      triton_config[2] = "stages=1 waves_per_eu=2"
      triton_config[3] = "stages=3 waves_per_eu=2"
      triton_config[4] = "stages=2 waves_per_eu=0"
      triton_config[5] = "stages=2 waves_per_eu=1"
      triton_config[6] = "stages=2 waves_per_eu=4"
      kernel = 0
    }
    /^logs \{/ {
      kernel++
      delete backend_candidate
    }
    /nanos:/ { nanos = $2 }
    /name:/ {
      backend = $2
      gsub(/"/, "", backend)
      backend_candidate[backend]++
      candidate = backend_candidate[backend]
      value = nanos / 1000.0
      if (backend == "TRITON") {
        config = triton_config[candidate]
      } else if (backend == "FLY_FUSION") {
        stage = int((candidate - 1) / 4) + 1
        occupancy = ((candidate - 1) % 4) + 1
        config = "stages=" stage " waves_per_eu=" fly_occupancy[occupancy]
      } else {
        config = "candidate=" candidate
      }
      printf "  kernel %d %-10s %-25s %10.3f us\n", \
             kernel, backend, config, value
      if (minimum[kernel] == 0 || value < minimum[kernel]) minimum[kernel] = value
      key = kernel SUBSEP backend
      if (backend_minimum[key] == 0 || value < backend_minimum[key]) {
        backend_minimum[key] = value
      }
    }
    END {
      total = 0
      for (i = 1; i <= kernel; ++i) {
        total += minimum[i]
        for (key in backend_minimum) {
          split(key, fields, SUBSEP)
          if (fields[1] == i) {
            printf "  kernel %d best %-10s %25s %10.3f us\n", \
                   i, fields[2], "", backend_minimum[key]
          }
        }
      }
      printf "  selected kernel-time sum             %10.3f us\n", total
    }
  ' "${log}"
done
