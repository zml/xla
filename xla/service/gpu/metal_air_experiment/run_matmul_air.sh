#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOOLCHAIN_DEFAULT="/Users/steeve/Library/Developer/DVTDownloads/MetalToolchain/mounts/4ab058bc1c53034b8c0a9baca6fba2d2b78bb965/Metal.xctoolchain/usr/bin"
TOOLCHAIN="${METAL_TOOLCHAIN:-$TOOLCHAIN_DEFAULT}"
OUT="${OUT:-/private/tmp/xla_metal_air_matmul}"

METAL="${METAL:-$TOOLCHAIN/metal}"
AIR_AS="${AIR_AS:-$TOOLCHAIN/air-as}"
AIR_OBJDUMP="${AIR_OBJDUMP:-$TOOLCHAIN/air-objdump}"
METALLIB="${METALLIB:-$TOOLCHAIN/metallib}"
CXX="${CXX:-clang++}"

mkdir -p "$OUT"
export CLANG_MODULE_CACHE_PATH="$OUT/module-cache"

"$METAL" -std=metal3.2 -O3 -ffast-math \
  -fmodules-cache-path="$OUT/module-cache" -S -emit-llvm -c \
  "$ROOT/matmul_air.metal" -o "$OUT/matmul_air.ll"
"$AIR_AS" "$OUT/matmul_air.ll" -o "$OUT/matmul_air.air"
"$METALLIB" "$OUT/matmul_air.air" -o "$OUT/matmul_air.metallib"
"$AIR_AS" "$ROOT/matmul_air_direct.ll" -o "$OUT/matmul_air_direct.air"
"$TOOLCHAIN/air-opt" --O3 "$OUT/matmul_air_direct.air" \
  -o "$OUT/matmul_air_direct.opt.air"
"$METALLIB" "$OUT/matmul_air_direct.opt.air" \
  -o "$OUT/matmul_air_direct.metallib"
"$CXX" -std=c++17 -O3 -fobjc-arc "$ROOT/matmul_air_runner.mm" \
  -framework Foundation -framework Metal -o "$OUT/matmul_air_runner"

"$AIR_OBJDUMP" --metallib --air-version "$OUT/matmul_air.metallib" || true
"$AIR_OBJDUMP" --metallib --air-version "$OUT/matmul_air_direct.metallib" || true

"$OUT/matmul_air_runner" "$OUT/matmul_air.metallib" matmul_tiled16 \
  512 512 512 20 3
"$OUT/matmul_air_runner" "$OUT/matmul_air.metallib" matmul_simdgroup_8x8 \
  2048 2048 2048 50 5
"$OUT/matmul_air_runner" "$OUT/matmul_air_direct.metallib" matmul_simdgroup_8x8 \
  2048 2048 2048 50 5
