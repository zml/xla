set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
export USE_BAZEL_VERSION=8.5.1

TARGET="//xla/pjrt/c:pjrt_c_api_gpu_plugin"
ARGS=(--config=hermetic_macos_arm64 --config=remote --config=bzlmod
      --host_cpu=aarch64)

echo ">> fetching repos (analysis only) ..."
bazel build --announce_rc --nobuild "${ARGS[@]}" "$TARGET"


SDK="$(bazel info output_base)/external/llvm++osx+macos_sdk/sysroot"
AS="$SDK/System/Library/Frameworks/ApplicationServices.framework/Versions/A/Frameworks"
echo ">> stripping dangling macOS-SDK symlinks ..."
rm -f "$AS/CoreText.framework" \
      "$AS/ImageIO.framework" \
      "$AS/ColorSync.framework" \
      "$AS/ATS.framework/Versions/A/Resources/libFontParser.tbd" \
      "$AS/ATS.framework/Versions/A/Resources/libType1Scaler.tbd"

echo ">> remote build ..."
bazel build "${ARGS[@]}" "$@" "$TARGET" 
echo ">> done -> $(pwd)/bazel-bin/xla/pjrt/c/libpjrt_c_api_gpu_plugin.dylib"
