// XLA FFI handler for "cutlass$fp4_mm_dyn": NVFP4 W4A4 GEMM on sm120a.
// Operands (from the FusedScaledDotRewriter cutlass arm):
//   x      bf16[.., K]     activation, quantized to fp4 inside the kernel
//   b      f4e2m1[N, K]    fp4-packed weight (bytes read as-is)
//   b_sf   f8e4m3[szB]     weight block scale, cutlass swizzled SF layout
//   wgs    f32[1]          weight_global_scale, raw as stored
//   igs    f32[1]          input_global_scale
// Result:  d  bf16[.., N]
// The handler does not swizzle: the block scale reaches the kernel already in
// cutlass's SF layout, because the framework does that transform once at load
// time (ZML ops.swizzleNvfp4Scale), as plain HLO tied to no backend.
#include <cuda_runtime.h>

#include <cstdio>

#include "absl/status/status.h"
#include "xla/backends/gpu/ffi.h"
#include "xla/ffi/ffi.h"
#include "xla/stream_executor/stream.h"

using AllocFn = void* (*)(void* ctx, size_t size);
extern "C" void cutlass_fp4_mm_dyn_sm120a(void* D, const void* x, const void* B,
                                          const void* Bsf_sw, const float* wgs,
                                          const float* igs, int M, int N, int K,
                                          cudaStream_t stream, AllocFn alloc,
                                          void* actx);

namespace xla::gpu {
namespace ffi = ::xla::ffi;

// Bridges XLA's graph-compatible ScratchAllocator to the C alloc callback.
static void* ScratchAlloc(void* ctx, size_t size) {
  auto* s = static_cast<se::OwningScratchAllocator<>*>(ctx);
  auto mem = s->AllocateBytes(size);
  return mem.ok() ? mem->opaque() : nullptr;
}

// Dynamic path: handler quantizes the bf16 activation itself (vLLM-style),
// controlling the exact fp4 byte format cutlass consumes.
static absl::Status CutlassFp4MmDynImpl(
    se::Stream* stream, se::OwningScratchAllocator<> scratch,
    ffi::Buffer<PrimitiveType::BF16> x, ffi::Buffer<PrimitiveType::F4E2M1FN> b,
    ffi::Buffer<PrimitiveType::F8E4M3FN> b_sf,
    ffi::Buffer<PrimitiveType::F32> wgs, ffi::Buffer<PrimitiveType::F32> igs,
    ffi::Result<ffi::Buffer<PrimitiveType::BF16>> d) {
  // b is the NVFP4 weight in unpacked logical form f4e2m1[N,K]; its byte buffer
  // is the same 2-per-byte packing the kernel reads (K is taken from x below,
  // b.dims[1] is unused). The FusedScaledDotRewriter cutlass arm feeds the
  // scaled_dot rhs straight through, so no re-pack is needed.
  auto xd = x.dimensions();
  auto bd = b.dimensions();
  if (xd.size() < 2 || bd.size() != 2)
    return absl::InvalidArgumentError("cutlass$fp4_mm_dyn: x rank>=2, b rank-2");
  int M = 1;
  for (size_t i = 0; i + 1 < xd.size(); ++i) M *= static_cast<int>(xd[i]);
  const int K = static_cast<int>(xd[xd.size() - 1]);
  const int N = static_cast<int>(bd[0]);
  static int ndyn = 0;
  static int last_M = -1;
  ndyn++;
  if (M != last_M) {
    last_M = M;
    fprintf(stderr, "[dyn #%d] M CHANGED to %d (N=%d K=%d)\n", ndyn, M, N, K);
  }
  // b_sf must be the flat pre-swizzled weight scale [szB]. A rank-2 [N,kg] scale
  // is the natural layout, which nothing here converts — the rewriter is
  // responsible for only emitting this call when it can supply the swizzled form
  // (it declines otherwise, and the scaled_dot falls to Triton / the dequant floor).
  if (b_sf.dimensions().size() != 1)
    return absl::InvalidArgumentError(
        "cutlass$fp4_mm_dyn: b_sf must be the rank-1 swizzled weight scale");
  cudaStream_t cu = reinterpret_cast<cudaStream_t>(
      stream->platform_specific_handle().stream);
  cutlass_fp4_mm_dyn_sm120a(d->untyped_data(), x.untyped_data(),
                            b.untyped_data(), b_sf.untyped_data(),
                            reinterpret_cast<const float*>(wgs.untyped_data()),
                            reinterpret_cast<const float*>(igs.untyped_data()), M,
                            N, K, cu, &ScratchAlloc, &scratch);
  return absl::OkStatus();
}
XLA_FFI_DEFINE_HANDLER(kCutlassFp4MmDyn, CutlassFp4MmDynImpl,
                       ffi::Ffi::Bind()
                           .Ctx<ffi::Stream>()
                           .Ctx<ffi::ScratchAllocator>()
                           .Arg<ffi::Buffer<PrimitiveType::BF16>>()
                           .Arg<ffi::Buffer<PrimitiveType::F4E2M1FN>>()
                           .Arg<ffi::Buffer<PrimitiveType::F8E4M3FN>>()
                           .Arg<ffi::Buffer<PrimitiveType::F32>>()
                           .Arg<ffi::Buffer<PrimitiveType::F32>>()
                           .Ret<ffi::Buffer<PrimitiveType::BF16>>(),
                       {ffi::Traits::kCmdBufferCompatible});
XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(), "cutlass$fp4_mm_dyn", "CUDA",
                         kCutlassFp4MmDyn);

}  // namespace xla::gpu
