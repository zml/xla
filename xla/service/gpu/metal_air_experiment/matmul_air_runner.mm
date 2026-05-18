#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <string>
#include <vector>

namespace {

struct MatmulParams {
  uint32_t m;
  uint32_t n;
  uint32_t k;
  uint32_t reserved;
};

struct DispatchShape {
  MTLSize threadgroups;
  MTLSize threads_per_threadgroup;
};

uint32_t ParseU32(const char* value, const char* name) {
  char* end = nullptr;
  unsigned long parsed = std::strtoul(value, &end, 10);
  if (end == value || *end != '\0' ||
      parsed > std::numeric_limits<uint32_t>::max()) {
    std::fprintf(stderr, "invalid %s: %s\n", name, value);
    std::exit(2);
  }
  return static_cast<uint32_t>(parsed);
}

float InputA(size_t index) {
  return static_cast<float>((static_cast<int>(index * 13 % 257) - 128) *
                            0.00390625f);
}

float InputB(size_t index) {
  return static_cast<float>((static_cast<int>(index * 17 % 263) - 131) *
                            0.0037878788f);
}

void FillInputs(float* a, float* b, size_t a_count, size_t b_count) {
  for (size_t i = 0; i < a_count; ++i) a[i] = InputA(i);
  for (size_t i = 0; i < b_count; ++i) b[i] = InputB(i);
}

double ReferenceElement(const float* a, const float* b, uint32_t m, uint32_t n,
                        uint32_t k, uint32_t row, uint32_t col) {
  (void)m;
  double sum = 0.0;
  for (uint32_t kk = 0; kk < k; ++kk) {
    sum += static_cast<double>(a[row * k + kk]) *
           static_cast<double>(b[kk * n + col]);
  }
  return sum;
}

bool ValidateResult(const float* a, const float* b, const float* c, uint32_t m,
                    uint32_t n, uint32_t k) {
  const uint64_t full_check_ops = uint64_t{2} * m * n * k;
  std::vector<std::pair<uint32_t, uint32_t>> points;
  if (full_check_ops <= uint64_t{2} * 256 * 256 * 256) {
    points.reserve(static_cast<size_t>(m) * n);
    for (uint32_t row = 0; row < m; ++row) {
      for (uint32_t col = 0; col < n; ++col) points.push_back({row, col});
    }
  } else {
    points = {{0, 0},
              {0, n - 1},
              {m - 1, 0},
              {m - 1, n - 1},
              {m / 3, n / 5},
              {m / 2, n / 2},
              {(m * 7) / 8, (n * 3) / 4}};
  }

  double max_abs = 0.0;
  double max_rel = 0.0;
  for (auto [row, col] : points) {
    const double expected = ReferenceElement(a, b, m, n, k, row, col);
    const double actual = c[row * n + col];
    const double abs_err = std::abs(actual - expected);
    const double rel_err = abs_err / std::max(1.0, std::abs(expected));
    max_abs = std::max(max_abs, abs_err);
    max_rel = std::max(max_rel, rel_err);
    if (abs_err > 2.0e-3 && rel_err > 2.0e-4) {
      std::fprintf(stderr,
                   "validation failed at (%u, %u): actual=%g expected=%g "
                   "abs=%g rel=%g\n",
                   row, col, actual, expected, abs_err, rel_err);
      return false;
    }
  }

  std::printf("validation: checked %zu values, max_abs=%g max_rel=%g\n",
              points.size(), max_abs, max_rel);
  return true;
}

DispatchShape ShapeForKernel(const std::string& function_name, uint32_t m,
                             uint32_t n, uint32_t k) {
  if (function_name == "matmul_simdgroup_8x8") {
    if ((m % 16) != 0 || (n % 32) != 0 || (k % 8) != 0) {
      std::fprintf(stderr,
                   "matmul_simdgroup_8x8 requires M multiple of 16, N "
                   "multiple of 32, and K multiple of 8\n");
      std::exit(2);
    }
    return {MTLSizeMake((n + 31) / 32, (m + 15) / 16, 1),
            MTLSizeMake(256, 1, 1)};
  }

  if (function_name == "matmul_tiled16") {
    return {MTLSizeMake((n + 15) / 16, (m + 15) / 16, 1),
            MTLSizeMake(16, 16, 1)};
  }

  std::fprintf(stderr, "unknown kernel function: %s\n", function_name.c_str());
  std::exit(2);
}

bool RunOnce(id<MTLCommandQueue> queue, id<MTLComputePipelineState> pipeline,
             id<MTLBuffer> a, id<MTLBuffer> b, id<MTLBuffer> c,
             const MatmulParams& params, const DispatchShape& shape,
             double* gpu_ms) {
  @autoreleasepool {
    id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder =
        [command_buffer computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:a offset:0 atIndex:0];
    [encoder setBuffer:b offset:0 atIndex:1];
    [encoder setBuffer:c offset:0 atIndex:2];
    [encoder setBytes:&params length:sizeof(params) atIndex:3];
    [encoder dispatchThreadgroups:shape.threadgroups
             threadsPerThreadgroup:shape.threads_per_threadgroup];
    [encoder endEncoding];
    [command_buffer commit];
    [command_buffer waitUntilCompleted];

    if (command_buffer.status != MTLCommandBufferStatusCompleted) {
      std::fprintf(stderr, "command buffer failed: %s\n",
                   command_buffer.error.localizedDescription.UTF8String);
      return false;
    }

    const CFTimeInterval start = command_buffer.GPUStartTime;
    const CFTimeInterval end = command_buffer.GPUEndTime;
    *gpu_ms = (end > start) ? (end - start) * 1000.0 : 0.0;
    return true;
  }
}

}  // namespace

int main(int argc, char** argv) {
  @autoreleasepool {
    if (argc != 8) {
      std::fprintf(stderr,
                   "usage: %s <metallib> <function> <M> <N> <K> <iters> "
                   "<warmup>\n",
                   argv[0]);
      return 2;
    }

    const char* metallib_path = argv[1];
    const std::string function_name = argv[2];
    MatmulParams params = {
        ParseU32(argv[3], "M"), ParseU32(argv[4], "N"),
        ParseU32(argv[5], "K"), 0};
    const uint32_t iterations = ParseU32(argv[6], "iters");
    const uint32_t warmup = ParseU32(argv[7], "warmup");
    const DispatchShape shape =
        ShapeForKernel(function_name, params.m, params.n, params.k);

    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (device == nil) {
      std::fprintf(stderr, "no Metal device available\n");
      return 1;
    }
    std::printf("device: %s\n", device.name.UTF8String);

    NSError* error = nil;
    NSURL* url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:metallib_path]];
    id<MTLLibrary> library = [device newLibraryWithURL:url error:&error];
    if (library == nil) {
      std::fprintf(stderr, "failed to load metallib: %s\n",
                   error.localizedDescription.UTF8String);
      return 1;
    }

    id<MTLFunction> function =
        [library newFunctionWithName:[NSString stringWithUTF8String:function_name.c_str()]];
    if (function == nil) {
      std::fprintf(stderr, "function not found: %s\n", function_name.c_str());
      return 1;
    }

    id<MTLComputePipelineState> pipeline =
        [device newComputePipelineStateWithFunction:function error:&error];
    if (pipeline == nil) {
      std::fprintf(stderr, "failed to create pipeline: %s\n",
                   error.localizedDescription.UTF8String);
      return 1;
    }

    const size_t a_count = static_cast<size_t>(params.m) * params.k;
    const size_t b_count = static_cast<size_t>(params.k) * params.n;
    const size_t c_count = static_cast<size_t>(params.m) * params.n;
    id<MTLBuffer> a =
        [device newBufferWithLength:a_count * sizeof(float)
                            options:MTLResourceStorageModeShared];
    id<MTLBuffer> b =
        [device newBufferWithLength:b_count * sizeof(float)
                            options:MTLResourceStorageModeShared];
    id<MTLBuffer> c =
        [device newBufferWithLength:c_count * sizeof(float)
                            options:MTLResourceStorageModeShared];
    if (a == nil || b == nil || c == nil) {
      std::fprintf(stderr, "failed to allocate buffers\n");
      return 1;
    }

    FillInputs(static_cast<float*>(a.contents), static_cast<float*>(b.contents),
               a_count, b_count);
    std::fill_n(static_cast<float*>(c.contents), c_count, 0.0f);

    id<MTLCommandQueue> queue = [device newCommandQueue];
    if (queue == nil) {
      std::fprintf(stderr, "failed to create command queue\n");
      return 1;
    }

    double gpu_ms = 0.0;
    for (uint32_t i = 0; i < warmup; ++i) {
      if (!RunOnce(queue, pipeline, a, b, c, params, shape, &gpu_ms)) return 1;
    }
    if (!ValidateResult(static_cast<const float*>(a.contents),
                        static_cast<const float*>(b.contents),
                        static_cast<const float*>(c.contents), params.m,
                        params.n, params.k)) {
      return 1;
    }

    double best_ms = std::numeric_limits<double>::infinity();
    double total_ms = 0.0;
    for (uint32_t i = 0; i < iterations; ++i) {
      if (!RunOnce(queue, pipeline, a, b, c, params, shape, &gpu_ms)) return 1;
      if (gpu_ms <= 0.0) {
        std::fprintf(stderr, "GPU timestamp unavailable\n");
        return 1;
      }
      best_ms = std::min(best_ms, gpu_ms);
      total_ms += gpu_ms;
    }

    const double flops = 2.0 * params.m * params.n * params.k;
    const double best_gflops = flops / (best_ms * 1.0e6);
    const double mean_ms = total_ms / iterations;
    const double mean_gflops = flops / (mean_ms * 1.0e6);
    std::printf(
        "kernel=%s M=%u N=%u K=%u iterations=%u best=%.3f ms %.1f GFLOP/s "
        "mean=%.3f ms %.1f GFLOP/s\n",
        function_name.c_str(), params.m, params.n, params.k, iterations,
        best_ms, best_gflops, mean_ms, mean_gflops);
  }

  return 0;
}
