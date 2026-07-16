/* Copyright 2024 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "xla/service/gpu/stream_executor_util.h"

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/time/time.h"
#include "xla/autotuning.pb.h"
#include "xla/service/gpu/cublas_cudnn.h"
#include "xla/service/hlo_module_config.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/kernel_spec.h"
#include "xla/stream_executor/mock_platform.h"
#include "xla/stream_executor/mock_stream_executor.h"
#include "xla/stream_executor/musa/musa_platform_id.h"
#include "xla/stream_executor/platform_id.h"
#include "xla/stream_executor/rocm/rocm_platform_id.h"
#include "xla/tsl/protobuf/dnn.pb.h"
#include "xla/tsl/util/proto/proto_utils.h"

namespace xla::gpu {
namespace {
using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;
using ::stream_executor::dnn::ConvolutionKind;
using ::testing::ElementsAre;
using ::testing::Return;

struct Result {
  int64_t run_time_ns;
  int64_t scratch_bytes;

  bool operator==(const Result& other) const {
    return other.run_time_ns == run_time_ns &&
           other.scratch_bytes == scratch_bytes;
  };

  explicit operator AutotuneResult() const {
    AutotuneResult result;
    *result.mutable_run_time() =
        tsl::proto_utils::ToDurationProto(absl::Nanoseconds(run_time_ns));
    result.set_scratch_bytes(scratch_bytes);
    return result;
  }
};

static Result ATRToResult(AutotuneResult atr) {
  return Result{.run_time_ns = absl::ToInt64Nanoseconds(
                    tsl::proto_utils::FromDurationProto(atr.run_time())),
                .scratch_bytes = atr.scratch_bytes()};
}

std::vector<AutotuneResult> Results(const std::vector<Result>& stats) {
  std::vector<AutotuneResult> results;
  for (const auto& s : stats) results.push_back(AutotuneResult(s));
  return results;
}

void ExpectBinaryKernelFormat(stream_executor::PlatformId platform_id,
                              bool expect_mubin) {
  stream_executor::MockPlatform platform;
  stream_executor::MockStreamExecutor executor;
  static constexpr std::array<uint8_t, 4> kBinary = {0x7f, 'E', 'L', 'F'};

  EXPECT_CALL(platform, id).WillRepeatedly(Return(platform_id));
  EXPECT_CALL(executor, GetPlatform).WillRepeatedly(Return(&platform));
  EXPECT_CALL(executor, LoadKernel)
      .WillOnce(
          [expect_mubin](const stream_executor::KernelLoaderSpec& spec)
              -> absl::StatusOr<std::unique_ptr<stream_executor::Kernel>> {
            EXPECT_EQ(spec.has_musa_mubin_in_memory(), expect_mubin);
            EXPECT_EQ(spec.has_cuda_cubin_in_memory(), !expect_mubin);
            EXPECT_EQ(spec.kernel_name(), "test_kernel");
            EXPECT_EQ(spec.arity(), 3);
            if (expect_mubin) {
              std::optional<stream_executor::MusaMubinInMemory> mubin =
                  spec.musa_mubin_in_memory();
              if (mubin.has_value()) {
                EXPECT_THAT(mubin->mubin_bytes,
                            ElementsAre(0x7f, 'E', 'L', 'F'));
              }
            } else {
              std::optional<stream_executor::CudaCubinInMemory> cubin =
                  spec.cuda_cubin_in_memory();
              if (cubin.has_value()) {
                EXPECT_THAT(cubin->cubin_bytes,
                            ElementsAre(0x7f, 'E', 'L', 'F'));
              }
            }
            return absl::CancelledError("loader spec inspected");
          });

  EXPECT_THAT(CreateKernel("test_kernel", /*num_args=*/3, kBinary, &executor),
              StatusIs(absl::StatusCode::kCancelled, "loader spec inspected"));
}

TEST(StreamExecutorTest, PickBestResult) {
  absl::StatusOr<AutotuneResult> atr;

  atr = PickBestResult(Results({{9000, 0}, {1000, 0}, {16000, 0}}), "", {});
  EXPECT_EQ(ATRToResult(atr.value()), Result({1000, 0}));

  atr = PickBestResult(Results({{4700, 0}, {4600, 0}, {4500, 0}}), "", {});
  EXPECT_EQ(ATRToResult(atr.value()), Result({4500, 0}));

  atr = PickBestResult(Results({{4700, 0}, {4600, 2}, {4500, 1}}), "", {});
  EXPECT_EQ(ATRToResult(atr.value()), Result({4700, 0}));

  atr = PickBestResult(Results({{5000, 1}, {6000, 0}, {7500, 0}}), "", {});
  EXPECT_EQ(ATRToResult(atr.value()), Result({6000, 0}));
}

TEST(StreamExecutorUtilTest, CudnnConvKindToProto) {
  EXPECT_EQ(CudnnConvKindToProto(CudnnConvKind::kBackwardFilter),
            ConvolutionKind::BACKWARD_FILTER);
  EXPECT_EQ(CudnnConvKindToProto(CudnnConvKind::kBackwardInput),
            ConvolutionKind::BACKWARD_DATA);
  EXPECT_EQ(CudnnConvKindToProto(CudnnConvKind::kForward),
            ConvolutionKind::FORWARD);
  EXPECT_EQ(CudnnConvKindToProto(CudnnConvKind::kForwardActivation),
            ConvolutionKind::FORWARD_BIAS_ACTIVATION);
  EXPECT_EQ(CudnnConvKindToProto(CudnnConvKind::kForwardGraph),
            ConvolutionKind::FORWARD_GRAPH);
}

TEST(StreamExecutorUtilTest, CudnnConvKindFromProto) {
  EXPECT_THAT(CudnnConvKindFromProto(ConvolutionKind::BACKWARD_FILTER),
              IsOkAndHolds(CudnnConvKind::kBackwardFilter));
  EXPECT_THAT(CudnnConvKindFromProto(ConvolutionKind::BACKWARD_DATA),
              IsOkAndHolds(CudnnConvKind::kBackwardInput));
  EXPECT_THAT(CudnnConvKindFromProto(ConvolutionKind::FORWARD),
              IsOkAndHolds(CudnnConvKind::kForward));
  EXPECT_THAT(CudnnConvKindFromProto(ConvolutionKind::FORWARD_BIAS_ACTIVATION),
              IsOkAndHolds(CudnnConvKind::kForwardActivation));
  EXPECT_THAT(CudnnConvKindFromProto(ConvolutionKind::FORWARD_GRAPH),
              IsOkAndHolds(CudnnConvKind::kForwardGraph));

  EXPECT_THAT(CudnnConvKindFromProto(ConvolutionKind::INVALID),
              StatusIs(absl::StatusCode::kInternal));
}

TEST(StreamExecutorUtilTest, RoutesMusaBinaryToMubinLoaderSpec) {
  ExpectBinaryKernelFormat(stream_executor::musa::kMusaPlatformId,
                           /*expect_mubin=*/true);
}

TEST(StreamExecutorUtilTest, PreservesRocmCubinLoaderSpecCompatibility) {
  ExpectBinaryKernelFormat(stream_executor::rocm::kROCmPlatformId,
                           /*expect_mubin=*/false);
}

}  // namespace

}  // namespace xla::gpu
