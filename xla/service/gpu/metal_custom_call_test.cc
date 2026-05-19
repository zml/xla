/* Copyright 2026 The OpenXLA Authors.

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

#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "xla/client/client.h"
#include "xla/client/client_library.h"
#include "xla/ffi/api/c_api_internal.h"
#include "xla/ffi/ffi.h"
#include "xla/ffi/ffi_api.h"
#include "xla/hlo/builder/xla_builder.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/service/platform_util.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/stream.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace ffi {

struct MetalTestStream {};

template <>
struct CtxDecoding<MetalTestStream> {
  using Type = se::Stream*;

  static std::optional<Type> Decode(const XLA_FFI_Api* api,
                                    XLA_FFI_ExecutionContext* ctx,
                                    DiagnosticEngine& diagnostic) {
    return internal::DecodeInternalCtx<Type>(
        api, ctx, diagnostic, api->internal_api->XLA_FFI_INTERNAL_Stream_Get,
        "stream");
  }
};

}  // namespace ffi

namespace {

absl::StatusOr<LocalClient*> GetMetalClient() {
  TF_ASSIGN_OR_RETURN(se::Platform * platform,
                      PlatformUtil::GetPlatform("metal"));
  if (platform->VisibleDeviceCount() == 0) {
    return absl::FailedPreconditionError("No visible Metal devices.");
  }
  LocalClientOptions options;
  options.set_platform(platform);
  return ClientLibrary::GetOrCreateLocalClient(options);
}

static absl::Status AddOne(se::Stream* stream, ffi::AnyBuffer src,
                           ffi::Result<ffi::AnyBuffer> dst) {
  int32_t data[4];
  se::DeviceAddressBase src_mem = src.device_memory();
  se::DeviceAddressBase dst_mem = dst->device_memory();
  TF_RETURN_IF_ERROR(stream->Memcpy(data, src_mem, sizeof(data)));
  TF_RETURN_IF_ERROR(stream->BlockHostUntilDone());
  for (int32_t& value : data) {
    value += 1;
  }
  TF_RETURN_IF_ERROR(stream->Memcpy(&dst_mem, data, sizeof(data)));
  TF_RETURN_IF_ERROR(stream->BlockHostUntilDone());
  return absl::OkStatus();
}

XLA_FFI_DEFINE_HANDLER(kAddOne, AddOne,
                       ffi::Ffi::Bind()
                           .Ctx<ffi::MetalTestStream>()
                           .Arg<ffi::AnyBuffer>()
                           .Ret<ffi::AnyBuffer>());

TEST(MetalCustomCallTest, TypedFfiAddOne) {
  auto client_result = GetMetalClient();
  if (absl::IsFailedPrecondition(client_result.status())) {
    GTEST_SKIP() << client_result.status();
  }
  TF_ASSERT_OK_AND_ASSIGN(LocalClient * client, std::move(client_result));

  XLA_FFI_Error* registration_error = ffi::Ffi::RegisterStaticHandler(
      ffi::GetXlaFfiApi(), "__xla_test$$metal_add_one", "METAL", kAddOne);
  ASSERT_EQ(registration_error, nullptr);

  XlaBuilder builder("metal_custom_call_add_one");
  Shape shape = ShapeUtil::MakeShape(S32, {4});
  XlaOp input = Parameter(&builder, 0, shape, "input");
  CustomCall(&builder, "__xla_test$$metal_add_one", /*operands=*/{input},
             shape, /*opaque=*/"", /*has_side_effect=*/false,
             /*output_operand_aliasing=*/{}, /*literal=*/nullptr,
             CustomCallSchedule::SCHEDULE_NONE,
             CustomCallApiVersion::API_VERSION_TYPED_FFI);
  TF_ASSERT_OK_AND_ASSIGN(XlaComputation computation, builder.Build());

  Literal input_literal = LiteralUtil::CreateR1<int32_t>({1, 2, 3, 4});
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<GlobalData> input_data,
                          client->TransferToServer(input_literal));
  std::vector<GlobalData*> arguments = {input_data.get()};
  TF_ASSERT_OK_AND_ASSIGN(Literal actual,
                          client->ExecuteAndTransfer(computation, arguments));

  ASSERT_TRUE(ShapeUtil::Compatible(actual.shape(), shape));
  for (int64_t i = 0; i < 4; ++i) {
    EXPECT_EQ(actual.Get<int32_t>({i}), i + 2);
  }
}

}  // namespace
}  // namespace xla
