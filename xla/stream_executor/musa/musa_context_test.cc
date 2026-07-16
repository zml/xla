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

#include "xla/stream_executor/musa/musa_context.h"

#include <atomic>
#include <cstdint>
#include <memory>
#include <thread>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/stream_executor/gpu/scoped_activate_context.h"
#include "xla/stream_executor/musa/musa_driver.h"
#include "xla/tsl/platform/test.h"

namespace stream_executor::musa {
namespace {

thread_local MUcontext fake_current_context = nullptr;

MUcontext ContextHandle(uintptr_t value) {
  return reinterpret_cast<MUcontext>(value);
}

class FakeMusaDriver final : public MusaDriver {
 public:
  absl::Status Init() override {
    ++init_count;
    return init_status;
  }

  absl::StatusOr<MUdevice> Device(int ordinal) override {
    last_ordinal = ordinal;
    return device;
  }

  absl::StatusOr<MusaPrimaryContextState> PrimaryContextState(
      MUdevice queried_device) override {
    last_device = queried_device;
    return primary_context_state;
  }

  absl::Status SetPrimaryContextFlags(MUdevice queried_device,
                                      unsigned int flags) override {
    last_device = queried_device;
    last_flags = flags;
    ++set_flags_count;
    return set_flags_status;
  }

  absl::StatusOr<MUcontext> RetainPrimaryContext(
      MUdevice queried_device) override {
    last_device = queried_device;
    ++retain_count;
    return retained_context;
  }

  absl::Status ReleasePrimaryContext(MUdevice queried_device) override {
    last_device = queried_device;
    ++release_count;
    return release_status;
  }

  absl::Status SetCurrentContext(MUcontext context) override {
    fake_current_context = context;
    ++set_current_count;
    return set_current_status;
  }

  absl::StatusOr<MUcontext> CurrentContext() override {
    ++get_current_count;
    return fake_current_context;
  }

  absl::Status SynchronizeContext() override {
    synchronized_while_active = fake_current_context == retained_context;
    ++synchronize_count;
    return synchronize_status;
  }

  int device = 7;
  MusaPrimaryContextState primary_context_state = {.flags = MU_CTX_SCHED_AUTO,
                                                   .active = false};
  MUcontext retained_context = ContextHandle(0x1234);

  absl::Status init_status = absl::OkStatus();
  absl::Status set_flags_status = absl::OkStatus();
  absl::Status release_status = absl::OkStatus();
  absl::Status set_current_status = absl::OkStatus();
  absl::Status synchronize_status = absl::OkStatus();

  std::atomic<int> init_count = 0;
  std::atomic<int> retain_count = 0;
  std::atomic<int> release_count = 0;
  std::atomic<int> set_flags_count = 0;
  std::atomic<int> set_current_count = 0;
  std::atomic<int> get_current_count = 0;
  std::atomic<int> synchronize_count = 0;
  std::atomic<bool> synchronized_while_active = false;
  int last_ordinal = -1;
  MUdevice last_device = -1;
  unsigned int last_flags = ~0u;
};

TEST(MusaContextTest, InactivePrimaryContextIsConfiguredAndBalanced) {
  fake_current_context = nullptr;
  FakeMusaDriver driver;
  driver.primary_context_state = {.flags = MU_CTX_SCHED_SPIN, .active = false};

  auto context_or = MusaContext::Create(/*device_ordinal=*/3, &driver);
  ASSERT_TRUE(context_or.ok()) << context_or.status();
  std::unique_ptr<MusaContext> context = std::move(*context_or);

  EXPECT_EQ(driver.init_count, 1);
  EXPECT_EQ(driver.last_ordinal, 3);
  EXPECT_EQ(driver.last_device, driver.device);
  EXPECT_EQ(driver.set_flags_count, 1);
  EXPECT_EQ(driver.last_flags, MU_CTX_SCHED_AUTO);
  EXPECT_EQ(driver.retain_count, 1);
  EXPECT_EQ(driver.release_count, 0);
  EXPECT_EQ(context->context(), driver.retained_context);
  EXPECT_EQ(context->device(), driver.device);
  EXPECT_EQ(context->device_ordinal(), 3);

  context->SetActive();
  EXPECT_TRUE(context->IsActive());
  EXPECT_TRUE(context->Synchronize().ok());
  EXPECT_TRUE(driver.synchronized_while_active);

  context.reset();
  EXPECT_EQ(fake_current_context, nullptr);
  EXPECT_EQ(driver.retain_count, 1);
  EXPECT_EQ(driver.release_count, 1);
}

TEST(MusaContextTest, ActivePrimaryContextWithDifferentFlagsIsReused) {
  fake_current_context = nullptr;
  FakeMusaDriver driver;
  driver.primary_context_state = {.flags = MU_CTX_SCHED_SPIN, .active = true};

  auto context_or = MusaContext::Create(/*device_ordinal=*/0, &driver);
  ASSERT_TRUE(context_or.ok()) << context_or.status();
  std::unique_ptr<MusaContext> context = std::move(*context_or);

  EXPECT_EQ(driver.set_flags_count, 0);
  EXPECT_EQ(driver.retain_count, 1);
  context.reset();
  EXPECT_EQ(driver.release_count, 1);
}

TEST(MusaContextTest, NullRetainedContextIsRejectedWithoutReleasing) {
  fake_current_context = nullptr;
  FakeMusaDriver driver;
  driver.retained_context = nullptr;

  auto context = MusaContext::Create(/*device_ordinal=*/0, &driver);

  ASSERT_FALSE(context.ok());
  EXPECT_EQ(context.status().code(), absl::StatusCode::kInternal);
  EXPECT_EQ(driver.retain_count, 1);
  EXPECT_EQ(driver.release_count, 0);
}

TEST(MusaContextTest, TeardownDoesNotClearAForeignCurrentContext) {
  FakeMusaDriver driver;
  auto context_or = MusaContext::Create(/*device_ordinal=*/0, &driver);
  ASSERT_TRUE(context_or.ok()) << context_or.status();
  std::unique_ptr<MusaContext> context = std::move(*context_or);

  MUcontext foreign_context = ContextHandle(0x5678);
  fake_current_context = foreign_context;
  context.reset();

  EXPECT_EQ(fake_current_context, foreign_context);
  EXPECT_EQ(driver.release_count, 1);
}

TEST(MusaContextTest, ActivatesIndependentlyOnWorkerThreads) {
  fake_current_context = nullptr;
  FakeMusaDriver driver;
  auto context_or = MusaContext::Create(/*device_ordinal=*/0, &driver);
  ASSERT_TRUE(context_or.ok()) << context_or.status();
  std::unique_ptr<MusaContext> context = std::move(*context_or);

  constexpr int kThreadCount = 8;
  constexpr int kIterations = 100;
  std::atomic<bool> all_active = true;
  std::vector<std::thread> threads;
  threads.reserve(kThreadCount);
  for (int i = 0; i < kThreadCount; ++i) {
    threads.emplace_back([&] {
      for (int j = 0; j < kIterations; ++j) {
        gpu::ScopedActivateContext activation(context.get());
        if (!context->IsActive()) {
          all_active = false;
        }
      }
    });
  }
  for (std::thread& thread : threads) {
    thread.join();
  }

  EXPECT_TRUE(all_active);
  EXPECT_EQ(driver.set_current_count, kThreadCount * kIterations);
  context.reset();
  EXPECT_EQ(driver.retain_count, 1);
  EXPECT_EQ(driver.release_count, 1);
}

TEST(MusaContextTest, NestedActivationRestoresTheOuterContext) {
  fake_current_context = nullptr;
  FakeMusaDriver outer_driver;
  outer_driver.device = 7;
  outer_driver.retained_context = ContextHandle(0x1234);
  FakeMusaDriver inner_driver;
  inner_driver.device = 8;
  inner_driver.retained_context = ContextHandle(0x5678);

  auto outer_or = MusaContext::Create(/*device_ordinal=*/0, &outer_driver);
  ASSERT_TRUE(outer_or.ok()) << outer_or.status();
  std::unique_ptr<MusaContext> outer = std::move(*outer_or);
  auto inner_or = MusaContext::Create(/*device_ordinal=*/1, &inner_driver);
  ASSERT_TRUE(inner_or.ok()) << inner_or.status();
  std::unique_ptr<MusaContext> inner = std::move(*inner_or);

  {
    gpu::ScopedActivateContext outer_activation(outer.get());
    EXPECT_EQ(fake_current_context, outer_driver.retained_context);
    {
      gpu::ScopedActivateContext inner_activation(inner.get());
      EXPECT_EQ(fake_current_context, inner_driver.retained_context);
    }
    EXPECT_EQ(fake_current_context, outer_driver.retained_context);
  }

  inner.reset();
  outer.reset();
  EXPECT_EQ(inner_driver.release_count, 1);
  EXPECT_EQ(outer_driver.release_count, 1);
}

TEST(MusaContextTest, RejectsNullDriver) {
  auto context = MusaContext::Create(/*device_ordinal=*/0, nullptr);
  ASSERT_FALSE(context.ok());
  EXPECT_EQ(context.status().code(), absl::StatusCode::kInvalidArgument);
}

}  // namespace
}  // namespace stream_executor::musa
