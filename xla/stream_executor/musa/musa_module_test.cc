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

#include "xla/stream_executor/musa/musa_module.h"

#include <array>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "musa.h"
#include "xla/stream_executor/module_spec.h"
#include "xla/stream_executor/musa/musa_context.h"
#include "xla/stream_executor/musa/musa_driver.h"

namespace stream_executor::musa {
namespace {

constexpr size_t kElfHeaderSize = 64;
constexpr size_t kProgramHeaderSize = 56;
constexpr size_t kLoadHeader = kElfHeaderSize;
constexpr size_t kNoteHeader = kLoadHeader + kProgramHeaderSize;
constexpr size_t kLoadOffset = 192;
constexpr size_t kNoteOffset = 208;

thread_local MUcontext fake_current_context = nullptr;

void WriteU16(std::vector<uint8_t>& bytes, size_t offset, uint16_t value) {
  bytes[offset] = value;
  bytes[offset + 1] = value >> 8;
}

void WriteU32(std::vector<uint8_t>& bytes, size_t offset, uint32_t value) {
  for (int i = 0; i < 4; ++i) bytes[offset + i] = value >> (8 * i);
}

void WriteU64(std::vector<uint8_t>& bytes, size_t offset, uint64_t value) {
  for (int i = 0; i < 8; ++i) bytes[offset + i] = value >> (8 * i);
}

std::vector<uint8_t> MakeValidMubin(uint8_t descriptor = 0xde) {
  std::vector<uint8_t> bytes(240);
  bytes[0] = 0x7f;
  bytes[1] = 'E';
  bytes[2] = 'L';
  bytes[3] = 'F';
  bytes[4] = 2;
  bytes[5] = 1;
  bytes[6] = 1;
  bytes[8] = 7;
  WriteU16(bytes, 16, 3);
  WriteU16(bytes, 18, 253);
  WriteU32(bytes, 20, 1);
  WriteU64(bytes, 32, kElfHeaderSize);
  WriteU16(bytes, 52, kElfHeaderSize);
  WriteU16(bytes, 54, kProgramHeaderSize);
  WriteU16(bytes, 56, 2);

  WriteU32(bytes, kLoadHeader, 1);
  WriteU32(bytes, kLoadHeader + 4, 5);
  WriteU64(bytes, kLoadHeader + 8, kLoadOffset);
  WriteU64(bytes, kLoadHeader + 32, 8);
  WriteU64(bytes, kLoadHeader + 40, 8);

  WriteU32(bytes, kNoteHeader, 4);
  WriteU32(bytes, kNoteHeader + 4, 4);
  WriteU64(bytes, kNoteHeader + 8, kNoteOffset);
  WriteU64(bytes, kNoteHeader + 32, 24);
  WriteU64(bytes, kNoteHeader + 40, 24);

  WriteU32(bytes, kNoteOffset, 6);
  WriteU32(bytes, kNoteOffset + 4, 4);
  WriteU32(bytes, kNoteOffset + 8, 0x40);
  const std::string owner = "MTGPU";
  for (size_t i = 0; i < owner.size(); ++i) {
    bytes[kNoteOffset + 12 + i] = owner[i];
  }
  bytes[kNoteOffset + 20] = descriptor;
  bytes[kNoteOffset + 21] = 0xad;
  bytes[kNoteOffset + 22] = 0xbe;
  bytes[kNoteOffset + 23] = 0xef;
  return bytes;
}

class FakeModuleDriver final : public MusaDriver {
 public:
  absl::Status Init() override { return absl::OkStatus(); }
  absl::StatusOr<MUdevice> Device(int ordinal) override {
    last_ordinal = ordinal;
    return device;
  }
  absl::StatusOr<MusaPrimaryContextState> PrimaryContextState(
      MUdevice) override {
    return MusaPrimaryContextState{MU_CTX_SCHED_AUTO, true};
  }
  absl::StatusOr<MUcontext> RetainPrimaryContext(MUdevice) override {
    ++retain_calls;
    return retained_context;
  }
  absl::Status ReleasePrimaryContext(MUdevice) override {
    ++release_calls;
    return absl::OkStatus();
  }
  absl::Status SetCurrentContext(MUcontext context) override {
    fake_current_context = context;
    return absl::OkStatus();
  }
  absl::StatusOr<MUcontext> CurrentContext() override {
    return fake_current_context;
  }

  absl::StatusOr<MUmodule> LoadModuleData(const void* image) override {
    ++load_calls;
    last_image = image;
    loaded_while_active = fake_current_context == retained_context;
    if (!load_status.ok()) return load_status;
    if (block_loads) {
      std::unique_lock<std::mutex> lock(load_mutex);
      ++load_arrivals;
      load_condition.notify_all();
      load_condition.wait(lock, [this] { return load_arrivals >= 2; });
    }
    uintptr_t value = 0x1000 + 0x10 * next_module.fetch_add(1);
    return reinterpret_cast<MUmodule>(value);
  }

  absl::Status UnloadModule(MUmodule module) override {
    ++unload_calls;
    last_unloaded = module;
    unloaded_while_active = fake_current_context == retained_context;
    if (on_unload) on_unload();
    return unload_status;
  }

  absl::StatusOr<MUfunction> GetModuleFunction(MUmodule module,
                                               const char* name) override {
    ++function_lookup_calls;
    last_lookup_module = module;
    last_lookup_name = name;
    lookup_while_active = fake_current_context == retained_context;
    return function;
  }

  absl::StatusOr<MusaModuleGlobal> GetModuleGlobal(MUmodule module,
                                                   const char* name) override {
    ++global_lookup_calls;
    last_lookup_module = module;
    last_lookup_name = name;
    lookup_while_active = fake_current_context == retained_context;
    return global;
  }

  int device = 7;
  MUcontext retained_context = reinterpret_cast<MUcontext>(uintptr_t{0x1234});
  int last_ordinal = -1;
  std::atomic<const void*> last_image{nullptr};
  MUmodule last_unloaded = nullptr;
  std::atomic<int> retain_calls{0};
  std::atomic<int> release_calls{0};
  std::atomic<int> load_calls{0};
  std::atomic<int> unload_calls{0};
  std::atomic<int> function_lookup_calls{0};
  std::atomic<int> global_lookup_calls{0};
  std::atomic<uintptr_t> next_module{1};
  std::atomic<bool> loaded_while_active{false};
  std::atomic<bool> unloaded_while_active{false};
  std::atomic<bool> lookup_while_active{false};
  MUfunction function = reinterpret_cast<MUfunction>(uintptr_t{0x5678});
  MusaModuleGlobal global{static_cast<MUdeviceptr>(0x9000), 64};
  MUmodule last_lookup_module = nullptr;
  std::string last_lookup_name;
  absl::Status load_status = absl::OkStatus();
  absl::Status unload_status = absl::OkStatus();
  std::function<void()> on_unload;
  bool block_loads = false;
  std::mutex load_mutex;
  std::condition_variable load_condition;
  int load_arrivals = 0;
};

absl::StatusOr<std::shared_ptr<MusaContext>> MakeContext(
    FakeModuleDriver* driver, int ordinal = 0) {
  auto context = MusaContext::Create(ordinal, driver);
  if (!context.ok()) return context.status();
  return std::shared_ptr<MusaContext>(std::move(*context));
}

TEST(MusaModuleTest, OwnsBytesAndUnloadsUnderRetainedContext) {
  fake_current_context = nullptr;
  FakeModuleDriver driver;
  auto context_or = MakeContext(&driver);
  ASSERT_TRUE(context_or.ok()) << context_or.status();
  std::shared_ptr<MusaContext> context = *std::move(context_or);
  std::vector<uint8_t> image = MakeValidMubin();

  auto module_or = MusaModule::Load(&driver, context, image, "mp_21");
  ASSERT_TRUE(module_or.ok()) << module_or.status();
  std::shared_ptr<MusaModule> module = *std::move(module_or);
  const uint8_t original_first_byte = module->bytes().front();
  image.front() = 0;

  EXPECT_EQ(module->bytes().front(), original_first_byte);
  EXPECT_EQ(driver.last_image.load(), module->bytes().data());
  EXPECT_TRUE(driver.loaded_while_active);
  EXPECT_EQ(module->architecture(), "mp_21");
  EXPECT_EQ(module->binary_abi_version(), kMubinLoaderAbiVersion);
  EXPECT_EQ(module->binary_kind(), MusaBinaryKind::kMubin);

  module.reset();
  EXPECT_EQ(driver.unload_calls, 1);
  EXPECT_TRUE(driver.unloaded_while_active);
}

TEST(MusaModuleTest, CachesFunctionsAndGlobalsUnderOwningContext) {
  fake_current_context = nullptr;
  FakeModuleDriver driver;
  auto context_or = MakeContext(&driver);
  ASSERT_TRUE(context_or.ok()) << context_or.status();
  auto module_or = MusaModule::Load(&driver, *std::move(context_or),
                                    MakeValidMubin(), "mp_21");
  ASSERT_TRUE(module_or.ok()) << module_or.status();
  std::shared_ptr<MusaModule> module = *std::move(module_or);

  auto first_function = module->GetFunction("add_one");
  auto second_function = module->GetFunction("add_one");
  ASSERT_TRUE(first_function.ok()) << first_function.status();
  ASSERT_TRUE(second_function.ok()) << second_function.status();
  EXPECT_EQ(*first_function, driver.function);
  EXPECT_EQ(*second_function, driver.function);
  EXPECT_EQ(driver.function_lookup_calls, 1);

  auto first_global = module->GetGlobal("constant");
  auto second_global = module->GetGlobal("constant");
  ASSERT_TRUE(first_global.ok()) << first_global.status();
  ASSERT_TRUE(second_global.ok()) << second_global.status();
  EXPECT_EQ(first_global->address, driver.global.address);
  EXPECT_EQ(first_global->size, driver.global.size);
  EXPECT_EQ(second_global->address, driver.global.address);
  EXPECT_EQ(driver.global_lookup_calls, 1);
  EXPECT_TRUE(driver.lookup_while_active);
}

TEST(MusaModuleCacheTest, DeduplicatesAndCountsHandleClients) {
  fake_current_context = nullptr;
  FakeModuleDriver driver;
  auto context_or = MakeContext(&driver);
  ASSERT_TRUE(context_or.ok()) << context_or.status();
  std::shared_ptr<MusaContext> context = *std::move(context_or);
  MusaModuleCache cache(&driver, context, "mp_21");
  std::vector<uint8_t> image = MakeValidMubin();

  auto first = cache.AcquireModuleHandle(image);
  auto second = cache.AcquireModuleHandle(image);
  ASSERT_TRUE(first.ok()) << first.status();
  ASSERT_TRUE(second.ok()) << second.status();
  EXPECT_EQ(*first, *second);
  EXPECT_EQ(driver.load_calls, 1);
  EXPECT_FALSE(cache.IsQuiescent());

  EXPECT_TRUE(cache.ReleaseModuleHandle(*first));
  EXPECT_EQ(driver.unload_calls, 0);
  EXPECT_TRUE(cache.ReleaseModuleHandle(*second));
  EXPECT_EQ(driver.unload_calls, 1);
  EXPECT_TRUE(cache.IsQuiescent());
  EXPECT_FALSE(cache.ReleaseModuleHandle(*second));
}

TEST(MusaModuleCacheTest, SeparatesDifferentBytesAndForeignHandles) {
  fake_current_context = nullptr;
  FakeModuleDriver driver;
  auto context_or = MakeContext(&driver);
  ASSERT_TRUE(context_or.ok()) << context_or.status();
  std::shared_ptr<MusaContext> context = *std::move(context_or);
  MusaModuleCache first_cache(&driver, context, "mp_21");
  MusaModuleCache second_cache(&driver, context, "mp_21");

  auto first = first_cache.AcquireModuleHandle(MakeValidMubin(0xde));
  auto different = first_cache.AcquireModuleHandle(MakeValidMubin(0xdf));
  ASSERT_TRUE(first.ok());
  ASSERT_TRUE(different.ok());
  EXPECT_NE(*first, *different);
  EXPECT_EQ(driver.load_calls, 2);
  EXPECT_FALSE(second_cache.ReleaseModuleHandle(*first));
  EXPECT_EQ(second_cache.LookupModule(*first).status().code(),
            absl::StatusCode::kNotFound);

  EXPECT_TRUE(first_cache.ReleaseModuleHandle(*first));
  EXPECT_TRUE(first_cache.ReleaseModuleHandle(*different));
}

TEST(MusaModuleCacheTest, FailedLoadIsNotInserted) {
  fake_current_context = nullptr;
  FakeModuleDriver driver;
  auto context_or = MakeContext(&driver);
  ASSERT_TRUE(context_or.ok());
  MusaModuleCache cache(&driver, *std::move(context_or), "mp_21");
  std::vector<uint8_t> image = MakeValidMubin();

  driver.load_status = absl::InternalError("injected load failure");
  auto failed = cache.AcquireModuleHandle(image);
  ASSERT_FALSE(failed.ok());
  EXPECT_TRUE(
      absl::StrContains(failed.status().message(), "injected load failure"));
  EXPECT_TRUE(cache.IsQuiescent());

  driver.load_status = absl::OkStatus();
  auto loaded = cache.AcquireModuleHandle(image);
  ASSERT_TRUE(loaded.ok()) << loaded.status();
  EXPECT_EQ(driver.load_calls, 2);
  EXPECT_TRUE(cache.ReleaseModuleHandle(*loaded));
}

TEST(MusaModuleCacheTest, ConcurrentSpeculativeLoserUnloadsOutsideCache) {
  fake_current_context = nullptr;
  FakeModuleDriver driver;
  driver.block_loads = true;
  auto context_or = MakeContext(&driver);
  ASSERT_TRUE(context_or.ok());
  MusaModuleCache cache(&driver, *std::move(context_or), "mp_21");
  std::vector<uint8_t> image = MakeValidMubin();
  std::array<absl::StatusOr<ModuleHandle>, 2> handles = {
      absl::UnknownError("not started"), absl::UnknownError("not started")};

  std::thread first([&] { handles[0] = cache.AcquireModuleHandle(image); });
  std::thread second([&] { handles[1] = cache.AcquireModuleHandle(image); });
  first.join();
  second.join();

  ASSERT_TRUE(handles[0].ok()) << handles[0].status();
  ASSERT_TRUE(handles[1].ok()) << handles[1].status();
  EXPECT_EQ(*handles[0], *handles[1]);
  EXPECT_EQ(driver.load_calls, 2);
  EXPECT_EQ(driver.unload_calls, 1);
  EXPECT_TRUE(cache.ReleaseModuleHandle(*handles[0]));
  EXPECT_TRUE(cache.ReleaseModuleHandle(*handles[1]));
  EXPECT_EQ(driver.unload_calls, 2);
}

TEST(MusaModuleCacheTest, DirectOwnerControlsQuiescence) {
  fake_current_context = nullptr;
  FakeModuleDriver driver;
  auto context_or = MakeContext(&driver);
  ASSERT_TRUE(context_or.ok());
  MusaModuleCache cache(&driver, *std::move(context_or), "mp_21");

  auto module_or = cache.GetOrLoadModule(MakeValidMubin());
  ASSERT_TRUE(module_or.ok()) << module_or.status();
  std::shared_ptr<MusaModule> module = *std::move(module_or);
  EXPECT_FALSE(cache.IsQuiescent());
  module.reset();
  EXPECT_TRUE(cache.IsQuiescent());
  EXPECT_EQ(driver.unload_calls, 1);
}

TEST(MusaModuleCacheTest, FinalReleaseCanReenterCacheDuringUnload) {
  fake_current_context = nullptr;
  FakeModuleDriver driver;
  auto context_or = MakeContext(&driver);
  ASSERT_TRUE(context_or.ok());
  MusaModuleCache cache(&driver, *std::move(context_or), "mp_21");

  auto handle = cache.AcquireModuleHandle(MakeValidMubin());
  ASSERT_TRUE(handle.ok()) << handle.status();
  driver.on_unload = [&cache] { EXPECT_TRUE(cache.IsQuiescent()); };

  EXPECT_TRUE(cache.ReleaseModuleHandle(*handle));
  EXPECT_EQ(driver.unload_calls, 1);
}

TEST(MusaModuleCacheTest, TeardownReleasesOutstandingHandleClientsOnce) {
  fake_current_context = nullptr;
  FakeModuleDriver driver;
  auto context_or = MakeContext(&driver);
  ASSERT_TRUE(context_or.ok());
  std::shared_ptr<MusaContext> context = *std::move(context_or);

  {
    MusaModuleCache cache(&driver, context, "mp_21");
    auto first = cache.AcquireModuleHandle(MakeValidMubin());
    auto second = cache.AcquireModuleHandle(MakeValidMubin());
    ASSERT_TRUE(first.ok()) << first.status();
    ASSERT_TRUE(second.ok()) << second.status();
    EXPECT_EQ(*first, *second);
    EXPECT_EQ(driver.load_calls, 1);
  }

  EXPECT_EQ(driver.unload_calls, 1);
  EXPECT_TRUE(driver.unloaded_while_active);
}

}  // namespace
}  // namespace stream_executor::musa
