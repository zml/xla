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

#include "xla/stream_executor/musa/musa_runtime.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "musa_runtime_api.h"
#include "xla/stream_executor/musa/musa_dso_loader.h"
#include "xla/stream_executor/musa/musa_status.h"

namespace stream_executor::musa {
namespace {

using SymbolMap = std::map<std::string, void*>;

std::atomic<int> g_memcpy_async_calls = 0;

struct OverlapState {
  std::mutex mutex;
  std::condition_variable condition;
  int active = 0;
  int max_active = 0;
};

OverlapState* g_overlap_state = nullptr;

musaError_t MUSARTAPI GetDeviceCountStub(int* count) {
  *count = 1;
  return musaSuccess;
}

musaError_t MUSARTAPI GetDevicePropertiesStub(musaDeviceProp*, int) {
  return musaSuccess;
}

musaError_t MUSARTAPI DeviceGetAttributeStub(int* value, musaDeviceAttr, int) {
  *value = 0;
  return musaSuccess;
}

musaError_t MUSARTAPI DeviceGetPciBusIdStub(char* pci_bus_id, int length, int) {
  if (length > 0) pci_bus_id[0] = '\0';
  return musaSuccess;
}

musaError_t MUSARTAPI SetDeviceStub(int) { return musaSuccess; }

musaError_t MUSARTAPI DeviceSynchronizeStub() {
  if (g_overlap_state == nullptr) return musaSuccess;

  std::unique_lock<std::mutex> lock(g_overlap_state->mutex);
  ++g_overlap_state->active;
  g_overlap_state->max_active =
      std::max(g_overlap_state->max_active, g_overlap_state->active);
  g_overlap_state->condition.notify_all();
  g_overlap_state->condition.wait_for(lock, std::chrono::seconds(1), [] {
    return g_overlap_state->max_active >= 2;
  });
  --g_overlap_state->active;
  g_overlap_state->condition.notify_all();
  return musaSuccess;
}

musaError_t MUSARTAPI MallocStub(void** ptr, size_t) {
  *ptr = reinterpret_cast<void*>(0x1000);
  return musaSuccess;
}

musaError_t MUSARTAPI FreeStub(void*) { return musaSuccess; }

musaError_t MUSARTAPI MemcpyStub(void*, const void*, size_t, ::musaMemcpyKind) {
  return musaSuccess;
}

musaError_t MUSARTAPI MemcpyAsyncStub(void*, const void*, size_t,
                                      ::musaMemcpyKind, musaStream_t) {
  ++g_memcpy_async_calls;
  return musaSuccess;
}

musaError_t MUSARTAPI MemsetAsyncStub(void*, int, size_t, musaStream_t) {
  return musaSuccess;
}

musaError_t MUSARTAPI HostAllocStub(void** ptr, size_t, unsigned int) {
  *ptr = reinterpret_cast<void*>(0x4000);
  return musaSuccess;
}

musaError_t MUSARTAPI FreeHostStub(void*) { return musaSuccess; }

musaError_t MUSARTAPI StreamCreateStub(musaStream_t* stream) {
  *stream = reinterpret_cast<musaStream_t>(0x2000);
  return musaSuccess;
}

musaError_t MUSARTAPI StreamCreateNullStub(musaStream_t* stream) {
  *stream = nullptr;
  return musaSuccess;
}

musaError_t MUSARTAPI StreamDestroyStub(musaStream_t) { return musaSuccess; }

musaError_t MUSARTAPI StreamSynchronizeStub(musaStream_t) {
  return musaSuccess;
}

musaError_t MUSARTAPI StreamQueryStub(musaStream_t) {
  return musaErrorNotReady;
}

musaError_t MUSARTAPI StreamWaitEventStub(musaStream_t, musaEvent_t,
                                          unsigned int) {
  return musaSuccess;
}

musaError_t MUSARTAPI LaunchHostFuncStub(musaStream_t, musaHostFn_t callback,
                                         void* user_data) {
  callback(user_data);
  return musaSuccess;
}

musaError_t MUSARTAPI EventCreateStub(musaEvent_t* event) {
  *event = reinterpret_cast<musaEvent_t>(0x3000);
  return musaSuccess;
}

musaError_t MUSARTAPI EventCreateNullStub(musaEvent_t* event) {
  *event = nullptr;
  return musaSuccess;
}

musaError_t MUSARTAPI EventCreateWithFlagsStub(musaEvent_t* event,
                                               unsigned int) {
  return EventCreateStub(event);
}

musaError_t MUSARTAPI EventCreateWithFlagsNullStub(musaEvent_t* event,
                                                   unsigned int) {
  return EventCreateNullStub(event);
}

musaError_t MUSARTAPI EventDestroyStub(musaEvent_t) { return musaSuccess; }

musaError_t MUSARTAPI EventRecordStub(musaEvent_t, musaStream_t) {
  return musaSuccess;
}

musaError_t MUSARTAPI EventSynchronizeStub(musaEvent_t) { return musaSuccess; }

musaError_t MUSARTAPI EventElapsedTimeStub(float* milliseconds, musaEvent_t,
                                           musaEvent_t) {
  *milliseconds = 1.25f;
  return musaSuccess;
}

musaError_t MUSARTAPI EventQueryStub(musaEvent_t) { return musaSuccess; }

template <typename Fn>
void* Symbol(Fn fn) {
  return reinterpret_cast<void*>(fn);
}

SymbolMap RequiredSymbols() {
  return {
      {"musaGetDeviceCount", Symbol(&GetDeviceCountStub)},
      {"musaGetDeviceProperties", Symbol(&GetDevicePropertiesStub)},
      {"musaDeviceGetAttribute", Symbol(&DeviceGetAttributeStub)},
      {"musaDeviceGetPCIBusId", Symbol(&DeviceGetPciBusIdStub)},
      {"musaSetDevice", Symbol(&SetDeviceStub)},
      {"musaDeviceSynchronize", Symbol(&DeviceSynchronizeStub)},
      {"musaMalloc", Symbol(&MallocStub)},
      {"musaFree", Symbol(&FreeStub)},
      {"musaMemcpy", Symbol(&MemcpyStub)},
      {"musaMemcpyAsync", Symbol(&MemcpyAsyncStub)},
      {"musaMemsetAsync", Symbol(&MemsetAsyncStub)},
      {"musaStreamCreate", Symbol(&StreamCreateStub)},
      {"musaStreamDestroy", Symbol(&StreamDestroyStub)},
      {"musaStreamSynchronize", Symbol(&StreamSynchronizeStub)},
      {"musaStreamQuery", Symbol(&StreamQueryStub)},
      {"musaStreamWaitEvent", Symbol(&StreamWaitEventStub)},
      {"musaLaunchHostFunc", Symbol(&LaunchHostFuncStub)},
      {"musaEventCreate", Symbol(&EventCreateStub)},
      {"musaEventCreateWithFlags", Symbol(&EventCreateWithFlagsStub)},
      {"musaEventDestroy", Symbol(&EventDestroyStub)},
      {"musaEventRecord", Symbol(&EventRecordStub)},
      {"musaEventSynchronize", Symbol(&EventSynchronizeStub)},
      {"musaEventElapsedTime", Symbol(&EventElapsedTimeStub)},
      {"musaEventQuery", Symbol(&EventQueryStub)},
  };
}

constexpr std::array<absl::string_view, 24> kRequiredSymbols = {
    "musaGetDeviceCount",
    "musaGetDeviceProperties",
    "musaDeviceGetAttribute",
    "musaDeviceGetPCIBusId",
    "musaSetDevice",
    "musaDeviceSynchronize",
    "musaMalloc",
    "musaFree",
    "musaMemcpy",
    "musaMemcpyAsync",
    "musaMemsetAsync",
    "musaStreamCreate",
    "musaStreamDestroy",
    "musaStreamSynchronize",
    "musaStreamQuery",
    "musaStreamWaitEvent",
    "musaLaunchHostFunc",
    "musaEventCreate",
    "musaEventCreateWithFlags",
    "musaEventDestroy",
    "musaEventRecord",
    "musaEventSynchronize",
    "musaEventElapsedTime",
    "musaEventQuery",
};

class FakeSymbolLoader final : public internal::MusaSymbolLoader {
 public:
  explicit FakeSymbolLoader(SymbolMap symbols,
                            absl::Status load_status = absl::OkStatus())
      : symbols_(std::move(symbols)), load_status_(std::move(load_status)) {}

  absl::Status Load() override {
    ++load_calls_;
    return load_status_;
  }

  absl::StatusOr<void*> Resolve(absl::string_view symbol) const override {
    auto it = symbols_.find(std::string(symbol));
    if (it == symbols_.end()) {
      return absl::NotFoundError(std::string(symbol));
    }
    return it->second;
  }

  absl::string_view loaded_path() const override { return "fake-libmusart.so"; }

  int load_calls() const { return load_calls_.load(); }

 private:
  SymbolMap symbols_;
  absl::Status load_status_;
  std::atomic<int> load_calls_ = 0;
};

TEST(MusaRuntimeTest, EveryRequiredSymbolFailsInitializationWhenMissing) {
  for (absl::string_view missing : kRequiredSymbols) {
    SymbolMap symbols = RequiredSymbols();
    symbols.erase(std::string(missing));
    auto loader = std::make_unique<FakeSymbolLoader>(std::move(symbols));
    FakeSymbolLoader* loader_ptr = loader.get();
    std::unique_ptr<MusaRuntime> runtime =
        MusaRuntime::CreateForTesting(std::move(loader));

    absl::Status first = runtime->Init();
    absl::Status second = runtime->Init();

    EXPECT_EQ(first.code(), absl::StatusCode::kFailedPrecondition) << missing;
    EXPECT_NE(first.message().find(missing), absl::string_view::npos)
        << missing;
    EXPECT_EQ(second, first) << missing;
    EXPECT_EQ(loader_ptr->load_calls(), 1) << missing;
  }
}

TEST(MusaRuntimeTest, OptionalSymbolsAreUnimplemented) {
  auto loader = std::make_unique<FakeSymbolLoader>(RequiredSymbols());
  std::unique_ptr<MusaRuntime> runtime =
      MusaRuntime::CreateForTesting(std::move(loader));
  ASSERT_TRUE(runtime->Init().ok());

  EXPECT_EQ(runtime->HostAlloc(1).status().code(),
            absl::StatusCode::kUnimplemented);
  EXPECT_EQ(runtime->FreeHost(reinterpret_cast<void*>(0x1000)).code(),
            absl::StatusCode::kUnimplemented);
  EXPECT_TRUE(runtime->MemsetAsync(nullptr, 0, 0, nullptr).ok());
  size_t free_bytes = 0;
  size_t total_bytes = 0;
  EXPECT_EQ(runtime->MemGetInfo(&free_bytes, &total_bytes).code(),
            absl::StatusCode::kUnimplemented);
  EXPECT_EQ(runtime->RuntimeVersion().status().code(),
            absl::StatusCode::kUnimplemented);

  g_memcpy_async_calls = 0;
  EXPECT_TRUE(runtime
                  ->MemcpyAsync(nullptr, nullptr, 0,
                                MusaMemcpyKind::kDeviceToDevice, nullptr)
                  .ok());
  EXPECT_EQ(g_memcpy_async_calls, 1);
}

TEST(MusaRuntimeTest, NativeHostCallbackAndElapsedTimeAreTyped) {
  auto loader = std::make_unique<FakeSymbolLoader>(RequiredSymbols());
  std::unique_ptr<MusaRuntime> runtime =
      MusaRuntime::CreateForTesting(std::move(loader));
  ASSERT_TRUE(runtime->Init().ok());

  int callback_calls = 0;
  auto callback = [](void* data) { ++*static_cast<int*>(data); };
  EXPECT_TRUE(runtime
                  ->LaunchHostFunc(reinterpret_cast<void*>(0x2000), callback,
                                   &callback_calls)
                  .ok());
  EXPECT_EQ(callback_calls, 1);
  EXPECT_EQ(runtime->LaunchHostFunc(nullptr, nullptr, nullptr).code(),
            absl::StatusCode::kInvalidArgument);

  auto elapsed = runtime->EventElapsedTime(reinterpret_cast<void*>(0x3000),
                                           reinterpret_cast<void*>(0x3001));
  ASSERT_TRUE(elapsed.ok()) << elapsed.status();
  EXPECT_FLOAT_EQ(*elapsed, 1.25f);
  EXPECT_EQ(runtime->EventElapsedTime(nullptr, reinterpret_cast<void*>(0x1))
                .status()
                .code(),
            absl::StatusCode::kInvalidArgument);
}

TEST(MusaRuntimeTest, HostAllocationSymbolsMustBePresentAsAPair) {
  for (absl::string_view one_sided_symbol : {"musaHostAlloc", "musaFreeHost"}) {
    SymbolMap symbols = RequiredSymbols();
    if (one_sided_symbol == "musaHostAlloc") {
      symbols[std::string(one_sided_symbol)] = Symbol(&HostAllocStub);
    } else {
      symbols[std::string(one_sided_symbol)] = Symbol(&FreeHostStub);
    }
    std::unique_ptr<MusaRuntime> runtime = MusaRuntime::CreateForTesting(
        std::make_unique<FakeSymbolLoader>(std::move(symbols)));

    absl::Status status = runtime->Init();
    EXPECT_EQ(status.code(), absl::StatusCode::kFailedPrecondition)
        << one_sided_symbol;
    EXPECT_NE(status.message().find("as a pair"), absl::string_view::npos)
        << one_sided_symbol;
  }
}

TEST(MusaRuntimeTest, RuntimeErrorsUseCanonicalStatusCodes) {
  EXPECT_EQ(ToStatus(musaErrorMemoryAllocation, "musaMalloc").code(),
            absl::StatusCode::kResourceExhausted);
  EXPECT_EQ(ToStatus(musaErrorInvalidValue, "musaMemcpy").code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(ToStatus(musaErrorNoDevice, "musaGetDeviceCount").code(),
            absl::StatusCode::kFailedPrecondition);
  EXPECT_EQ(ToStatus(musaErrorNotSupported, "musaOperation").code(),
            absl::StatusCode::kUnimplemented);
  EXPECT_EQ(ToStatus(musaErrorNotReady, "musaEventQuery").code(),
            absl::StatusCode::kUnavailable);
}

TEST(MusaRuntimeTest, SuccessfulCreationRejectsNullStreamAndEventHandles) {
  for (absl::string_view symbol :
       {"musaStreamCreate", "musaEventCreate", "musaEventCreateWithFlags"}) {
    SymbolMap symbols = RequiredSymbols();
    if (symbol == "musaStreamCreate") {
      symbols[std::string(symbol)] = Symbol(&StreamCreateNullStub);
    } else if (symbol == "musaEventCreate") {
      symbols[std::string(symbol)] = Symbol(&EventCreateNullStub);
    } else {
      symbols[std::string(symbol)] = Symbol(&EventCreateWithFlagsNullStub);
    }
    std::unique_ptr<MusaRuntime> runtime = MusaRuntime::CreateForTesting(
        std::make_unique<FakeSymbolLoader>(std::move(symbols)));
    ASSERT_TRUE(runtime->Init().ok());

    absl::Status status;
    if (symbol == "musaStreamCreate") {
      status = runtime->StreamCreate().status();
    } else {
      status = runtime->EventCreate(symbol == "musaEventCreate").status();
    }
    EXPECT_EQ(status.code(), absl::StatusCode::kInternal) << symbol;
    EXPECT_NE(status.message().find("success with a null"),
              absl::string_view::npos)
        << symbol;
  }
}

TEST(MusaRuntimeTest, InitializationResultIsCachedAcrossThreads) {
  auto loader = std::make_unique<FakeSymbolLoader>(RequiredSymbols());
  FakeSymbolLoader* loader_ptr = loader.get();
  std::unique_ptr<MusaRuntime> runtime =
      MusaRuntime::CreateForTesting(std::move(loader));

  constexpr int kThreadCount = 16;
  std::array<absl::Status, kThreadCount> statuses;
  std::vector<std::thread> threads;
  threads.reserve(kThreadCount);
  for (int i = 0; i < kThreadCount; ++i) {
    threads.emplace_back([&, i] { statuses[i] = runtime->Init(); });
  }
  for (std::thread& thread : threads) thread.join();

  for (const absl::Status& status : statuses) EXPECT_TRUE(status.ok());
  EXPECT_EQ(loader_ptr->load_calls(), 1);
}

TEST(MusaRuntimeTest, InitializationFailureIsCachedExactlyAcrossThreads) {
  const absl::Status expected =
      absl::UnavailableError("injected libmusart load failure");
  auto loader = std::make_unique<FakeSymbolLoader>(RequiredSymbols(), expected);
  FakeSymbolLoader* loader_ptr = loader.get();
  std::unique_ptr<MusaRuntime> runtime =
      MusaRuntime::CreateForTesting(std::move(loader));

  constexpr int kThreadCount = 16;
  std::array<absl::Status, kThreadCount> statuses;
  std::vector<std::thread> threads;
  threads.reserve(kThreadCount);
  for (int i = 0; i < kThreadCount; ++i) {
    threads.emplace_back([&, i] { statuses[i] = runtime->Init(); });
  }
  for (std::thread& thread : threads) thread.join();

  for (const absl::Status& status : statuses) EXPECT_EQ(status, expected);
  EXPECT_EQ(loader_ptr->load_calls(), 1);
}

TEST(MusaRuntimeTest, RuntimeCallsDoNotHoldTheInitializationMutex) {
  auto loader = std::make_unique<FakeSymbolLoader>(RequiredSymbols());
  std::unique_ptr<MusaRuntime> runtime =
      MusaRuntime::CreateForTesting(std::move(loader));
  ASSERT_TRUE(runtime->Init().ok());

  OverlapState overlap;
  g_overlap_state = &overlap;
  std::array<absl::Status, 2> statuses;
  std::thread first([&] { statuses[0] = runtime->DeviceSynchronize(); });
  std::thread second([&] { statuses[1] = runtime->DeviceSynchronize(); });
  first.join();
  second.join();
  g_overlap_state = nullptr;

  EXPECT_TRUE(statuses[0].ok());
  EXPECT_TRUE(statuses[1].ok());
  EXPECT_EQ(overlap.max_active, 2);
}

}  // namespace
}  // namespace stream_executor::musa
