/* Copyright 2019 The OpenXLA Authors.

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

#include "xla/pjrt/worker_thread.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <string>
#include <utility>

#include "absl/functional/any_invocable.h"
#include "absl/log/check.h"
#include "absl/synchronization/mutex.h"
#include "absl/synchronization/notification.h"
#include "xla/tsl/platform/env.h"

namespace xla {

WorkerThread::WorkerThread(tsl::Env* env, const std::string& name)
    : WorkerThread(env, tsl::ThreadOptions(), name,
                   std::chrono::microseconds::zero()) {}

WorkerThread::WorkerThread(tsl::Env* env, const tsl::ThreadOptions& options,
                           const std::string& name)
    : WorkerThread(env, options, name, std::chrono::microseconds::zero()) {}

WorkerThread::WorkerThread(
    tsl::Env* env, const tsl::ThreadOptions& options, const std::string& name,
    std::chrono::microseconds idle_spin_duration)
    : idle_spin_duration_(idle_spin_duration) {
  thread_.reset(env->StartThread(options, name, [this]() { WorkLoop(); }));
}

WorkerThread::~WorkerThread() {
  absl::MutexLock lock(mu_);
  work_queue_.push(nullptr);
  work_available_hint_.store(true, std::memory_order_release);
}

void WorkerThread::Schedule(absl::AnyInvocable<void() &&> fn) {
  CHECK(fn != nullptr);
  absl::MutexLock lock(mu_);
  work_queue_.push(std::move(fn));
  work_available_hint_.store(true, std::memory_order_release);
}

void WorkerThread::Drain() {
  absl::Notification done;
  // Schedule a sentinel closure after all currently-queued work. When the
  // worker thread executes it, we know every prior closure has completed.
  Schedule([&done]() { done.Notify(); });
  done.WaitForNotification();
}

bool WorkerThread::WorkAvailable() { return !work_queue_.empty(); }

void WorkerThread::SpinForWork() {
  if (idle_spin_duration_ <= std::chrono::microseconds::zero()) {
    return;
  }

  const auto deadline = std::chrono::steady_clock::now() + idle_spin_duration_;
  uint32_t iterations = 0;
  while (!work_available_hint_.load(std::memory_order_acquire)) {
#if defined(__x86_64__) || defined(__i386__)
    __builtin_ia32_pause();
#elif defined(__aarch64__) || defined(__arm__)
    asm volatile("yield" ::: "memory");
#else
    std::atomic_signal_fence(std::memory_order_seq_cst);
#endif
    // Reading the clock on every iteration is relatively expensive.
    if ((++iterations & 63) == 0 &&
        std::chrono::steady_clock::now() >= deadline) {
      return;
    }
  }
}

void WorkerThread::WorkLoop() {
  absl::MutexLock lock(mu_);
  bool has_executed_work = false;
  while (true) {
    if (has_executed_work &&
        idle_spin_duration_ > std::chrono::microseconds::zero() &&
        work_queue_.empty()) {
      mu_.unlock();
      SpinForWork();
      mu_.lock();
    }

    // Rechecking the queue under mu_ prevents a lost wakeup if work is
    // scheduled exactly when the adaptive spin expires.
    mu_.Await(absl::Condition(this, &WorkerThread::WorkAvailable));
    {
      // We must be careful to call fn's dtor when the lock is unlocked.
      absl::AnyInvocable<void() &&> fn = std::move(work_queue_.front());
      work_queue_.pop();
      work_available_hint_.store(!work_queue_.empty(),
                                 std::memory_order_release);
      if (!fn) {
        return;
      }
      is_running_ = true;
      mu_.unlock();
      std::move(fn)();
    }
    mu_.lock();
    is_running_ = false;
    has_executed_work = true;
  }
}

}  // namespace xla
