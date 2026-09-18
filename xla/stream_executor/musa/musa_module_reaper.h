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

#ifndef XLA_STREAM_EXECUTOR_MUSA_MUSA_MODULE_REAPER_H_
#define XLA_STREAM_EXECUTOR_MUSA_MUSA_MODULE_REAPER_H_

#include <atomic>
#include <cstddef>
#include <memory>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/synchronization/mutex.h"

namespace tsl {
class Thread;
}

namespace stream_executor::musa {

class MusaModule;

// Releases modules on an ordinary host worker after stream-ordered completion.
// MUSA host callbacks are not allowed to call MUSA APIs, so they may only move
// a module into this reaper. Uses for which completion is ambiguous are kept as
// deduplicated orphans until a successful context synchronization.
class MusaModuleReaper final {
 public:
  class ModuleUse final {
   public:
    ModuleUse(const ModuleUse&) = delete;
    ModuleUse& operator=(const ModuleUse&) = delete;

    // Exactly one of Complete and Orphan takes ownership of the module. Calls
    // after the first are no-ops, allowing callback success and error paths to
    // race without releasing or retaining twice.
    void Complete();
    void Orphan();

   private:
    friend class MusaModuleReaper;

    ModuleUse(MusaModuleReaper* reaper, std::shared_ptr<MusaModule> module);

    MusaModuleReaper* const reaper_;
    std::shared_ptr<MusaModule> module_;
    std::atomic<bool> claimed_{false};
  };

  explicit MusaModuleReaper(int device_ordinal);
  ~MusaModuleReaper();

  MusaModuleReaper(const MusaModuleReaper&) = delete;
  MusaModuleReaper& operator=(const MusaModuleReaper&) = delete;

  std::shared_ptr<ModuleUse> Track(std::shared_ptr<MusaModule> module);
  // Records a weak observation without extending normal module lifetime. Live
  // observations are promoted only if executor teardown synchronization fails.
  void Observe(const std::shared_ptr<MusaModule>& module);
  // Queues an unconditionally safe module reference for release on the worker.
  // This is used by MusaKernel destruction, which can itself occur on a MUSA
  // callback thread even for a never-launched kernel.
  void RetireModule(std::shared_ptr<MusaModule> module);
  void Orphan(std::shared_ptr<MusaModule> module);

  // May only be called after all activity in the owning MUSA context has been
  // synchronized successfully. Orphans are then safe to release on the worker.
  void ReleaseOrphansAfterSynchronization();

  bool HasOrphans() const;

  // Promotes live observations and transfers them with all orphans after
  // synchronization failure. The executor retains the result with its cache.
  std::vector<std::shared_ptr<MusaModule>> TakeModulesForProcessLifetime();

  // Exposed for deterministic hermetic tests. Production teardown drains the
  // worker in the destructor.
  void WaitUntilIdleForTesting();
  size_t OrphanCountForTesting() const;

 private:
  void WorkerLoop();
  bool HasWorkOrStopping() const ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_);
  bool IsIdle() const ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_);
  void AddOrphanLocked(std::shared_ptr<MusaModule> module)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_);

  mutable absl::Mutex mutex_;
  std::vector<std::shared_ptr<MusaModule>> retired_ ABSL_GUARDED_BY(mutex_);
  std::vector<std::shared_ptr<MusaModule>> orphans_ ABSL_GUARDED_BY(mutex_);
  std::vector<std::weak_ptr<MusaModule>> observed_ ABSL_GUARDED_BY(mutex_);
  size_t active_batches_ ABSL_GUARDED_BY(mutex_) = 0;
  bool stopping_ ABSL_GUARDED_BY(mutex_) = false;
  std::unique_ptr<tsl::Thread> worker_;
};

}  // namespace stream_executor::musa

#endif  // XLA_STREAM_EXECUTOR_MUSA_MUSA_MODULE_REAPER_H_
