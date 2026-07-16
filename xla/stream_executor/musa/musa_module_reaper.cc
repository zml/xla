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

#include "xla/stream_executor/musa/musa_module_reaper.h"

#include <algorithm>
#include <iterator>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/str_format.h"
#include "absl/synchronization/mutex.h"
#include "xla/stream_executor/musa/musa_module.h"
#include "xla/tsl/platform/env.h"

namespace stream_executor::musa {
namespace {

void RetainOrphansForProcessLifetime(
    std::vector<std::shared_ptr<MusaModule>> modules) {
  if (modules.empty()) return;
  static auto* mutex = new absl::Mutex;
  static auto* retained =
      new std::vector<std::vector<std::shared_ptr<MusaModule>>>;
  absl::MutexLock lock(mutex);
  retained->push_back(std::move(modules));
}

}  // namespace

MusaModuleReaper::ModuleUse::ModuleUse(MusaModuleReaper* reaper,
                                       std::shared_ptr<MusaModule> module)
    : reaper_(reaper), module_(std::move(module)) {
  CHECK(reaper_ != nullptr);
  CHECK(module_ != nullptr);
}

void MusaModuleReaper::ModuleUse::Complete() {
  if (claimed_.exchange(true, std::memory_order_acq_rel)) return;
  reaper_->RetireModule(std::move(module_));
}

void MusaModuleReaper::ModuleUse::Orphan() {
  if (claimed_.exchange(true, std::memory_order_acq_rel)) return;
  reaper_->Orphan(std::move(module_));
}

MusaModuleReaper::MusaModuleReaper(int device_ordinal) {
  worker_ = std::unique_ptr<tsl::Thread>(tsl::Env::Default()->StartThread(
      tsl::ThreadOptions(),
      absl::StrFormat("musa_module_reaper_%d", device_ordinal),
      [this] { WorkerLoop(); }));
}

MusaModuleReaper::~MusaModuleReaper() {
  {
    absl::MutexLock lock(&mutex_);
    stopping_ = true;
  }
  worker_.reset();

  std::vector<std::shared_ptr<MusaModule>> orphans;
  {
    absl::MutexLock lock(&mutex_);
    orphans = std::move(orphans_);
  }
  if (!orphans.empty()) {
    LOG(ERROR) << "Retaining " << orphans.size()
               << " MUSA module(s) for process lifetime because their "
                  "device completion was not established";
    RetainOrphansForProcessLifetime(std::move(orphans));
  }
}

std::shared_ptr<MusaModuleReaper::ModuleUse> MusaModuleReaper::Track(
    std::shared_ptr<MusaModule> module) {
  Observe(module);
  return std::shared_ptr<ModuleUse>(new ModuleUse(this, std::move(module)));
}

void MusaModuleReaper::Observe(const std::shared_ptr<MusaModule>& module) {
  if (module == nullptr) return;
  absl::MutexLock lock(&mutex_);
  observed_.erase(std::remove_if(observed_.begin(), observed_.end(),
                                 [](const std::weak_ptr<MusaModule>& observed) {
                                   return observed.expired();
                                 }),
                  observed_.end());
  auto duplicate =
      std::find_if(observed_.begin(), observed_.end(),
                   [&module](const std::weak_ptr<MusaModule>& observed) {
                     std::shared_ptr<MusaModule> live = observed.lock();
                     return live != nullptr && live.get() == module.get();
                   });
  if (duplicate == observed_.end()) observed_.push_back(module);
}

void MusaModuleReaper::RetireModule(std::shared_ptr<MusaModule> module) {
  if (module == nullptr) return;
  absl::MutexLock lock(&mutex_);
  if (stopping_) {
    AddOrphanLocked(std::move(module));
    return;
  }
  retired_.push_back(std::move(module));
}

void MusaModuleReaper::Orphan(std::shared_ptr<MusaModule> module) {
  if (module == nullptr) return;
  absl::MutexLock lock(&mutex_);
  AddOrphanLocked(std::move(module));
}

void MusaModuleReaper::AddOrphanLocked(std::shared_ptr<MusaModule> module) {
  auto duplicate =
      std::find_if(orphans_.begin(), orphans_.end(),
                   [&module](const std::shared_ptr<MusaModule>& orphan) {
                     return orphan.get() == module.get();
                   });
  if (duplicate == orphans_.end()) orphans_.push_back(std::move(module));
}

void MusaModuleReaper::ReleaseOrphansAfterSynchronization() {
  absl::MutexLock lock(&mutex_);
  if (stopping_) return;
  retired_.insert(retired_.end(), std::make_move_iterator(orphans_.begin()),
                  std::make_move_iterator(orphans_.end()));
  orphans_.clear();
}

bool MusaModuleReaper::HasOrphans() const {
  absl::MutexLock lock(&mutex_);
  return !orphans_.empty();
}

std::vector<std::shared_ptr<MusaModule>>
MusaModuleReaper::TakeModulesForProcessLifetime() {
  absl::MutexLock lock(&mutex_);
  std::vector<std::shared_ptr<MusaModule>> modules = std::move(orphans_);
  for (const std::weak_ptr<MusaModule>& observed : observed_) {
    std::shared_ptr<MusaModule> live = observed.lock();
    if (live == nullptr) continue;
    auto duplicate =
        std::find_if(modules.begin(), modules.end(),
                     [&live](const std::shared_ptr<MusaModule>& module) {
                       return module.get() == live.get();
                     });
    if (duplicate == modules.end()) modules.push_back(std::move(live));
  }
  observed_.clear();
  return modules;
}

void MusaModuleReaper::WaitUntilIdleForTesting() {
  absl::MutexLock lock(&mutex_);
  mutex_.Await(absl::Condition(this, &MusaModuleReaper::IsIdle));
}

size_t MusaModuleReaper::OrphanCountForTesting() const {
  absl::MutexLock lock(&mutex_);
  return orphans_.size();
}

bool MusaModuleReaper::HasWorkOrStopping() const {
  return stopping_ || !retired_.empty();
}

bool MusaModuleReaper::IsIdle() const {
  return retired_.empty() && active_batches_ == 0;
}

void MusaModuleReaper::WorkerLoop() {
  while (true) {
    std::vector<std::shared_ptr<MusaModule>> batch;
    {
      absl::MutexLock lock(&mutex_);
      mutex_.Await(absl::Condition(this, &MusaModuleReaper::HasWorkOrStopping));
      if (retired_.empty() && stopping_) return;
      batch.swap(retired_);
      ++active_batches_;
    }

    // Clearing this batch may unload native modules. It intentionally runs on
    // this ordinary worker, never on a MUSA host-callback thread.
    batch.clear();

    {
      absl::MutexLock lock(&mutex_);
      CHECK_GT(active_batches_, 0);
      --active_batches_;
    }
  }
}

}  // namespace stream_executor::musa
