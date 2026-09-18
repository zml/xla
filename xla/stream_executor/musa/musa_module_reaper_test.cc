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

#include <atomic>
#include <cstdint>
#include <memory>
#include <thread>

#include <gtest/gtest.h>
#include "absl/synchronization/notification.h"
#include "absl/time/time.h"
#include "xla/stream_executor/musa/musa_module.h"

namespace stream_executor::musa {
namespace {

std::shared_ptr<MusaModule> MakeFakeModule(absl::Notification* released,
                                           std::atomic<int>* release_count,
                                           std::thread::id* release_thread) {
  return std::shared_ptr<MusaModule>(
      reinterpret_cast<MusaModule*>(uintptr_t{0x1234}),
      [released, release_count, release_thread](MusaModule*) {
        *release_thread = std::this_thread::get_id();
        ++*release_count;
        released->Notify();
      });
}

TEST(MusaModuleReaperTest, CompleteReleasesOnWorkerThread) {
  MusaModuleReaper reaper(/*device_ordinal=*/0);
  absl::Notification released;
  std::atomic<int> release_count = 0;
  std::thread::id release_thread;
  const std::thread::id caller_thread = std::this_thread::get_id();
  std::shared_ptr<MusaModule> module =
      MakeFakeModule(&released, &release_count, &release_thread);

  std::shared_ptr<MusaModuleReaper::ModuleUse> use = reaper.Track(module);
  module.reset();
  use->Complete();
  use.reset();

  ASSERT_TRUE(released.WaitForNotificationWithTimeout(absl::Seconds(5)));
  EXPECT_EQ(release_count, 1);
  EXPECT_NE(release_thread, caller_thread);
}

TEST(MusaModuleReaperTest, CompletionTicketIsClaimedExactlyOnce) {
  MusaModuleReaper reaper(/*device_ordinal=*/0);
  absl::Notification released;
  std::atomic<int> release_count = 0;
  std::thread::id release_thread;
  std::shared_ptr<MusaModule> module =
      MakeFakeModule(&released, &release_count, &release_thread);

  std::shared_ptr<MusaModuleReaper::ModuleUse> use = reaper.Track(module);
  module.reset();
  use->Orphan();
  use->Complete();
  use->Orphan();
  use.reset();

  EXPECT_EQ(reaper.OrphanCountForTesting(), 1);
  EXPECT_EQ(release_count, 0);
  reaper.ReleaseOrphansAfterSynchronization();
  ASSERT_TRUE(released.WaitForNotificationWithTimeout(absl::Seconds(5)));
  EXPECT_EQ(release_count, 1);
}

TEST(MusaModuleReaperTest, DeduplicatesOrphansUntilSynchronization) {
  MusaModuleReaper reaper(/*device_ordinal=*/0);
  absl::Notification released;
  std::atomic<int> release_count = 0;
  std::thread::id release_thread;
  std::shared_ptr<MusaModule> module =
      MakeFakeModule(&released, &release_count, &release_thread);

  reaper.Orphan(module);
  reaper.Orphan(module);
  module.reset();

  EXPECT_EQ(reaper.OrphanCountForTesting(), 1);
  EXPECT_EQ(release_count, 0);
  reaper.ReleaseOrphansAfterSynchronization();
  ASSERT_TRUE(released.WaitForNotificationWithTimeout(absl::Seconds(5)));
  EXPECT_EQ(release_count, 1);
}

TEST(MusaModuleReaperTest, FailedSynchronizationTransferDoesNotRelease) {
  MusaModuleReaper reaper(/*device_ordinal=*/0);
  absl::Notification released;
  std::atomic<int> release_count = 0;
  std::thread::id release_thread;
  std::shared_ptr<MusaModule> module =
      MakeFakeModule(&released, &release_count, &release_thread);

  reaper.Orphan(module);
  module.reset();
  std::vector<std::shared_ptr<MusaModule>> retained =
      reaper.TakeModulesForProcessLifetime();

  ASSERT_EQ(retained.size(), 1);
  EXPECT_EQ(reaper.OrphanCountForTesting(), 0);
  EXPECT_EQ(release_count, 0);
  retained.clear();
  ASSERT_TRUE(released.WaitForNotificationWithTimeout(absl::Seconds(5)));
  EXPECT_EQ(release_count, 1);
}

TEST(MusaModuleReaperTest, PromotesWeakObservationOnlyOnSyncFailure) {
  MusaModuleReaper reaper(/*device_ordinal=*/0);
  absl::Notification released;
  std::atomic<int> release_count = 0;
  std::thread::id release_thread;
  std::shared_ptr<MusaModule> module =
      MakeFakeModule(&released, &release_count, &release_thread);

  reaper.Observe(module);
  std::vector<std::shared_ptr<MusaModule>> retained =
      reaper.TakeModulesForProcessLifetime();
  module.reset();

  ASSERT_EQ(retained.size(), 1);
  EXPECT_EQ(release_count, 0);
  retained.clear();
  ASSERT_TRUE(released.WaitForNotificationWithTimeout(absl::Seconds(5)));
  EXPECT_EQ(release_count, 1);
}

}  // namespace
}  // namespace stream_executor::musa
