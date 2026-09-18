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

#include <string>

#include "absl/log/check.h"
#include "absl/strings/string_view.h"
#include "xla/stream_executor/abi/runtime_abi_version_manager.h"
#include "xla/stream_executor/musa/musa_platform_id.h"
#include "xla/stream_executor/musa/musa_runtime_abi_version.h"

namespace stream_executor::musa {

static bool InitModule() {
  CHECK_OK(
      RuntimeAbiVersionManager::GetInstance().RegisterRuntimeAbiVersionFactory(
          std::string(kMusaPlatformId->ToName()), [](absl::string_view proto) {
            return MusaRuntimeAbiVersion::FromSerializedProto(proto);
          }));
  return true;
}
static bool module_initialized = InitModule();

}  // namespace stream_executor::musa
