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

#include "xla/stream_executor/musa/musa_graph_arguments.h"

#include <cstdint>
#include <memory>

#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/kernel_args.h"
#include "xla/stream_executor/kernel_args_packed_vector.h"

namespace stream_executor::musa {
namespace {

TEST(MusaGraphArgumentsTest, DeepCopyOutlivesSourceArguments) {
  std::unique_ptr<KernelArgsPackedVector> retained;
  {
    KernelArgsPackedArray source(/*num_args=*/2);
    source.add_argument(DeviceAddressBase(reinterpret_cast<void*>(0x123400)));
    source.add_argument(int64_t{42});
    auto cloned = CloneMusaGraphKernelArguments(source);
    ASSERT_TRUE(cloned.ok()) << cloned.status();
    retained = std::move(*cloned);
  }

  ASSERT_EQ(retained->argument_addresses().size(), 2);
  EXPECT_EQ(*static_cast<void* const*>(retained->argument_addresses()[0]),
            reinterpret_cast<void*>(0x123400));
  EXPECT_EQ(*static_cast<const int64_t*>(retained->argument_addresses()[1]),
            42);
}

TEST(MusaGraphArgumentsTest, RejectsPackedArgumentsWithoutSizeMetadata) {
  KernelArgsPackedTuple<int64_t> source(int64_t{42},
                                        /*shared_memory_bytes=*/0);
  EXPECT_EQ(CloneMusaGraphKernelArguments(source).status().code(),
            absl::StatusCode::kUnimplemented);
}

TEST(MusaGraphArgumentsTest, ClonesSerializablePackedVector) {
  KernelArgsPackedVector source({{'a', 'b'}, {'c', 'd', 'e'}},
                                /*shared_memory_bytes=*/16);
  auto cloned = CloneMusaGraphKernelArguments(source);
  ASSERT_TRUE(cloned.ok()) << cloned.status();
  EXPECT_EQ((*cloned)->argument_storage(), source.argument_storage());
  EXPECT_EQ((*cloned)->number_of_shared_bytes(), 16);
}

}  // namespace
}  // namespace stream_executor::musa
