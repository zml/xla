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

#include "xla/stream_executor/musa/musa_compute_capability.h"

#include <gtest/gtest.h>
#include "absl/status/statusor.h"

namespace stream_executor {
namespace {

TEST(MusaComputeCapabilityTest, ProtoRoundTripPreservesDistinctWidths) {
  MusaComputeCapability capability("mp_21", 2, 1,
                                   /*hardware_warp_size=*/128,
                                   /*logical_subgroup_size=*/32);
  absl::StatusOr<MusaComputeCapability> from_proto =
      MusaComputeCapability::FromProto(capability.ToProto());
  ASSERT_TRUE(from_proto.ok()) << from_proto.status();
  EXPECT_EQ(*from_proto, capability);
  EXPECT_EQ(capability.architecture(), "mp_21");
  EXPECT_EQ(capability.major(), 2);
  EXPECT_EQ(capability.minor(), 1);
  EXPECT_EQ(capability.hardware_warp_size(), 128);
  EXPECT_EQ(capability.logical_subgroup_size(), 32);
}

TEST(MusaComputeCapabilityTest, LegacyProtoRemainsBackwardCompatible) {
  MusaComputeCapabilityProto proto;
  proto.set_architecture("mp_21");
  absl::StatusOr<MusaComputeCapability> capability =
      MusaComputeCapability::FromProto(proto);
  ASSERT_TRUE(capability.ok()) << capability.status();
  EXPECT_EQ(capability->architecture(), "mp_21");
  EXPECT_EQ(capability->major(), 0);
  EXPECT_EQ(capability->minor(), 0);
  EXPECT_EQ(capability->hardware_warp_size(), 0);
  EXPECT_EQ(capability->logical_subgroup_size(), 0);
}

TEST(MusaComputeCapabilityTest, RejectsPartialExtendedProto) {
  MusaComputeCapabilityProto proto;
  proto.set_architecture("mp_21");
  proto.set_major(2);
  proto.set_minor(1);
  EXPECT_FALSE(MusaComputeCapability::FromProto(proto).ok());
}

TEST(MusaComputeCapabilityTest, RejectsInvalidExtendedValues) {
  MusaComputeCapabilityProto proto;
  proto.set_architecture("mp_21");
  proto.set_major(2);
  proto.set_minor(1);
  proto.set_hardware_warp_size(128);
  proto.set_logical_subgroup_size(32);

  MusaComputeCapabilityProto malformed = proto;
  malformed.set_major(-2);
  EXPECT_FALSE(MusaComputeCapability::FromProto(malformed).ok());

  malformed = proto;
  malformed.set_minor(-1);
  EXPECT_FALSE(MusaComputeCapability::FromProto(malformed).ok());

  malformed = proto;
  malformed.set_hardware_warp_size(0);
  EXPECT_FALSE(MusaComputeCapability::FromProto(malformed).ok());

  malformed = proto;
  malformed.set_logical_subgroup_size(0);
  EXPECT_FALSE(MusaComputeCapability::FromProto(malformed).ok());
}

TEST(MusaComputeCapabilityTest, RejectsArchitectureMismatch) {
  MusaComputeCapabilityProto proto;
  proto.set_architecture("mp_21");
  proto.set_major(9);
  proto.set_minor(9);
  proto.set_hardware_warp_size(128);
  proto.set_logical_subgroup_size(32);
  EXPECT_FALSE(MusaComputeCapability::FromProto(proto).ok());
}

TEST(MusaComputeCapabilityTest, RejectsInconsistentWidths) {
  MusaComputeCapabilityProto proto;
  proto.set_architecture("mp_21");
  proto.set_major(2);
  proto.set_minor(1);
  proto.set_hardware_warp_size(128);
  proto.set_logical_subgroup_size(48);
  EXPECT_FALSE(MusaComputeCapability::FromProto(proto).ok());
}

}  // namespace
}  // namespace stream_executor
