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

#include "xla/stream_executor/musa/musa_mubin.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"

namespace stream_executor::musa {
namespace {

using ::absl_testing::StatusIs;

constexpr size_t kElfHeaderSize = 64;
constexpr size_t kProgramHeaderSize = 56;
constexpr size_t kLoadHeader = kElfHeaderSize;
constexpr size_t kNoteHeader = kLoadHeader + kProgramHeaderSize;
constexpr size_t kLoadOffset = 192;
constexpr size_t kNoteOffset = 208;

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

std::vector<uint8_t> MakeValidMubin() {
  std::vector<uint8_t> bytes(240);
  bytes[0] = 0x7f;
  bytes[1] = 'E';
  bytes[2] = 'L';
  bytes[3] = 'F';
  bytes[4] = 2;  // ELFCLASS64.
  bytes[5] = 1;  // ELFDATA2LSB.
  bytes[6] = 1;  // EV_CURRENT.
  bytes[8] = 7;  // An arbitrary ELF ABI version preserved as metadata.
  WriteU16(bytes, 16, 3);    // ET_DYN.
  WriteU16(bytes, 18, 253);  // Qualified MUBIN e_machine.
  WriteU32(bytes, 20, 1);    // EV_CURRENT.
  WriteU64(bytes, 32, kElfHeaderSize);
  WriteU16(bytes, 52, kElfHeaderSize);
  WriteU16(bytes, 54, kProgramHeaderSize);
  WriteU16(bytes, 56, 2);

  WriteU32(bytes, kLoadHeader, 1);      // PT_LOAD.
  WriteU32(bytes, kLoadHeader + 4, 5);  // PF_R | PF_X.
  WriteU64(bytes, kLoadHeader + 8, kLoadOffset);
  WriteU64(bytes, kLoadHeader + 32, 8);
  WriteU64(bytes, kLoadHeader + 40, 8);

  WriteU32(bytes, kNoteHeader, 4);      // PT_NOTE.
  WriteU32(bytes, kNoteHeader + 4, 4);  // PF_R.
  WriteU64(bytes, kNoteHeader + 8, kNoteOffset);
  WriteU64(bytes, kNoteHeader + 32, 24);
  WriteU64(bytes, kNoteHeader + 40, 24);

  WriteU32(bytes, kNoteOffset, 6);      // "MTGPU\0".
  WriteU32(bytes, kNoteOffset + 4, 4);  // Four opaque descriptor bytes.
  WriteU32(bytes, kNoteOffset + 8, 0x40);
  const std::string owner = "MTGPU";
  for (size_t i = 0; i < owner.size(); ++i) {
    bytes[kNoteOffset + 12 + i] = owner[i];
  }
  bytes[kNoteOffset + 20] = 0xde;
  bytes[kNoteOffset + 21] = 0xad;
  bytes[kNoteOffset + 22] = 0xbe;
  bytes[kNoteOffset + 23] = 0xef;
  return bytes;
}

void ExpectInvalid(const std::vector<uint8_t>& bytes) {
  EXPECT_THAT(ValidateMusaMubin(bytes),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(MusaMubinTest, AcceptsQualifiedMubinAndReturnsCheckedMetadata) {
  ASSERT_EQ(kMubinLoaderAbiVersion, 1);
  auto metadata = ValidateMusaMubin(MakeValidMubin());
  ASSERT_TRUE(metadata.ok()) << metadata.status();
  EXPECT_EQ(metadata->elf_abi_version, 7);
  EXPECT_EQ(metadata->elf_machine, 253);
  EXPECT_EQ(metadata->vendor_note_type, 0x40);
}

TEST(MusaMubinTest, RejectsEmptyAndTruncatedArtifacts) {
  ExpectInvalid({});
  ExpectInvalid(std::vector<uint8_t>(8));
  auto bytes = MakeValidMubin();
  bytes.resize(63);
  ExpectInvalid(bytes);
}

TEST(MusaMubinTest, RejectsWrongElfIdentity) {
  for (const auto [offset, value] :
       {std::pair<size_t, uint8_t>{0, 0}, {4, 1}, {5, 2}, {6, 0}}) {
    auto bytes = MakeValidMubin();
    bytes[offset] = value;
    ExpectInvalid(bytes);
  }
}

TEST(MusaMubinTest, RejectsWrongTypeMachineAndVersion) {
  auto bytes = MakeValidMubin();
  WriteU16(bytes, 16, 1);
  ExpectInvalid(bytes);

  bytes = MakeValidMubin();
  WriteU16(bytes, 18, 62);
  ExpectInvalid(bytes);

  bytes = MakeValidMubin();
  WriteU32(bytes, 20, 0);
  ExpectInvalid(bytes);
}

TEST(MusaMubinTest, RejectsMalformedElfAndProgramHeaderSizes) {
  auto bytes = MakeValidMubin();
  WriteU16(bytes, 52, 63);
  ExpectInvalid(bytes);

  bytes = MakeValidMubin();
  WriteU16(bytes, 54, 55);
  ExpectInvalid(bytes);

  bytes = MakeValidMubin();
  WriteU16(bytes, 56, 0);
  ExpectInvalid(bytes);
}

TEST(MusaMubinTest, RejectsOutOfBoundsProgramHeaderTable) {
  auto bytes = MakeValidMubin();
  WriteU64(bytes, 32, std::numeric_limits<uint64_t>::max());
  ExpectInvalid(bytes);

  bytes = MakeValidMubin();
  WriteU64(bytes, 32, bytes.size() - kProgramHeaderSize + 1);
  ExpectInvalid(bytes);
}

TEST(MusaMubinTest, RejectsOutOfBoundsProgramSegments) {
  auto bytes = MakeValidMubin();
  WriteU64(bytes, kLoadHeader + 8, std::numeric_limits<uint64_t>::max());
  ExpectInvalid(bytes);

  bytes = MakeValidMubin();
  WriteU64(bytes, kNoteHeader + 32, std::numeric_limits<uint64_t>::max());
  ExpectInvalid(bytes);
}

TEST(MusaMubinTest, RejectsLoadWhoseFileSizeExceedsMemorySize) {
  auto bytes = MakeValidMubin();
  WriteU64(bytes, kLoadHeader + 32, 9);
  ExpectInvalid(bytes);
}

TEST(MusaMubinTest, RequiresExecutableLoadSegment) {
  auto bytes = MakeValidMubin();
  WriteU32(bytes, kLoadHeader + 4, 4);  // PF_R, without PF_X.
  ExpectInvalid(bytes);
}

TEST(MusaMubinTest, RequiresMtgpuNoteWithQualifiedType) {
  auto bytes = MakeValidMubin();
  bytes[kNoteOffset + 12] = 'X';
  ExpectInvalid(bytes);

  bytes = MakeValidMubin();
  WriteU32(bytes, kNoteOffset + 8, 0x41);
  ExpectInvalid(bytes);

  bytes = MakeValidMubin();
  WriteU32(bytes, kNoteHeader, 1);  // Replace PT_NOTE with PT_LOAD.
  ExpectInvalid(bytes);
}

TEST(MusaMubinTest, AcceptsMtgpuNoteOwnerWithoutTerminatingNull) {
  auto bytes = MakeValidMubin();
  WriteU32(bytes, kNoteOffset, 5);
  EXPECT_TRUE(ValidateMusaMubin(bytes).ok());
}

TEST(MusaMubinTest, RejectsTruncatedAndOverflowingNotes) {
  auto bytes = MakeValidMubin();
  WriteU64(bytes, kNoteHeader + 32, 11);
  ExpectInvalid(bytes);

  bytes = MakeValidMubin();
  WriteU32(bytes, kNoteOffset, std::numeric_limits<uint32_t>::max());
  ExpectInvalid(bytes);

  bytes = MakeValidMubin();
  WriteU32(bytes, kNoteOffset + 4, std::numeric_limits<uint32_t>::max());
  ExpectInvalid(bytes);
}

TEST(MusaMubinTest, RejectsTrailingPartialNoteHeader) {
  auto bytes = MakeValidMubin();
  WriteU64(bytes, kNoteHeader + 32, 25);
  ExpectInvalid(bytes);
}

}  // namespace
}  // namespace stream_executor::musa
