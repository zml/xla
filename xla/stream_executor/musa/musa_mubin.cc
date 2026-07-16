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

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>

#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/stream_executor/musa/musa_target_contract.h"

namespace stream_executor::musa {
namespace {

// ELF constants and record offsets are spelled out here so validation does not
// depend on the host compiler's ELF structs, byte order, or target naming.
constexpr size_t kElfIdentificationSize = 16;
constexpr size_t kElf64HeaderSize = 64;
constexpr size_t kElf64ProgramHeaderSize = 56;
constexpr size_t kElfNoteHeaderSize = 12;

constexpr size_t kElfClassOffset = 4;
constexpr size_t kElfDataOffset = 5;
constexpr size_t kElfIdentVersionOffset = 6;
constexpr size_t kElfAbiVersionOffset = 8;
constexpr size_t kElfTypeOffset = 16;
constexpr size_t kElfMachineOffset = 18;
constexpr size_t kElfVersionOffset = 20;
constexpr size_t kElfProgramHeaderOffset = 32;
constexpr size_t kElfHeaderSizeOffset = 52;
constexpr size_t kElfProgramHeaderEntrySizeOffset = 54;
constexpr size_t kElfProgramHeaderCountOffset = 56;

constexpr size_t kProgramHeaderTypeOffset = 0;
constexpr size_t kProgramHeaderFlagsOffset = 4;
constexpr size_t kProgramHeaderFileOffset = 8;
constexpr size_t kProgramHeaderFileSizeOffset = 32;
constexpr size_t kProgramHeaderMemorySizeOffset = 40;

constexpr std::array<uint8_t, 4> kElfMagic = {0x7f, 'E', 'L', 'F'};
constexpr uint8_t kElfClass64 = 2;
constexpr uint8_t kElfDataLittleEndian = 1;
constexpr uint32_t kElfCurrentVersion = 1;
constexpr uint16_t kElfTypeDynamic = 3;
constexpr uint32_t kProgramHeaderTypeLoad = 1;
constexpr uint32_t kProgramHeaderTypeNote = 4;
constexpr uint32_t kProgramHeaderFlagExecutable = 1;
constexpr uint32_t kMtgpuNoteType = 0x40;
constexpr std::array<uint8_t, 5> kMtgpuNoteOwner = {'M', 'T', 'G', 'P', 'U'};

absl::Status InvalidMubin(absl::string_view reason) {
  return absl::InvalidArgumentError(
      absl::StrFormat("invalid MUBIN: %s", reason));
}

uint16_t ReadU16(absl::Span<const uint8_t> bytes, size_t offset) {
  return static_cast<uint16_t>(bytes[offset]) |
         (static_cast<uint16_t>(bytes[offset + 1]) << 8);
}

uint32_t ReadU32(absl::Span<const uint8_t> bytes, size_t offset) {
  return static_cast<uint32_t>(bytes[offset]) |
         (static_cast<uint32_t>(bytes[offset + 1]) << 8) |
         (static_cast<uint32_t>(bytes[offset + 2]) << 16) |
         (static_cast<uint32_t>(bytes[offset + 3]) << 24);
}

uint64_t ReadU64(absl::Span<const uint8_t> bytes, size_t offset) {
  uint64_t value = 0;
  for (int i = 7; i >= 0; --i) {
    value = (value << 8) | bytes[offset + i];
  }
  return value;
}

bool RangeIsInBounds(uint64_t offset, uint64_t size, size_t container_size) {
  const uint64_t limit = container_size;
  return offset <= limit && size <= limit - offset;
}

bool MultiplyWithoutOverflow(uint64_t lhs, uint64_t rhs, uint64_t* result) {
  if (lhs != 0 && rhs > std::numeric_limits<uint64_t>::max() / lhs) {
    return false;
  }
  *result = lhs * rhs;
  return true;
}

bool AlignToFourWithoutOverflow(uint64_t size, uint64_t* aligned_size) {
  if (size > std::numeric_limits<uint64_t>::max() - 3) return false;
  *aligned_size = (size + 3) & ~uint64_t{3};
  return true;
}

bool IsMtgpuOwner(absl::Span<const uint8_t> owner) {
  // ELF note names normally include their terminating NUL in n_namesz. Accept
  // the un-terminated spelling too, but no other suffix or embedded NUL.
  if (owner.size() == kMtgpuNoteOwner.size() + 1 && owner.back() == 0) {
    owner.remove_suffix(1);
  }
  return owner.size() == kMtgpuNoteOwner.size() &&
         std::equal(owner.begin(), owner.end(), kMtgpuNoteOwner.begin());
}

absl::Status ValidateNoteSegment(absl::Span<const uint8_t> bytes,
                                 uint64_t segment_offset, uint64_t segment_size,
                                 bool* found_vendor_note) {
  const uint64_t segment_end = segment_offset + segment_size;
  uint64_t cursor = segment_offset;
  while (cursor < segment_end) {
    if (segment_end - cursor < kElfNoteHeaderSize) {
      return InvalidMubin("truncated PT_NOTE header");
    }
    const size_t header_offset = static_cast<size_t>(cursor);
    const uint32_t name_size = ReadU32(bytes, header_offset);
    const uint32_t descriptor_size = ReadU32(bytes, header_offset + 4);
    const uint32_t note_type = ReadU32(bytes, header_offset + 8);
    cursor += kElfNoteHeaderSize;

    uint64_t padded_name_size;
    if (!AlignToFourWithoutOverflow(name_size, &padded_name_size) ||
        padded_name_size > segment_end - cursor) {
      return InvalidMubin("PT_NOTE owner exceeds its segment");
    }
    const size_t name_offset = static_cast<size_t>(cursor);
    const absl::Span<const uint8_t> owner =
        bytes.subspan(name_offset, name_size);
    cursor += padded_name_size;

    uint64_t padded_descriptor_size;
    if (!AlignToFourWithoutOverflow(descriptor_size, &padded_descriptor_size) ||
        padded_descriptor_size > segment_end - cursor) {
      return InvalidMubin("PT_NOTE descriptor exceeds its segment");
    }
    cursor += padded_descriptor_size;

    if (IsMtgpuOwner(owner) && note_type == kMtgpuNoteType) {
      *found_vendor_note = true;
    }
  }
  return absl::OkStatus();
}

}  // namespace

absl::StatusOr<MusaMubinMetadata> ValidateMusaMubin(
    absl::Span<const uint8_t> bytes) {
  static_assert(kMubinElfMachine <= std::numeric_limits<uint16_t>::max());

  if (bytes.empty()) return InvalidMubin("artifact is empty");
  if (bytes.size() < kElfIdentificationSize) {
    return InvalidMubin("truncated ELF identification");
  }
  if (!std::equal(kElfMagic.begin(), kElfMagic.end(), bytes.begin())) {
    return InvalidMubin("ELF magic is missing");
  }
  if (bytes[kElfClassOffset] != kElfClass64) {
    return InvalidMubin("ELF class is not ELF64");
  }
  if (bytes[kElfDataOffset] != kElfDataLittleEndian) {
    return InvalidMubin("ELF byte order is not little-endian");
  }
  if (bytes[kElfIdentVersionOffset] != kElfCurrentVersion) {
    return InvalidMubin("ELF identification version is not current");
  }
  if (bytes.size() < kElf64HeaderSize) {
    return InvalidMubin("truncated ELF64 header");
  }
  if (ReadU16(bytes, kElfTypeOffset) != kElfTypeDynamic) {
    return InvalidMubin("ELF object type is not ET_DYN");
  }
  const uint16_t elf_machine = ReadU16(bytes, kElfMachineOffset);
  if (elf_machine != kMubinElfMachine) {
    return InvalidMubin(absl::StrFormat("ELF machine is %d, expected %d",
                                        elf_machine, kMubinElfMachine));
  }
  if (ReadU32(bytes, kElfVersionOffset) != kElfCurrentVersion) {
    return InvalidMubin("ELF header version is not current");
  }
  if (ReadU16(bytes, kElfHeaderSizeOffset) != kElf64HeaderSize) {
    return InvalidMubin("unexpected ELF64 header size");
  }

  const uint16_t program_header_size =
      ReadU16(bytes, kElfProgramHeaderEntrySizeOffset);
  const uint16_t program_header_count =
      ReadU16(bytes, kElfProgramHeaderCountOffset);
  if (program_header_size != kElf64ProgramHeaderSize) {
    return InvalidMubin("unexpected ELF64 program-header size");
  }
  if (program_header_count == 0) {
    return InvalidMubin("ELF has no program headers");
  }

  uint64_t program_header_table_size;
  if (!MultiplyWithoutOverflow(program_header_count, program_header_size,
                               &program_header_table_size)) {
    return InvalidMubin("program-header table size overflows");
  }
  const uint64_t program_header_offset =
      ReadU64(bytes, kElfProgramHeaderOffset);
  if (!RangeIsInBounds(program_header_offset, program_header_table_size,
                       bytes.size())) {
    return InvalidMubin("program-header table exceeds the artifact");
  }

  bool found_executable_load = false;
  bool found_vendor_note = false;
  for (uint16_t i = 0; i < program_header_count; ++i) {
    const uint64_t header_offset =
        program_header_offset + uint64_t{i} * program_header_size;
    const size_t offset = static_cast<size_t>(header_offset);
    const uint32_t type = ReadU32(bytes, offset + kProgramHeaderTypeOffset);
    const uint32_t flags = ReadU32(bytes, offset + kProgramHeaderFlagsOffset);
    const uint64_t file_offset =
        ReadU64(bytes, offset + kProgramHeaderFileOffset);
    const uint64_t file_size =
        ReadU64(bytes, offset + kProgramHeaderFileSizeOffset);
    const uint64_t memory_size =
        ReadU64(bytes, offset + kProgramHeaderMemorySizeOffset);

    if (!RangeIsInBounds(file_offset, file_size, bytes.size())) {
      return InvalidMubin(
          absl::StrFormat("program header %d exceeds the artifact", i));
    }
    if (type == kProgramHeaderTypeLoad && file_size > memory_size) {
      return InvalidMubin(absl::StrFormat(
          "PT_LOAD program header %d has p_filesz greater than p_memsz", i));
    }
    if (type == kProgramHeaderTypeLoad &&
        (flags & kProgramHeaderFlagExecutable) != 0) {
      found_executable_load = true;
    }
    if (type == kProgramHeaderTypeNote) {
      absl::Status status = ValidateNoteSegment(bytes, file_offset, file_size,
                                                &found_vendor_note);
      if (!status.ok()) return status;
    }
  }

  if (!found_executable_load) {
    return InvalidMubin("ELF has no executable PT_LOAD segment");
  }
  if (!found_vendor_note) {
    return InvalidMubin("ELF has no MTGPU note with type 0x40");
  }

  return MusaMubinMetadata{
      .elf_abi_version = bytes[kElfAbiVersionOffset],
      .elf_machine = elf_machine,
      .vendor_note_type = kMtgpuNoteType,
  };
}

}  // namespace stream_executor::musa
