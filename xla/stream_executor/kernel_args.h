/* Copyright 2015 The OpenXLA Authors.

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

#ifndef XLA_STREAM_EXECUTOR_KERNEL_ARGS_H_
#define XLA_STREAM_EXECUTOR_KERNEL_ARGS_H_

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <tuple>
#include <type_traits>
#include <utility>
#include <variant>

#include "absl/base/macros.h"
#include "absl/container/fixed_array.h"
#include "absl/container/inlined_vector.h"
#include "absl/functional/function_ref.h"
#include "absl/functional/overload.h"
#include "absl/log/check.h"
#include "absl/meta/type_traits.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/kernel_metadata.h"
#include "xla/stream_executor/tensor_map.h"

namespace stream_executor {

// A virtual base class for storing packed arguments.
struct PackedArgBase {
  virtual ~PackedArgBase() = default;
  virtual void* argument_address() const = 0;
  virtual int64_t size() const = 0;
};

//===----------------------------------------------------------------------===//
// Kernel arguments
//===----------------------------------------------------------------------===//

// Builting Kernel argument type widely used in kernels compiled by XLA.
using KernelArg = std::variant<DeviceAddressBase, TensorMap, int64_t>;

// A virtual base class for passing kernel arguments to a stream executor APIs.
class KernelArgs {
 public:
  template <typename T>
  using IsKernelArgs = std::enable_if_t<std::is_base_of<KernelArgs, T>::value>;

  enum class Kind {
    // A list of type-erased DeviceAddressBase pointers to on-device memory.
    // This type of kernel arguments used only when the kernel has to do its own
    // custom packing, e.g. wrap all device pointers into a custom structure,
    // but can't be implemented as a TypedKernel because it has to be passed
    // around as a generic Kernel.
    kDeviceAddressArray,

    // A list of kernel arguments packed into a storage that can be passed
    // directly to device kernel as void** kernel parameters.
    kPackedArray
  };

  virtual ~KernelArgs() = default;

  // Gets the number of arguments added so far, including shared memory
  // arguments.
  virtual size_t number_of_arguments() const = 0;

  // Gets the total number of shared memory bytes added so far.
  virtual uint64_t number_of_shared_bytes() const = 0;

  virtual Kind kind() const = 0;
};

class PackableKernelArgs {
 public:
  virtual ~PackableKernelArgs() = default;
  virtual absl::Span<const PackedArgBase* const> packed_args() const = 0;
};

//===----------------------------------------------------------------------===//
// Kernel argument packing
//===----------------------------------------------------------------------===//

// KernelArgPacking template specialization defines how arguments are passed to
// the device kernel. It is essentially a functor that converts user-defined C++
// type to another C++ type that can be passed to device kernel, and in general
// such type must be a POD data structure (in C++ terms it must satisfy
// `std::is_trivially_copyable` type trait), which is later passed to the device
// kernel (and it must match the device ABI).
//
// Packed arguments passed by pointer to the underlying device kernel launch
// API (see `argument_addresses` below), and how exactly the value is passed to
// the device kernel is platform specific (we assume that bytes are copied into
// the kernel launch command and sent to the device, and for this reason we
// require packed type to be trivially copyable).
//
// Default argument packing rules:
//
//   (1) `DeviceAddress` passed as an opaque `void*` pointer.
//   (2) We have a special case for passing pointers to `DeviceAddress` where we
//       also pass it as an opaque device pointer.
//   (3) We do not support pointer arguments, as we should not be passing a
//       pointers to host memory to device kernels. We check this at compile
//       time, to avoid hard to debug run time errors later.
//   (4) For all other POD types we always strip references and store a copy of
//       an argument in the kernel arguments array.
//
// Users can override default kernel argument packing by specializing this
// template, i.e. it allows packing custom user-defined types according to the
// ABI requirement of a device kernel. This indirection allows library
// implementation to hide device kernel ABI from the end user:
//
// Example: hiding library implementation from headers included by users
//
//   struct LibraryArg {
//     void* type_erased_handle;
//   };
//
//  // Hide type erased handle packing by specializing the template and
//  // passing an opaque array of bytes to the device kernel.
//  template<>
//  struct KernelArgPacking<LibraryArg> {
//    using Type = std::aligned_storage_t</*size=*/128, /*alignment=*/8>;
//    static Type Pack(const LibraryArg& arg);
//  };
//
template <typename T>
struct KernelArgPacking {
  static_assert(!std::is_pointer_v<T>, "cannot pass raw pointer to the device");

  // A type of the argument passed to the device kernel.
  using Type = T;

  // Packs an argument as the device argument.
  static Type Pack(const T& arg) { return arg; }
};

// A template specialization for packing statically sized arrays.
template <typename T, size_t N>
struct KernelArgPacking<T[N]> {
  using Type = std::array<T, N>;

  static Type Pack(const T (&arg)[N]) {
    std::array<T, N> arr;
    std::copy_n(arg, N, arr.begin());
    return arr;
  }
};

// A collection of DeviceAddress(Base) specializations: device address is always
// passed as a simple pointer. We assume that the device kernel itself knows the
// addressable range (always true for kernels compiled by XLA), or size is
// passed as a separate argument (for custom kernels).

template <>
struct KernelArgPacking<DeviceAddressBase> {
  using Type = void*;
  static void* Pack(const DeviceAddressBase& addr) { return addr.opaque(); }
};

template <>
struct KernelArgPacking<DeviceAddressBase*> {
  using Type = void*;
  static void* Pack(const DeviceAddressBase* addr) { return addr->opaque(); }
};

template <>
struct KernelArgPacking<const DeviceAddressBase*> {
  using Type = const void*;
  static void* Pack(const DeviceAddressBase* addr) { return addr->opaque(); }
};

template <typename T>
struct KernelArgPacking<DeviceAddress<T>> {
  using Type = T*;
  static T* Pack(const DeviceAddress<T> addr) { return addr.base(); }
};

template <typename T>
struct KernelArgPacking<DeviceAddress<T>*> {
  using Type = T*;
  static T* Pack(const DeviceAddress<T>* addr) { return addr->base(); }
};

template <typename T>
struct KernelArgPacking<const DeviceAddress<T>*> {
  using Type = const T*;
  static T* Pack(const DeviceAddress<T>* addr) { return addr->base(); }
};

//===----------------------------------------------------------------------===//
// Packed kernel argument for hidden POD types.
//===----------------------------------------------------------------------===//

// A dynamically-sized POD storage for packed kernel argument. This allows
// passing hidden POD library types by value to device kernels, without having
// to leak these types via header files.
//
// Example: We use this mechanism to pass NCCL device communicator to device
// kernels, without having to export NCCL headers from XLA:GPU communicator
// library (see `gpu_communicator.h`), GPU device communicator simply returns
// a packed kernel argument to the caller, and NCCL implementation of the
// collectives library memcpy device communicator bytes into the packed storage.
class PackedKernelArg {
 public:
  // Constructs a packed kernel argument by allocating properly sized storage,
  // and calling user-defined initializer that must initialize storage with
  // byte-array that will be passed by value to the device kernel.
  using Initializer = absl::FunctionRef<void(absl::Span<char> packed)>;
  PackedKernelArg(size_t size_bytes, Initializer initialize)
      : storage_(size_bytes) {
    initialize(absl::MakeSpan(storage_));
  }

  void* data() { return storage_.data(); }
  size_t size_bytes() const { return storage_.size(); }

 private:
  absl::FixedArray<char> storage_;
};

// Kernel argument packing for `PackedKernelArg` is a no-op, we simply forward
// already packed data to the caller.
template <>
struct KernelArgPacking<PackedKernelArg> {
  using Type = PackedKernelArg;
  static PackedKernelArg Pack(PackedKernelArg arg) { return arg; }
};

//===----------------------------------------------------------------------===//
// Kernel arguments packed array
//===----------------------------------------------------------------------===//

// A virtual base class for passing kernel arguments packed into a storage so
// that we have stable addresses for all arguments. This is a low level API
// for passing arguments in a platform-specific way that relies on the
// knowledge of the ABI of the underlying platform.
//
// For example `cuLaunchKernel` accepts arguments as `void** kernelParams`,
// and packed array base guarantees that `argument_addresses` are compatible
// with the CUDA APIs.
//
// See: https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EXEC.html
class KernelArgsPackedArrayBase : public KernelArgs {
 public:
  // Gets the list of argument addresses.
  virtual absl::Span<const void* const> argument_addresses() const = 0;

  static bool classof(const KernelArgs* args) {
    return args->kind() == Kind::kPackedArray;
  }

  Kind kind() const final { return Kind::kPackedArray; }
};

//===----------------------------------------------------------------------===//
// Kernel arguments LLVM-style RTTI library
//===----------------------------------------------------------------------===//

template <class T, KernelArgs::IsKernelArgs<T>* = nullptr>
T* Cast(KernelArgs* args) {
  CHECK(T::classof(args)) << "Invalid arguments casting to a destination type: "
                          << typeid(T).name();
  CHECK(args != nullptr) << "Casted arguments must be not null";
  return static_cast<const T*>(args);
}

template <class T, KernelArgs::IsKernelArgs<T>* = nullptr>
const T* Cast(const KernelArgs* args) {
  CHECK(T::classof(args)) << "Invalid arguments casting to a destination type: "
                          << typeid(T).name();
  CHECK(args != nullptr) << "Casted arguments must be not null";
  return static_cast<const T*>(args);
}

template <class T, KernelArgs::IsKernelArgs<T>* = nullptr>
const T* DynCast(const KernelArgs* args) {
  CHECK(args != nullptr) << "Casted arguments must be not null";
  return T::classof(args) ? static_cast<const T*>(args) : nullptr;
}

template <class T, KernelArgs::IsKernelArgs<T>* = nullptr>
const T* DynCastOrNull(const KernelArgs* args) {
  return args && T::classof(args) ? static_cast<const T*>(args) : nullptr;
}

//===----------------------------------------------------------------------===//
// Kernel arguments packing for device address and POD args
//===----------------------------------------------------------------------===//

namespace internal {

// Trivially copyable packed argument type.
template <typename Packed>
struct PackedArg final : PackedArgBase {
  static_assert(std::is_trivially_copyable_v<Packed>,
                "Packed type must be trivially copyable");
  explicit PackedArg(Packed packed_arg) : packed_arg(std::move(packed_arg)) {}
  void* argument_address() const final {
    return const_cast<Packed*>(&packed_arg);
  }
  int64_t size() const final { return sizeof(Packed); }
  Packed packed_arg;
};

// Template specialization for already packed argument.
template <>
struct PackedArg<PackedKernelArg> : PackedArgBase {
  explicit PackedArg(PackedKernelArg packed_arg)
      : packed_arg(std::move(packed_arg)) {}
  void* argument_address() const final {
    return const_cast<PackedKernelArg&>(packed_arg).data();
  }
  int64_t size() const final { return packed_arg.size_bytes(); }
  PackedKernelArg packed_arg;
};

template <typename T>
PackedArg(T) -> PackedArg<T>;

// Type-erased storage for dynamically typed packed arguments. Small POD values
// are stored inline so that building the argument array used by ordinary XLA
// kernel launches does not allocate once per argument. Large and dynamically
// sized values retain the old out-of-line representation.
class PackedArgStorage final : public PackedArgBase {
 public:
  static constexpr size_t kInlineSize = 2 * sizeof(void*);

  template <typename Packed>
  explicit PackedArgStorage(Packed packed_arg) {
    static_assert(std::is_trivially_copyable_v<Packed>,
                  "Packed type must be trivially copyable");
    if constexpr (sizeof(Packed) <= kInlineSize &&
                  alignof(Packed) <= alignof(std::max_align_t)) {
      std::memcpy(inline_storage_.data(), &packed_arg, sizeof(Packed));
      size_ = sizeof(Packed);
    } else {
      out_of_line_ =
          std::make_unique<PackedArg<Packed>>(std::move(packed_arg));
    }
  }

  explicit PackedArgStorage(PackedKernelArg packed_arg)
      : out_of_line_(std::make_unique<PackedArg<PackedKernelArg>>(
            std::move(packed_arg))) {}

  PackedArgStorage(PackedArgStorage&&) = default;
  PackedArgStorage& operator=(PackedArgStorage&&) = default;
  PackedArgStorage(const PackedArgStorage&) = delete;
  PackedArgStorage& operator=(const PackedArgStorage&) = delete;

  void* argument_address() const final {
    if (out_of_line_) return out_of_line_->argument_address();
    return const_cast<std::byte*>(inline_storage_.data());
  }

  int64_t size() const final {
    return out_of_line_ ? out_of_line_->size() : size_;
  }

 private:
  alignas(std::max_align_t) std::array<std::byte, kInlineSize> inline_storage_;
  std::unique_ptr<PackedArgBase> out_of_line_;
  int64_t size_ = 0;
};

}  // namespace internal

//===----------------------------------------------------------------------===//
// Kernel arguments device address array
//===----------------------------------------------------------------------===//

class KernelArgsDeviceAddressArray : public KernelArgs,
                                     public PackableKernelArgs {
 public:
  KernelArgsDeviceAddressArray(absl::Span<const DeviceAddressBase> args,
                               size_t shared_memory_bytes)
      : device_addr_args_(args.begin(), args.end()),
        shared_memory_bytes_(shared_memory_bytes) {
    packed_args_.reserve(args.size());
    packed_arg_ptrs_.reserve(args.size());
    for (int i = 0; i < args.size(); i++) {
      auto& packed = packed_args_.emplace_back(
          KernelArgPacking<const DeviceAddressBase*>::Pack(&args[i]));
      packed_arg_ptrs_.push_back(&packed);
    }
  }

  static bool classof(const KernelArgs* args) {
    return args->kind() == Kind::kDeviceAddressArray;
  }

  Kind kind() const final { return Kind::kDeviceAddressArray; }

  size_t number_of_arguments() const final {
    return device_addr_args_.size() + (shared_memory_bytes_ > 0);
  }

  uint64_t number_of_shared_bytes() const final { return shared_memory_bytes_; }

  absl::Span<const DeviceAddressBase> device_addr_args() const {
    return device_addr_args_;
  }

  const void* device_addr_ptr(size_t index) const {
    return device_addr_args_[index].opaque();
  }

  size_t device_addr_size(size_t index) const {
    return device_addr_args_[index].size();
  }

  absl::Span<const PackedArgBase* const> packed_args() const final {
    return packed_arg_ptrs_;
  }

 private:
  // A storage for packed POD arguments added to this array.
  absl::InlinedVector<internal::PackedArgStorage, 8> packed_args_;
  absl::InlinedVector<const PackedArgBase*, 8> packed_arg_ptrs_;

  absl::InlinedVector<DeviceAddressBase, 4> device_addr_args_;
  size_t shared_memory_bytes_ = 0;
};

// TODO(ezhulenev): Remove this alias once all users are migrated.
using KernelArgsDeviceMemoryArray ABSL_DEPRECATE_AND_INLINE() =
    KernelArgsDeviceAddressArray;

// KernelArgsPackedArray is optimized for packing DeviceAddressBase pointers
// and POD arguments (i.e. scalars) when the number and type of arguments are
// not known at compile time. When the kernel signature is fully known at
// compile time, prefer `KernelArgsPackedTuple` as it requires fewer memory
// allocations and has lower overheads on a hot path.
class KernelArgsPackedArray : public KernelArgsPackedArrayBase,
                              public PackableKernelArgs {
 public:
  // The `num_args` is the maximum number of device address arguments that can
  // be stored in the array. Adding more arguments will can lead to UB because
  // of `device_addr_args_` storage reallocation. We don't reserve space for
  // packed arguments as we don't know how many of them we'll see.
  explicit KernelArgsPackedArray(size_t num_args) {
    device_addr_args_.reserve(num_args);
    packed_args_.reserve(num_args);
    packed_arg_ptrs_.reserve(num_args);
    argument_addresses_.reserve(num_args);
  }

  // KernelArgsPackedArray is not copyable or movable because argument
  // addresses point to inline storage that can't be moved.
  KernelArgsPackedArray(const KernelArgsPackedArray&) = delete;
  KernelArgsPackedArray& operator=(const KernelArgsPackedArray&) = delete;

  // Adds already packed argument to the list.
  void add_argument(PackedKernelArg arg) {
    AddPackedArgument(std::move(arg));
  }

  // Adds an argument to the list.
  template <typename T>
  void add_argument(const T& arg) {
    AddPackedArgument(KernelArgPacking<T>::Pack(arg));
  }

  // Adds a device address argument to the list.
  void add_argument(const DeviceAddressBase& arg) {
    DCHECK_LT(device_addr_args_.size(), device_addr_args_.capacity());
    device_addr_args_.emplace_back(arg.opaque());

    AddPackedArgument(
        KernelArgPacking<const DeviceAddressBase*>::Pack(&arg));
  }

  // Adds a shared memory argument to the list.
  //
  // The only significant information about a shared argument is its size, so
  // that is the only parameter in this function.
  void add_shared_bytes(size_t number_of_bytes) {
    shared_memory_bytes_ += number_of_bytes;
  }

  absl::Span<void*> device_addr_args() {
    return absl::MakeSpan(device_addr_args_);
  }

  // Gets the number of arguments added so far, including shared memory
  // arguments.
  size_t number_of_arguments() const final {
    return argument_addresses_.size() + (shared_memory_bytes_ > 0);
  }

  // Gets the total number of shared memory bytes added so far.
  uint64_t number_of_shared_bytes() const final { return shared_memory_bytes_; }

  // Gets the list of argument addresses.
  absl::Span<const void* const> argument_addresses() const final {
    return argument_addresses_;
  }

  size_t shared_memory_bytes() const { return shared_memory_bytes_; }

  absl::Span<const PackedArgBase* const> packed_args() const final {
    return packed_arg_ptrs_;
  }

 private:
  template <typename Packed>
  void AddPackedArgument(Packed packed_arg) {
    const internal::PackedArgStorage* old_storage = packed_args_.data();
    auto& emplaced = packed_args_.emplace_back(std::move(packed_arg));

    // The constructor reservation covers all in-tree callers, but keep the
    // public add_argument API safe if a caller exceeds it. Inline argument
    // addresses move together with packed_args_ during a reallocation.
    if (old_storage != packed_args_.data() && !argument_addresses_.empty()) {
      packed_arg_ptrs_.clear();
      argument_addresses_.clear();
      packed_arg_ptrs_.reserve(packed_args_.size());
      argument_addresses_.reserve(packed_args_.size());
      for (internal::PackedArgStorage& packed : packed_args_) {
        packed_arg_ptrs_.push_back(&packed);
        argument_addresses_.push_back(packed.argument_address());
      }
      return;
    }

    packed_arg_ptrs_.push_back(&emplaced);
    argument_addresses_.push_back(emplaced.argument_address());
  }

  // A storage for device address arguments added to this array.
  absl::InlinedVector<void*, 8> device_addr_args_;

  // A storage for packed POD arguments added to this array.
  absl::InlinedVector<internal::PackedArgStorage, 8> packed_args_;

  // Type-erased pointers into `packed_args_` for custom argument packing.
  absl::InlinedVector<const PackedArgBase*, 8> packed_arg_ptrs_;

  // Pointers to entries `device_addr_args_` or `packed_args_`.
  absl::InlinedVector<void*, 8> argument_addresses_;

  // Shared memory required by a kernel.
  size_t shared_memory_bytes_ = 0;
};

inline void PackKernelArgsInto(absl::Span<const DeviceAddressBase> args,
                               uint32_t shmem_bytes,
                               KernelArgsPackedArray* packed) {
  for (const DeviceAddressBase& buf : args) {
    packed->add_argument(buf);
  }
  packed->add_shared_bytes(shmem_bytes);
}

inline void PackKernelArgsInto(absl::Span<const KernelArg> args,
                               uint32_t shmem_bytes,
                               KernelArgsPackedArray* packed) {
  for (const auto& arg : args) {
    std::visit(
        absl::Overload{
            [&](const DeviceAddressBase& ptr) { packed->add_argument(ptr); },
            [&](int64_t i64) { packed->add_argument(i64); },
            [&](const TensorMap& m) { packed->add_argument(m.storage); },
        },
        arg);
  }
  packed->add_shared_bytes(shmem_bytes);
}

inline std::unique_ptr<KernelArgsPackedArray> PackKernelArgs(
    absl::Span<const DeviceAddressBase> args, uint32_t shmem_bytes) {
  auto packed = std::make_unique<KernelArgsPackedArray>(args.size());
  PackKernelArgsInto(args, shmem_bytes, packed.get());
  return packed;
}

inline std::unique_ptr<KernelArgsPackedArray> PackKernelArgs(
    absl::Span<const KernelArg> args, uint32_t shmem_bytes) {
  auto packed = std::make_unique<KernelArgsPackedArray>(args.size());
  PackKernelArgsInto(args, shmem_bytes, packed.get());
  return packed;
}

inline absl::StatusOr<std::unique_ptr<KernelArgsPackedArray>> PackKernelArgs(
    absl::Span<const DeviceAddressBase> args, const KernelMetadata& metadata) {
  return PackKernelArgs(args, metadata.shared_memory_bytes().value_or(0));
}

inline absl::StatusOr<std::unique_ptr<KernelArgsPackedArray>> PackKernelArgs(
    absl::Span<const KernelArg> args, const KernelMetadata& metadata) {
  return PackKernelArgs(args, metadata.shared_memory_bytes().value_or(0));
}

//===----------------------------------------------------------------------===//
// Kernel arguments tuple for statically know argument types
//===----------------------------------------------------------------------===//

// KernelArgsPackedTuple is optimized for packing arguments when their types
// are known at compile time, and somewhat similar to `std::tuple` but with a
// few special rules for passing device address arguments.
template <typename... Args>
class KernelArgsPackedTuple : public KernelArgsPackedArrayBase {
 public:
  static constexpr size_t kSize = sizeof...(Args);

  template <typename Arg>
  using Packed = typename KernelArgPacking<absl::remove_cvref_t<Arg>>::Type;
  using Storage = std::tuple<internal::PackedArg<Packed<Args>>...>;

  explicit KernelArgsPackedTuple(Args... args, size_t shared_memory_bytes)
      : storage_(internal::PackedArg(
            KernelArgPacking<absl::remove_cvref_t<Args>>::Pack(
                std::forward<Args>(args)))...),
        shared_memory_bytes_(shared_memory_bytes) {
    InitializeArgumentAddresses(std::make_index_sequence<kSize>{});
  }

  // KernelArgsPackedTuple is not copyable or movable because argument
  // addresses point to inline storage that can't be moved.
  KernelArgsPackedTuple(const KernelArgsPackedTuple&) = delete;
  KernelArgsPackedTuple& operator=(const KernelArgsPackedTuple&) = delete;

  size_t number_of_arguments() const final {
    return kSize + (shared_memory_bytes_ > 0);
  }

  uint64_t number_of_shared_bytes() const final { return shared_memory_bytes_; }

  absl::Span<const void* const> argument_addresses() const final {
    return absl::Span<const void* const>(argument_addresses_.data(), kSize);
  }

  // Compile time check that KernelArgsPackedTuple is compatible with
  // `OtherArgs`: after stripping const and reference all types match.
  template <typename... OtherArgs>
  static void CheckCompatibleStaticAssert() {
    static constexpr size_t kOtherSize = sizeof...(OtherArgs);
    static_assert(kSize == kOtherSize, "length of arguments packs must match");

    using StrippedArgs = std::tuple<absl::remove_cvref_t<Args>...>;
    using StrippedOtherArgs = std::tuple<absl::remove_cvref_t<OtherArgs>...>;
    static_assert(std::is_same_v<StrippedArgs, StrippedOtherArgs>,
                  "arguments types do not match");
  }

 private:
  template <size_t... Is>
  void InitializeArgumentAddresses(std::index_sequence<Is...>) {
    ((argument_addresses_[Is] = std::get<Is>(storage_).argument_address()),
     ...);
  }

  // Storage for packed kernel arguments.
  Storage storage_;

  // Shared memory required by a kernel.
  size_t shared_memory_bytes_ = 0;

  // Pointers into `storage_`.
  std::array<const void*, kSize> argument_addresses_;
};

// Packs the given arguments into a KernelArgsPackedTuple.
template <typename... Args>
std::unique_ptr<KernelArgsPackedArrayBase> PackKernelArgs(int64_t shmem_bytes,
                                                          Args... args) {
  using PackedArgs = KernelArgsPackedTuple<Args...>;
  return std::make_unique<PackedArgs>(std::forward<Args>(args)..., shmem_bytes);
}

}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_KERNEL_ARGS_H_
