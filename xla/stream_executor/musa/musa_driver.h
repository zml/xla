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

#ifndef XLA_STREAM_EXECUTOR_MUSA_MUSA_DRIVER_H_
#define XLA_STREAM_EXECUTOR_MUSA_MUSA_DRIVER_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "musa.h"
#include "xla/stream_executor/musa/musa_dso_loader.h"

namespace stream_executor::musa {

struct MusaPrimaryContextState {
  unsigned int flags;
  bool active;
};

// Directional facts for accesses issued by `source` to allocations owned by
// `peer`. The link attributes are meaningful only when `can_access_peer` is
// true. MUSA peer access is unidirectional, so callers must not reuse one
// result for the reverse direction.
struct MusaPeerAccessInfo {
  bool can_access_peer = false;
  bool link_attributes_available = false;
  int performance_rank = 0;
  bool native_atomic_supported = false;
  bool musa_array_access_supported = false;
  int mtlink_port_count = 0;

  friend bool operator==(const MusaPeerAccessInfo& lhs,
                         const MusaPeerAccessInfo& rhs) {
    return lhs.can_access_peer == rhs.can_access_peer &&
           lhs.link_attributes_available == rhs.link_attributes_available &&
           lhs.performance_rank == rhs.performance_rank &&
           lhs.native_atomic_supported == rhs.native_atomic_supported &&
           lhs.musa_array_access_supported == rhs.musa_array_access_supported &&
           lhs.mtlink_port_count == rhs.mtlink_port_count;
  }
};

// Address and size of a named global owned by a loaded MUSA module. The
// address remains valid only while the module remains loaded.
struct MusaModuleGlobal {
  MUdeviceptr address;
  size_t size;
};

// Typed, dynamically loaded subset of the MUSA driver API used by
// StreamExecutor. Initialization is attempted exactly once; both success and
// failure are cached. The resolved function table is immutable and ordinary
// calls never take the loader's initialization lock.
class MusaDriver {
 public:
  MusaDriver();
  explicit MusaDriver(
      std::unique_ptr<internal::MusaSymbolLoader> symbol_loader);
  virtual ~MusaDriver();

  MusaDriver(const MusaDriver&) = delete;
  MusaDriver& operator=(const MusaDriver&) = delete;

  static MusaDriver& Instance();

  virtual absl::Status Init();
  virtual absl::StatusOr<int> DriverVersion();
  virtual absl::StatusOr<int> DeviceCount();
  virtual absl::StatusOr<MUdevice> Device(int ordinal);
  virtual absl::StatusOr<MusaPeerAccessInfo> PeerAccessInfo(MUdevice source,
                                                            MUdevice peer);

  virtual absl::StatusOr<MUcontext> RetainPrimaryContext(MUdevice device);
  virtual absl::Status ReleasePrimaryContext(MUdevice device);
  virtual absl::Status SetPrimaryContextFlags(MUdevice device,
                                              unsigned int flags);
  virtual absl::StatusOr<MusaPrimaryContextState> PrimaryContextState(
      MUdevice device);

  virtual absl::Status SetCurrentContext(MUcontext context);
  virtual absl::StatusOr<MUcontext> CurrentContext();
  virtual absl::StatusOr<MUdevice> CurrentDevice();
  virtual absl::Status SynchronizeContext();
  // Enables the qualified default path from the current context to
  // `peer_context`. An already-enabled mapping is successful.
  virtual absl::Status EnablePeerAccess(MUcontext peer_context);
  virtual absl::StatusOr<MUcontext> ContextForPointer(MUdeviceptr pointer);
  virtual absl::Status MemcpyPeerAsync(MUdeviceptr destination,
                                       MUcontext destination_context,
                                       MUdeviceptr source,
                                       MUcontext source_context, uint64_t bytes,
                                       MUstream stream);

  virtual absl::StatusOr<MUmodule> LoadModuleData(const void* image);
  virtual absl::Status UnloadModule(MUmodule module);
  virtual absl::StatusOr<MUfunction> GetModuleFunction(MUmodule module,
                                                       const char* name);
  virtual absl::StatusOr<MusaModuleGlobal> GetModuleGlobal(MUmodule module,
                                                           const char* name);

  virtual absl::StatusOr<int> FunctionAttribute(MUfunction function,
                                                MUfunction_attribute attribute);
  virtual absl::StatusOr<int> MaxActiveBlocksPerMultiprocessor(
      MUfunction function, int block_size, size_t dynamic_shared_memory_bytes);

  virtual absl::Status LaunchKernel(
      MUfunction function, unsigned int grid_dim_x, unsigned int grid_dim_y,
      unsigned int grid_dim_z, unsigned int block_dim_x,
      unsigned int block_dim_y, unsigned int block_dim_z,
      unsigned int shared_memory_bytes, MUstream stream,
      void** kernel_parameters, void** extra);

  // `count` is the number of 32-bit elements, matching the native driver ABI;
  // callers with a byte count must validate divisibility by four first.
  virtual absl::Status MemsetD32Async(MUdeviceptr destination, uint32_t value,
                                      size_t count, MUstream stream);

  // Graph entry points are optional as a complete capability so older driver
  // DSOs keep ordinary stream execution. Command buffers require all of them.
  virtual bool GraphsAvailable();
  virtual absl::StatusOr<MUgraph> GraphCreate();
  virtual absl::Status GraphDestroy(MUgraph graph);
  virtual absl::StatusOr<MUgraphNode> GraphAddEmptyNode(
      MUgraph graph, absl::Span<const MUgraphNode> dependencies);
  virtual absl::StatusOr<MUgraphNode> GraphAddKernelNode(
      MUgraph graph, absl::Span<const MUgraphNode> dependencies,
      const MUSA_KERNEL_NODE_PARAMS& params);
  virtual absl::Status GraphKernelNodeSetParams(
      MUgraphNode node, const MUSA_KERNEL_NODE_PARAMS& params);
  virtual absl::StatusOr<MUgraphNode> GraphAddMemcpyD2DNode(
      MUgraph graph, absl::Span<const MUgraphNode> dependencies,
      MUdeviceptr destination, MUdeviceptr source, size_t bytes);
  virtual absl::Status GraphMemcpyD2DNodeSetParams(MUgraphNode node,
                                                   MUdeviceptr destination,
                                                   MUdeviceptr source,
                                                   size_t bytes);
  virtual absl::StatusOr<MUgraphNode> GraphAddMemsetNode(
      MUgraph graph, absl::Span<const MUgraphNode> dependencies,
      const MUSA_MEMSET_NODE_PARAMS& params);
  virtual absl::Status GraphMemsetNodeSetParams(
      MUgraphNode node, const MUSA_MEMSET_NODE_PARAMS& params);
  virtual absl::StatusOr<size_t> GraphNodeCount(MUgraph graph);
  virtual absl::StatusOr<size_t> GraphRootNodeCount(MUgraph graph);
  virtual absl::StatusOr<MUgraphExec> GraphInstantiate(MUgraph graph);
  virtual absl::Status GraphExecDestroy(MUgraphExec executable);
  virtual absl::Status GraphLaunch(MUgraphExec executable, MUstream stream);
  virtual absl::Status StreamSynchronize(MUstream stream);
  virtual absl::Status StreamBeginCapture(MUstream stream);
  virtual absl::StatusOr<MUgraph> StreamEndCapture(MUstream stream);

 private:
  struct Api;

  absl::Status Initialize();
  absl::Status ResultStatus(MUresult result, const char* expression) const;

  std::unique_ptr<internal::MusaSymbolLoader> symbol_loader_;
  std::once_flag init_once_;
  absl::Status init_status_ =
      absl::UnknownError("MUSA driver initialization has not run");
  std::unique_ptr<const Api> api_;
};

}  // namespace stream_executor::musa

#endif  // XLA_STREAM_EXECUTOR_MUSA_MUSA_DRIVER_H_
