/* Copyright 2024 The OpenXLA Authors.

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
#ifndef XLA_SERVICE_GPU_FUSIONS_MLIR_OPS
#define XLA_SERVICE_GPU_FUSIONS_MLIR_OPS

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "xla/backends/gpu/codegen/emitters/ir/xla_gpu_dialect.td"
include "xla/backends/gpu/codegen/emitters/ir/xla_gpu_attrs.td"
include "xla/backends/gpu/codegen/emitters/ir/xla_gpu_types.td"

class XLAGPU_Op<string mnemonic, list<Trait> traits = []> :
      Op<XlaGpuDialect, mnemonic, traits> {
}

def XLAGPU_AllocateSharedOp : XLAGPU_Op<"allocate_shared", [
      DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>
  ]> {
  let summary = "Allocates a shared memory tile.";

  let description = [{
    Allocates a shared memory tensor. The tensor is shared among all threads in
    a block.

    ```mlir
    %shared = xla_gpu.allocate_shared : tensor<32x32xf32>
    ```
  }];

  let results = (outs AnyStaticShapeTensor:$result);

  let assemblyFormat = "attr-dict `:` type($result)";
}

def XLAGPU_SyncThreadsOp : XLAGPU_Op<"sync_threads", [
   TypesMatchWith<"result type matches type of dest",
                   "operands", "results", "$_self">,
   DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>
  ]> {
  let summary = "Synchronizes threads.";

  let description = [{
    Synchronizes threads, taking any number of distributed tensors and returning
    the synchronized state.
  }];

  let arguments = (ins Variadic<AnyRankedTensor>:$operands);
  let results = (outs Variadic<AnyRankedTensor>:$results);

  let assemblyFormat = "operands attr-dict `:` type($operands)";
}

def XLAGPU_AsyncCopyGlobalToSharedOp :
    XLAGPU_Op<"async_copy_global_to_shared", [
      TypesMatchWith<"result type matches destination type",
                     "destination", "result", "$_self">
    ]> {
  let summary = "Copies one lane's contiguous bytes directly into LDS.";

  let description = [{
    Issues an architecture-specific asynchronous global-to-shared copy. The
    destination index names the uniform base address for the current wave; the
    hardware adds the lane's byte offset. This matches CDNA buffer-to-LDS copy
    instructions and avoids staging data through vector registers.
  }];

  let arguments = (ins AnyRankedTensor:$source,
                       Index:$source_index,
                       AnyRankedTensor:$destination,
                       Index:$destination_index,
                       I32Attr:$copy_bytes);
  let results = (outs AnyRankedTensor:$result);

  let assemblyFormat = [{
    $source `[` $source_index `]` `to` $destination
    `[` $destination_index `]` `bytes` `=` $copy_bytes attr-dict
    `:` type($source) `,` type($destination)
  }];
}

def XLAGPU_BufferLoadOp : XLAGPU_Op<"buffer_load"> {
  let summary = "Loads a contiguous vector through an AMD buffer descriptor.";

  let description = [{
    Loads a contiguous vector from a physical linear tensor offset using an
    AMD raw buffer instruction. This preserves the BufferCopy128b operation
    used by FlyDSL kernels instead of allowing a generic global vector load.
    `cache_policy` is forwarded to the target instruction; gfx942 uses bit 1
    for the non-temporal policy used by single-use attention K/V tiles.
  }];

  let arguments = (ins AnyRankedTensor:$source, Index:$source_index,
                       DefaultValuedAttr<I32Attr, "0">:$cache_policy);
  // Packed sub-byte loads may use i32 to fetch eight S4 values at once.
  let results = (outs AnyTypeOf<[AnyVectorOfNonZeroRank, I32]>:$result);

  let assemblyFormat = [{
    $source `[` $source_index `]` attr-dict `:` type($source) `->` type($result)
  }];
}

def XLAGPU_ScalarBufferLoadOp : XLAGPU_Op<"scalar_buffer_load"> {
  let summary = "Loads wave-uniform i32 metadata into an AMD scalar register.";

  let description = [{
    Loads one i32 from a wave-uniform physical tensor offset using
    llvm.amdgcn.s.buffer.load. This is intended for kernel metadata such as
    sequence lengths and page-table entries that must remain in SGPRs.
  }];

  let arguments = (ins AnyRankedTensor:$source, Index:$source_index);
  let results = (outs I32:$result);

  let assemblyFormat = [{
    $source `[` $source_index `]` attr-dict `:` type($source) `->` type($result)
  }];
}

def XLAGPU_BufferStoreOp : XLAGPU_Op<"buffer_store", [
    TypesMatchWith<"result type matches destination type",
                   "destination", "result", "$_self">
  ]> {
  let summary = "Stores a contiguous vector through an AMD buffer descriptor.";

  let description = [{
    Stores a contiguous vector to a physical linear tensor offset using an AMD
    raw buffer instruction. The descriptor bounds suppress out-of-range writes,
    which permits branchless GEMM epilogues at matrix edges. `cache_policy` is
    forwarded to the target buffer instruction; gfx942 uses bit 1 for `nt`.
  }];

  let arguments = (ins AnyVectorOfNonZeroRank:$value,
                       AnyRankedTensor:$destination,
                       Index:$destination_index,
                       I32Attr:$cache_policy);
  let results = (outs AnyRankedTensor:$result);

  let assemblyFormat = [{
    $value `,` $destination `[` $destination_index `]`
    `cache_policy` `=` $cache_policy attr-dict
    `:` type($value) `,` type($destination)
  }];
}

def XLAGPU_BufferCompletionTicketOp :
    XLAGPU_Op<"buffer_completion_ticket"> {
  let summary = "Claims a generation-tagged inter-workgroup completion ticket.";

  let description = [{
    Treats two adjacent U32 buffer elements as an epoch and completion count.
    The operation returns the previous count for the current AMD dispatch. It
    lowers to a noinline device helper so ticket bookkeeping does not increase
    register pressure in the producer/reducer kernel.
  }];

  let arguments = (ins AnyRankedTensor:$tickets, Index:$ticket_index);
  let results = (outs I32:$previous);

  let assemblyFormat = [{
    $tickets `[` $ticket_index `]` attr-dict `:` type($tickets)
  }];
}

def XLAGPU_AccumulatorStoreOp : XLAGPU_Op<"accumulator_store", [
    TypesMatchWith<"result type matches destination type",
                   "destination", "result", "$_self">
  ]> {
  let summary = "Stores four FP32 accumulator values directly from AGPRs.";

  let description = [{
    Stores four contiguous FP32 values with an AMD `global_store_dwordx4`
    whose payload is constrained to the accumulator register file. This
    avoids four explicit AGPR-to-VGPR reads in wide MFMA epilogues. The
    destination index must be in bounds; unlike `buffer_store`, this operation
    does not use descriptor bounds to suppress invalid writes. Its byte offset
    must fit in 32 bits.
  }];

  let arguments = (ins VectorOfLengthAndType<[4], [F32]>:$value,
                       AnyRankedTensor:$destination,
                       Index:$destination_index);
  let results = (outs AnyRankedTensor:$result);

  let assemblyFormat = [{
    $value `,` $destination `[` $destination_index `]` attr-dict
    `:` type($value) `,` type($destination)
  }];
}

def XLAGPU_TileBufferStoreOp : XLAGPU_Op<"tile_buffer_store", [
    TypesMatchWith<"result type matches destination type",
                   "destination", "result", "$_self">
  ]> {
  let summary = "Stores a vector through a tile-scoped AMD buffer descriptor.";

  let description = [{
    Stores a contiguous vector through a descriptor whose base pointer is
    advanced to a workgroup-uniform tile offset. The remaining vector offset
    maps to the lane-varying voffset operand. This matches the descriptor form
    used by Triton for fully in-bounds tiled kernels.
  }];

  let arguments = (ins AnyVectorOfNonZeroRank:$value,
                       AnyRankedTensor:$destination,
                       Index:$vector_index,
                       Index:$tile_index,
                       I32Attr:$cache_policy);
  let results = (outs AnyRankedTensor:$result);

  let assemblyFormat = [{
    $value `,` $destination `[` $tile_index `+` $vector_index `]`
    `cache_policy` `=` $cache_policy attr-dict
    `:` type($value) `,` type($destination)
  }];
}

def XLAGPU_SplitBufferLoadOp : XLAGPU_Op<"split_buffer_load"> {
  let summary = "Loads a contiguous vector with split AMD buffer offsets.";

  let description = [{
    Loads a contiguous vector using separate lane-varying and uniform physical
    linear tensor offsets. The lowering maps these to the voffset and soffset
    operands of an AMD raw buffer instruction, allowing loop-varying uniform
    offsets to remain in SGPRs instead of being combined in every lane. For a
    sub-byte source, both logical offsets must independently be byte-aligned.
  }];

  let arguments = (ins AnyRankedTensor:$source, Index:$vector_index,
                       Index:$scalar_index);
  // Packed sub-byte loads may use an integer result whose bit width matches
  // the physical bytes fetched by the buffer instruction.
  let results = (outs AnyTypeOf<[AnyVectorOfNonZeroRank, I32]>:$result);

  let assemblyFormat = [{
    $source `[` $vector_index `+` $scalar_index `]` attr-dict
    `:` type($source) `->` type($result)
  }];
}

def XLAGPU_TileBufferLoadOp : XLAGPU_Op<"tile_buffer_load"> {
  let summary = "Loads a vector through a tile-scoped AMD buffer descriptor.";

  let description = [{
    Loads a contiguous vector through a descriptor whose base pointer is
    advanced to a workgroup-uniform tile offset. The remaining vector offset
    maps to the lane-varying voffset operand. This operation is intended for
    specialized kernels whose tile accesses are known to be fully in bounds.
  }];

  let arguments = (ins AnyRankedTensor:$source, Index:$vector_index,
                       Index:$tile_index);
  let results = (outs AnyVectorOfNonZeroRank:$result);

  let assemblyFormat = [{
    $source `[` $tile_index `+` $vector_index `]` attr-dict
    `:` type($source) `->` type($result)
  }];
}

def XLAGPU_ShuffleReduceOp : XLAGPU_Op<"shuffle_reduce",
    [Pure,  CallOpInterface,
     TypesMatchWith<"result type matches type of operands",
                     "operands", "results", "$_self">]> {
  let summary = "Performs a full warp shuffle and reduces the values";
  let description = [{
    This op performs a full warp shuffle and reduces the results using the given
    function. The function is invoked with the operands from the low lanes,
    followed by the operands from the high lanes. For example:

    ```
    %result:2 = xla_gpu.shuffle_reduce (%in0, %in1) to 16 combiner=@argmax
    ```

    Will perform shuffles with distance 16, 8, 4, 2 and 1, and will invoke
    @argmax five times. The first invocations will be

    ```
    @argmax(%in0[i], %in1[i], %in0[16+i], %in1[16+i])
    ```
  }];
  let builders = [
    OpBuilder<(ins "mlir::func::FuncOp":$combiner, "mlir::ValueRange":$operands, "int64_t":$max_distance), [{
      $_state.addOperands(operands);
      $_state.addAttribute("combiner", mlir::SymbolRefAttr::get(combiner));
      $_state.addAttribute("max_distance",
          mlir::IntegerAttr::get(
              mlir::IntegerType::get(combiner.getContext(), 64),
              max_distance));
      $_state.addTypes(combiner.getFunctionType().getResults());
    }]>];
  let arguments = (ins FlatSymbolRefAttr:$combiner,
                   Variadic<AnyType>:$operands,
                   I64Attr:$max_distance);
  let results = (outs Variadic<AnyType>:$results);

  let extraClassDeclaration = [{
    mlir::CallInterfaceCallable getCallableForCallee() {
      return (*this)->getAttrOfType<mlir::SymbolRefAttr>("combiner");
    }
    operand_range getArgOperands() {
      return getOperands();
    }
    mlir::MutableOperandRange getArgOperandsMutable() {
      return getOperandsMutable();
    }
    void setCalleeFromCallable(mlir::CallInterfaceCallable callee) {
      (*this)->setAttr("reducer", callee.get<mlir::SymbolRefAttr>());
    }
    mlir::Attribute removeArgAttrsAttr() { return nullptr; }
    mlir::Attribute removeResAttrsAttr() { return nullptr; }
    mlir::ArrayAttr getArgAttrsAttr() { return nullptr; }
    mlir::ArrayAttr getResAttrsAttr() { return nullptr; }
    void setArgAttrsAttr(mlir::ArrayAttr) { return; }
    void setResAttrsAttr(mlir::ArrayAttr) { return; }
  }];
  let hasCustomAssemblyFormat = 1;
}

def XLAGPU_InsertOp : XLAGPU_Op<"insert", [TypesMatchWith<
                   "expected result type to match dest type",
                   "result", "dest", "$_self">]> {
  let summary = "Inserts an indexed vector into a tensor";
  let arguments = (ins XLAGPU_IndexedVectorType:$source,
                       Variadic<Index>:$indices,
                       AnyRankedTensor:$dest,
                       XLA_IndexingMapAttr:$map);
  let results = (outs AnyRankedTensor:$result);
  let hasVerifier = 1;
  let assemblyFormat = [{
    $source `(` $indices `)` `into` $dest `at` $map attr-dict `:` type($source) `->` type($result)
  }];
}

def XLAGPU_ReduceOp : XLAGPU_Op<"reduce", [
    Pure,  CallOpInterface, SameVariadicOperandSize,
    DeclareOpInterfaceMethods<InferTypeOpInterface>
    ]> {
  let summary = "Performs a reduction";
  let description = [{
    The `xla_gpu.reduce` op performs a variadic reduction of the provided
    operands using the list of dimensions and a symbol for a combiner function.

    ```mlir
    func.func @add(%a_acc: f32, %b_acc: i32, %a: f32, %b: i32)
        -> (f32, i32) {
      %0 = arith.addf %a_acc, %a : f32
      %1 = arith.addi %b_acc, %b : i32
      func.return %0, %1 : f32, i32
    }
    %sum:2 = xla_gpu.reduce (%in0, %in1) inits(%init0, %init1) dimensions=[0, 2]
      combiner=@add : tensor<16x8x4xf32>, tensor<16x8x4xi32>
        to tensor<8xf32>, tensor<8xi32>
    ```
  }];
  let arguments = (ins
    Variadic<AnyRankedTensor>:$inputs,
    Variadic<AnyType>:$inits,
    ConfinedAttr<DenseI64ArrayAttr,
      [DenseArrayStrictlySorted<DenseI64ArrayAttr>]>:$dimensions,
    FlatSymbolRefAttr:$combiner);
  let results = (outs Variadic<AnyRankedTensor>:$results);

  let extraClassDeclaration = [{
    operand_range getArgOperands() {
      return getInputs();
    }
    mlir::MutableOperandRange getArgOperandsMutable() {
      return getInputsMutable();
    }
    mlir::CallInterfaceCallable getCallableForCallee() {
      return (*this)->getAttrOfType<mlir::SymbolRefAttr>("combiner");
    }
    void setCalleeFromCallable(mlir::CallInterfaceCallable callee) {
      (*this)->setAttr("combiner", callee.get<mlir::SymbolRefAttr>());
    }
    mlir::Attribute removeArgAttrsAttr() { return nullptr; }
    mlir::Attribute removeResAttrsAttr() { return nullptr; }
    mlir::ArrayAttr getArgAttrsAttr() { return nullptr; }
    mlir::ArrayAttr getResAttrsAttr() { return nullptr; }
    void setArgAttrsAttr(mlir::ArrayAttr) { return; }
    void setResAttrsAttr(mlir::ArrayAttr) { return; }
  }];
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def PdlWaitOp : XLAGPU_Op<"pdl_wait", []> {
  let summary = "Waits for upstream dependencies.";
}

def PdlLaunchDependentsOp : XLAGPU_Op<"pdl_launch_dependents", []> {
  let summary = "Triggers programmatic dependent launch.";
}

#endif // XLA_SERVICE_GPU_FUSIONS_MLIR_OPS
