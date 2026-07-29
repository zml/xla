
load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library", "td_library")
load("@rules_cc//cc:cc_binary.bzl", "cc_binary")
load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0 WITH LLVM-exception

exports_files(["LICENSE.txt"])

cc_binary(
    name = "cuda-tile-tblgen",
    srcs = glob([
        "tools/cuda-tile-tblgen/*.cpp",
        "tools/cuda-tile-tblgen/*.h",
    ]),
    deps = [
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:TableGen",
        "@llvm-project//llvm:config",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MlirTableGenMain",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TableGen",
    ],
)

td_library(
    name = "cuda_tile_td_files",
    srcs = glob(["include/cuda_tile/Dialect/CudaTile/IR/*.td"]) + [
        "lib/Dialect/CudaTile/IR/OpsCanonicalization.td",
    ],
    includes = ["include"],
    deps = [
        "@llvm-project//mlir:BuiltinDialectTdFiles",
        "@llvm-project//mlir:ControlFlowInterfacesTdFiles",
        "@llvm-project//mlir:FunctionInterfacesTdFiles",
        "@llvm-project//mlir:InferTypeOpInterfaceTdFiles",
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:SideEffectInterfacesTdFiles",
        "@llvm-project//mlir:ViewLikeInterfaceTdFiles",
    ],
)

gentbl_cc_library(
    name = "dialect_inc_gen",
    strip_include_prefix = "include",
    tbl_outs = {
        "include/cuda_tile/Dialect/CudaTile/IR/Dialect.h.inc": [
            "-gen-dialect-decls",
            "-dialect=cuda_tile",
        ],
        "include/cuda_tile/Dialect/CudaTile/IR/Dialect.cpp.inc": [
            "-gen-dialect-defs",
            "-dialect=cuda_tile",
        ],
    },
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/cuda_tile/Dialect/CudaTile/IR/Dialect.td",
    deps = [":cuda_tile_td_files"],
)

gentbl_cc_library(
    name = "interfaces_inc_gen",
    strip_include_prefix = "include",
    tbl_outs = {
        "include/cuda_tile/Dialect/CudaTile/IR/AttrInterfaces.h.inc": ["-gen-attr-interface-decls"],
        "include/cuda_tile/Dialect/CudaTile/IR/AttrInterfaces.cpp.inc": ["-gen-attr-interface-defs"],
        "include/cuda_tile/Dialect/CudaTile/IR/TypeInterfaces.h.inc": ["-gen-type-interface-decls"],
        "include/cuda_tile/Dialect/CudaTile/IR/TypeInterfaces.cpp.inc": ["-gen-type-interface-defs"],
    },
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/cuda_tile/Dialect/CudaTile/IR/Interfaces.td",
    deps = [":cuda_tile_td_files"],
)

gentbl_cc_library(
    name = "types_inc_gen",
    strip_include_prefix = "include",
    tbl_outs = {
        "include/cuda_tile/Dialect/CudaTile/IR/Types.h.inc": [
            "-gen-typedef-decls",
            "-typedefs-dialect=cuda_tile",
        ],
        "include/cuda_tile/Dialect/CudaTile/IR/Types.cpp.inc": [
            "-gen-typedef-defs",
            "-typedefs-dialect=cuda_tile",
        ],
        "include/cuda_tile/Dialect/CudaTile/IR/TypeConstraints.h.inc": [
            "-gen-type-constraint-decls",
            "-dialect=cuda_tile",
        ],
        "include/cuda_tile/Dialect/CudaTile/IR/TypeConstraints.cpp.inc": [
            "-gen-type-constraint-defs",
            "-dialect=cuda_tile",
        ],
    },
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/cuda_tile/Dialect/CudaTile/IR/Types.td",
    deps = [":cuda_tile_td_files"],
)

gentbl_cc_library(
    name = "attrdefs_inc_gen",
    strip_include_prefix = "include",
    tbl_outs = {
        "include/cuda_tile/Dialect/CudaTile/IR/AttrDefs.h.inc": [
            "-gen-attrdef-decls",
            "-attrdefs-dialect=cuda_tile",
        ],
        "include/cuda_tile/Dialect/CudaTile/IR/AttrDefs.cpp.inc": [
            "-gen-attrdef-defs",
            "-attrdefs-dialect=cuda_tile",
        ],
        "include/cuda_tile/Dialect/CudaTile/IR/Enums.h.inc": [
            "-gen-enum-decls",
            "-dialect=cuda_tile",
        ],
        "include/cuda_tile/Dialect/CudaTile/IR/Enums.cpp.inc": [
            "-gen-enum-defs",
            "-dialect=cuda_tile",
        ],
    },
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/cuda_tile/Dialect/CudaTile/IR/AttrDefs.td",
    deps = [":cuda_tile_td_files"],
)

gentbl_cc_library(
    name = "remarks_inc_gen",
    strip_include_prefix = "include",
    tbl_outs = {
        "include/cuda_tile/Dialect/CudaTile/IR/TileIRRemarks.h.inc": ["-gen-enum-decls"],
        "include/cuda_tile/Dialect/CudaTile/IR/TileIRRemarks.cpp.inc": ["-gen-enum-defs"],
    },
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/cuda_tile/Dialect/CudaTile/IR/Remarks.td",
    deps = [":cuda_tile_td_files"],
)

gentbl_cc_library(
    name = "ops_inc_gen",
    strip_include_prefix = "include",
    tbl_outs = {
        "include/cuda_tile/Dialect/CudaTile/IR/Ops.h.inc": ["-gen-op-decls"],
        "include/cuda_tile/Dialect/CudaTile/IR/Ops.cpp.inc": ["-gen-op-defs"],
    },
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/cuda_tile/Dialect/CudaTile/IR/Ops.td",
    deps = [":cuda_tile_td_files"],
)

gentbl_cc_library(
    name = "ops_canonicalization_inc_gen",
    strip_include_prefix = "canon_inc",
    tbl_outs = {"canon_inc/OpsCanonicalization.inc": ["-gen-rewriters"]},
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "lib/Dialect/CudaTile/IR/OpsCanonicalization.td",
    deps = [":cuda_tile_td_files"],
)

gentbl_cc_library(
    name = "bytecode_ops_inc_gen",
    strip_include_prefix = "bytecode_inc",
    tbl_outs = {
        "bytecode_inc/Bytecode.inc": ["-gen-cuda-tile-bytecode"],
        "bytecode_inc/BytecodeReader.inc": ["-gen-cuda-tile-bytecode-reader"],
    },
    tblgen = ":cuda-tile-tblgen",
    td_file = "include/cuda_tile/Dialect/CudaTile/IR/Ops.td",
    deps = [":cuda_tile_td_files"],
)

gentbl_cc_library(
    name = "bytecode_opcodes_inc_gen",
    strip_include_prefix = "bytecode_inc",
    tbl_outs = {"bytecode_inc/StaticOpcodes.inc": ["-gen-cuda-tile-opcodes"]},
    tblgen = ":cuda-tile-tblgen",
    td_file = "include/cuda_tile/Dialect/CudaTile/IR/BytecodeOpcodes.td",
    deps = [":cuda_tile_td_files"],
)

gentbl_cc_library(
    name = "bytecode_type_inc_gen",
    strip_include_prefix = "bytecode_inc",
    tbl_outs = {
        "bytecode_inc/TypeBytecode.inc": ["-gen-cuda-tile-type-bytecode"],
        "bytecode_inc/TypeBytecodeReader.inc": ["-gen-cuda-tile-type-bytecode-reader"],
    },
    tblgen = ":cuda-tile-tblgen",
    td_file = "include/cuda_tile/Dialect/CudaTile/IR/BytecodeTypeOpcodes.td",
    deps = [":cuda_tile_td_files"],
)

gentbl_cc_library(
    name = "bytecode_attr_inc_gen",
    strip_include_prefix = "bytecode_inc",
    tbl_outs = {"bytecode_inc/AttrBytecode.inc": ["-gen-cuda-tile-attr-bytecode"]},
    tblgen = ":cuda-tile-tblgen",
    td_file = "include/cuda_tile/Dialect/CudaTile/IR/BytecodeAttrOpcodes.td",
    deps = [":cuda_tile_td_files"],
)

cc_library(
    name = "dialect",
    srcs = [
        "lib/Dialect/CudaTile/IR/Attributes.cpp",
        "lib/Dialect/CudaTile/IR/CudaTile.cpp",
        "lib/Dialect/CudaTile/IR/Interfaces.cpp",
        "lib/Dialect/CudaTile/IR/Traits.cpp",
        "lib/Dialect/CudaTile/IR/Types.cpp",
    ],
    hdrs = glob(["include/cuda_tile/Dialect/CudaTile/IR/*.h"]),
    strip_include_prefix = "include",
    deps = [
        ":attrdefs_inc_gen",
        ":dialect_inc_gen",
        ":interfaces_inc_gen",
        ":ops_canonicalization_inc_gen",
        ":ops_inc_gen",
        ":remarks_inc_gen",
        ":types_inc_gen",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:BytecodeOpInterface",
        "@llvm-project//mlir:ControlFlowInterfaces",
        "@llvm-project//mlir:FunctionInterfaces",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:InferTypeOpInterface",
        "@llvm-project//mlir:InliningUtils",
        "@llvm-project//mlir:SideEffectInterfaces",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:ViewLikeInterface",
    ],
)

cc_library(
    name = "bytecode_common",
    srcs = [
        "lib/Bytecode/Common/CommandLineOptions.cpp",
        "lib/Bytecode/Common/Version.cpp",
        "lib/Bytecode/Common/VersionUtils.h",
    ],
    hdrs = [
        "include/cuda_tile/Bytecode/Common/CommandLineOptions.h",
        "include/cuda_tile/Bytecode/Common/Version.h",
    ],
    strip_include_prefix = "include",
    deps = [
        ":bytecode_opcodes_inc_gen",
        ":dialect",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "bytecode_writer",
    srcs = [
        "lib/Bytecode/BytecodeEnums.h",
        "lib/Bytecode/Common/VersionUtils.h",
        "lib/Bytecode/Writer/BytecodeWriter.cpp",
    ],
    hdrs = ["include/cuda_tile/Bytecode/Writer/BytecodeWriter.h"],
    strip_include_prefix = "include",
    deps = [
        ":bytecode_attr_inc_gen",
        ":bytecode_common",
        ":bytecode_opcodes_inc_gen",
        ":bytecode_ops_inc_gen",
        ":bytecode_type_inc_gen",
        ":dialect",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "bytecode_reader",
    srcs = [
        "lib/Bytecode/BytecodeEnums.h",
        "lib/Bytecode/Common/VersionUtils.h",
        "lib/Bytecode/Reader/BytecodeReader.cpp",
    ],
    hdrs = ["include/cuda_tile/Bytecode/Reader/BytecodeReader.h"],
    strip_include_prefix = "include",
    deps = [
        ":bytecode_attr_inc_gen",
        ":bytecode_common",
        ":bytecode_opcodes_inc_gen",
        ":bytecode_ops_inc_gen",
        ":bytecode_type_inc_gen",
        ":dialect",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:Support",
    ],
)
