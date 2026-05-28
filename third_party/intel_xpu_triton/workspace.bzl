"""Local overlay for Intel's XPU Triton backend."""

_BUILD_FILE_CONTENT = r"""
load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library", "td_library")

package(
    default_visibility = ["//visibility:public"],
    features = [
        "-parse_headers",
        "-use_header_modules",
    ],
)

td_library(
    name = "td_files",
    srcs = glob([
        "third_party/intel/include/**/*.td",
        "third_party/intel/lib/**/*.td",
    ]),
    includes = ["third_party"],
    deps = [
        "@llvm-project//mlir:ArithOpsTdFiles",
        "@llvm-project//mlir:ControlFlowInterfacesTdFiles",
        "@llvm-project//mlir:FunctionInterfacesTdFiles",
        "@llvm-project//mlir:InferTypeOpInterfaceTdFiles",
        "@llvm-project//mlir:LLVMOpsTdFiles",
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:PassBaseTdFiles",
        "@llvm-project//mlir:SideEffectInterfacesTdFiles",
        "@triton//:td_files",
    ],
)

gentbl_cc_library(
    name = "triton_gen_dialect_inc_gen",
    tbl_outs = {
        "third_party/intel/include/Dialect/TritonGEN/IR/TritonGENDialect.h.inc": [
            "--gen-dialect-decls",
            "--dialect=triton_gen",
        ],
        "third_party/intel/include/Dialect/TritonGEN/IR/TritonGENDialect.cpp.inc": [
            "--gen-dialect-defs",
            "--dialect=triton_gen",
        ],
    },
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "third_party/intel/include/Dialect/TritonGEN/IR/TritonGENDialect.td",
    deps = [":td_files"],
)

gentbl_cc_library(
    name = "triton_gen_attr_inc_gen",
    tbl_outs = {
        "third_party/intel/include/Dialect/TritonGEN/IR/TritonGENOpsAttrDefs.h.inc": ["--gen-attrdef-decls"],
        "third_party/intel/include/Dialect/TritonGEN/IR/TritonGENOpsAttrDefs.cpp.inc": ["--gen-attrdef-defs"],
        "third_party/intel/include/Dialect/TritonGEN/IR/TritonGENOpsEnums.h.inc": ["--gen-enum-decls"],
        "third_party/intel/include/Dialect/TritonGEN/IR/TritonGENOpsEnums.cpp.inc": ["--gen-enum-defs"],
    },
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "third_party/intel/include/Dialect/TritonGEN/IR/TritonGENAttrDefs.td",
    deps = [":td_files"],
)

gentbl_cc_library(
    name = "triton_gen_ops_inc_gen",
    tbl_outs = {
        "third_party/intel/include/Dialect/TritonGEN/IR/TritonGENOps.h.inc": ["--gen-op-decls"],
        "third_party/intel/include/Dialect/TritonGEN/IR/TritonGENOps.cpp.inc": ["--gen-op-defs"],
    },
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "third_party/intel/include/Dialect/TritonGEN/IR/TritonGENOps.td",
    deps = [":td_files"],
)

gentbl_cc_library(
    name = "triton_intel_gpu_dialect_inc_gen",
    tbl_outs = {
        "third_party/intel/include/Dialect/TritonIntelGPU/IR/Dialect.h.inc": [
            "--gen-dialect-decls",
            "--dialect=ttig",
        ],
        "third_party/intel/include/Dialect/TritonIntelGPU/IR/Dialect.cpp.inc": [
            "--gen-dialect-defs",
            "--dialect=ttig",
        ],
    },
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "third_party/intel/include/Dialect/TritonIntelGPU/IR/TritonIntelGPUDialect.td",
    deps = [":td_files"],
)

gentbl_cc_library(
    name = "triton_intel_gpu_attr_inc_gen",
    tbl_outs = {
        "third_party/intel/include/Dialect/TritonIntelGPU/IR/TritonIntelGPUAttrDefs.h.inc": [
            "--gen-attrdef-decls",
            "--attrdefs-dialect=ttig",
        ],
        "third_party/intel/include/Dialect/TritonIntelGPU/IR/TritonIntelGPUAttrDefs.cpp.inc": [
            "--gen-attrdef-defs",
            "--attrdefs-dialect=ttig",
        ],
        "third_party/intel/include/Dialect/TritonIntelGPU/IR/TritonIntelGPUEnums.h.inc": ["--gen-enum-decls"],
        "third_party/intel/include/Dialect/TritonIntelGPU/IR/TritonIntelGPUEnums.cpp.inc": ["--gen-enum-defs"],
    },
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "third_party/intel/include/Dialect/TritonIntelGPU/IR/TritonIntelGPUAttrDefs.td",
    deps = [":td_files"],
)

gentbl_cc_library(
    name = "triton_intel_gpu_ops_inc_gen",
    tbl_outs = {
        "third_party/intel/include/Dialect/TritonIntelGPU/IR/Ops.h.inc": ["--gen-op-decls"],
        "third_party/intel/include/Dialect/TritonIntelGPU/IR/Ops.cpp.inc": ["--gen-op-defs"],
    },
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "third_party/intel/include/Dialect/TritonIntelGPU/IR/TritonIntelGPUOps.td",
    deps = [":td_files"],
)

gentbl_cc_library(
    name = "triton_intel_transforms_inc_gen",
    tbl_outs = {"third_party/intel/include/Dialect/Triton/Transforms/Passes.h.inc": [
        "--gen-pass-decls",
        "--name=TritonIntelTransforms",
    ]},
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "third_party/intel/include/Dialect/Triton/Transforms/Passes.td",
    deps = [":td_files"],
)

gentbl_cc_library(
    name = "triton_intel_gpu_transforms_inc_gen",
    tbl_outs = {"third_party/intel/include/Dialect/TritonIntelGPU/Transforms/Passes.h.inc": [
        "--gen-pass-decls",
        "--name=TritonIntelGPU",
    ]},
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "third_party/intel/include/Dialect/TritonIntelGPU/Transforms/Passes.td",
    deps = [":td_files"],
)

gentbl_cc_library(
    name = "triton_annotate_module_pass_inc_gen",
    tbl_outs = {"third_party/intel/include/TritonAnnotateModule/Passes.h.inc": [
        "--gen-pass-decls",
        "--name=TritonAnnotateModule",
    ]},
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "third_party/intel/include/TritonAnnotateModule/Passes.td",
    deps = [":td_files"],
)

gentbl_cc_library(
    name = "gpu_to_triton_gen_pass_inc_gen",
    tbl_outs = {"third_party/intel/include/GPUToTritonGEN/Passes.h.inc": [
        "--gen-pass-decls",
        "--name=GPUToTritonGEN",
    ]},
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "third_party/intel/include/GPUToTritonGEN/Passes.td",
    deps = [":td_files"],
)

gentbl_cc_library(
    name = "gpu_to_triton_gen_rewriters_inc_gen",
    tbl_outs = {"third_party/intel/lib/GPUToTritonGEN/GPUToTritonGEN.cpp.inc": ["--gen-rewriters"]},
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "third_party/intel/lib/GPUToTritonGEN/GPUToTritonGEN.td",
    deps = [":td_files"],
)

gentbl_cc_library(
    name = "triton_gen_to_llvm_pass_inc_gen",
    tbl_outs = {"third_party/intel/include/TritonGENToLLVM/Passes.h.inc": [
        "--gen-pass-decls",
        "--name=TritonGENToLLVM",
    ]},
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "third_party/intel/include/TritonGENToLLVM/Passes.td",
    deps = [":td_files"],
)

gentbl_cc_library(
    name = "triton_gen_to_spirv_pass_inc_gen",
    tbl_outs = {"third_party/intel/include/TritonGENToSPIRV/Passes.h.inc": [
        "--gen-pass-decls",
        "--name=TritonGENToSPIRV",
    ]},
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "third_party/intel/include/TritonGENToSPIRV/Passes.td",
    deps = [":td_files"],
)

gentbl_cc_library(
    name = "triton_intel_gpu_to_llvm_pass_inc_gen",
    tbl_outs = {"third_party/intel/include/TritonIntelGPUToLLVM/Passes.h.inc": [
        "--gen-pass-decls",
        "--name=TritonIntelGPUToLLVM",
    ]},
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "third_party/intel/include/TritonIntelGPUToLLVM/Passes.td",
    deps = [":td_files"],
)

_intel_includes = [
    "third_party",
    "third_party/intel/include",
    "third_party/intel/lib",
    "third_party/intel/lib/GPUToTritonGEN",
    "third_party/intel/lib/TritonIntelGPUToLLVM",
    "third_party/intel/lib/TritonIntelGPUTransforms",
]

_intel_copts = [
    "-DgetScratchSizeInBytesOld=getScratchSizeInBytes",
    "-Wno-ctad-maybe-unsupported",
    "-Wno-deprecated-declarations",
    "-Wno-implicit-fallthrough",
    "-Wno-logical-op-parentheses",
    "-Wno-reorder-ctor",
    "-Wno-return-type",
    "-Wno-unused-but-set-parameter",
    "-Wno-unused-variable",
]

_mlir_common_deps = [
    "@llvm-project//llvm:Core",
    "@llvm-project//llvm:Support",
    "@llvm-project//mlir:Analysis",
    "@llvm-project//mlir:ArithDialect",
    "@llvm-project//mlir:ControlFlowDialect",
    "@llvm-project//mlir:FuncDialect",
    "@llvm-project//mlir:GPUDialect",
    "@llvm-project//mlir:IR",
    "@llvm-project//mlir:LLVMDialect",
    "@llvm-project//mlir:MathDialect",
    "@llvm-project//mlir:Pass",
    "@llvm-project//mlir:SCFDialect",
    "@llvm-project//mlir:SideEffectInterfaces",
    "@llvm-project//mlir:SPIRVDialect",
    "@llvm-project//mlir:Support",
    "@llvm-project//mlir:TensorDialect",
    "@llvm-project//mlir:TransformUtils",
    "@llvm-project//mlir:Transforms",
]

cc_library(
    name = "TritonGENIR",
    copts = _intel_copts,
    includes = _intel_includes,
    srcs = glob(["third_party/intel/lib/Dialect/TritonGEN/IR/*.cpp"]),
    hdrs = glob(["third_party/intel/include/Dialect/TritonGEN/IR/*.h"]),
    deps = _mlir_common_deps + [
        ":triton_gen_attr_inc_gen",
        ":triton_gen_dialect_inc_gen",
        ":triton_gen_ops_inc_gen",
        ":triton_intel_gpu_attr_inc_gen",
        ":triton_intel_gpu_dialect_inc_gen",
        ":triton_intel_gpu_ops_inc_gen",
        "@triton//:TritonDialects",
    ],
)

cc_library(
    name = "TritonIntelGPUIR",
    copts = _intel_copts,
    includes = _intel_includes,
    srcs = glob(["third_party/intel/lib/Dialect/TritonIntelGPU/IR/*.cpp"]),
    hdrs = glob(["third_party/intel/include/Dialect/TritonIntelGPU/IR/*.h"]),
    deps = _mlir_common_deps + [
        ":TritonGENIR",
        ":triton_intel_gpu_attr_inc_gen",
        ":triton_intel_gpu_dialect_inc_gen",
        ":triton_intel_gpu_ops_inc_gen",
        "@triton//:TritonDialects",
    ],
)

cc_library(
    name = "TritonIntelAnalysis",
    copts = _intel_copts,
    includes = _intel_includes,
    srcs = glob(["third_party/intel/lib/Analysis/*.cpp"]),
    hdrs = glob(["third_party/intel/include/Analysis/*.h"]),
    deps = _mlir_common_deps + [
        ":TritonIntelGPUIR",
        "@triton//:TritonDialects",
        "@triton//:TritonGPUTransforms",
    ],
)

cc_library(
    name = "TritonIntelUtils",
    copts = _intel_copts,
    includes = _intel_includes,
    srcs = glob(["third_party/intel/lib/Utils/*.cpp"]),
    hdrs = glob(["third_party/intel/include/Utils/*.h"]),
    deps = _mlir_common_deps + [
        ":TritonGENIR",
        ":TritonIntelGPUIR",
        "@llvm-project//mlir:LLVMCommonConversion",
        "@triton//:TritonDialects",
    ],
)

cc_library(
    name = "TritonIntelTransforms",
    copts = _intel_copts,
    includes = _intel_includes,
    srcs = glob(["third_party/intel/lib/Dialect/Triton/Transforms/*.cpp"]),
    hdrs = glob(["third_party/intel/include/Dialect/Triton/Transforms/*.h"]),
    deps = _mlir_common_deps + [
        ":TritonIntelAnalysis",
        ":triton_intel_transforms_inc_gen",
        "@triton//:TritonDialects",
    ],
)

cc_library(
    name = "TritonAnnotateModule",
    copts = _intel_copts,
    includes = _intel_includes,
    srcs = ["third_party/intel/lib/TritonAnnotateModule/TritonAnnotateModule.cpp"],
    hdrs = glob(["third_party/intel/include/TritonAnnotateModule/*.h"]),
    deps = _mlir_common_deps + [
        ":TritonIntelGPUIR",
        ":triton_annotate_module_pass_inc_gen",
        "@triton//:TritonDialects",
    ],
)

cc_library(
    name = "GPUToTritonGEN",
    copts = _intel_copts,
    includes = _intel_includes,
    srcs = ["third_party/intel/lib/GPUToTritonGEN/GPUToTritonGENPass.cpp"],
    hdrs = glob(["third_party/intel/include/GPUToTritonGEN/*.h"]),
    deps = _mlir_common_deps + [
        ":TritonGENIR",
        ":gpu_to_triton_gen_pass_inc_gen",
        ":gpu_to_triton_gen_rewriters_inc_gen",
        "@llvm-project//mlir:FuncToLLVM",
        "@llvm-project//mlir:GPUToGPURuntimeTransforms",
        "@llvm-project//mlir:LLVMCommonConversion",
        "@llvm-project//mlir:MemRefToLLVM",
    ],
)

cc_library(
    name = "TritonGENToLLVM",
    copts = _intel_copts,
    includes = _intel_includes,
    srcs = glob(["third_party/intel/lib/TritonGENToLLVM/*.cpp"]),
    hdrs = glob(["third_party/intel/include/TritonGENToLLVM/*.h"]),
    deps = _mlir_common_deps + [
        ":TritonGENIR",
        ":TritonIntelUtils",
        ":triton_gen_to_llvm_pass_inc_gen",
        ":triton_gen_to_spirv_pass_inc_gen",
    ],
)

cc_library(
    name = "TritonGENToSPIRV",
    copts = _intel_copts,
    includes = _intel_includes,
    srcs = glob(["third_party/intel/lib/TritonGENToSPIRV/*.cpp"]),
    hdrs = glob(["third_party/intel/include/TritonGENToSPIRV/*.h"]),
    deps = _mlir_common_deps + [
        ":TritonGENIR",
        ":triton_gen_to_spirv_pass_inc_gen",
    ],
)

cc_library(
    name = "TritonIntelXeAsm",
    copts = _intel_copts,
    includes = _intel_includes,
    srcs = ["third_party/intel/lib/TritonIntelGPUToLLVM/XeAsmFormat.cpp"],
    hdrs = ["third_party/intel/include/TritonIntelGPUToLLVM/XeAsmFormat.h"],
    deps = _mlir_common_deps + ["@triton//:TritonDialects"],
)

cc_library(
    name = "TritonIntelGPUTransforms",
    copts = _intel_copts,
    includes = _intel_includes,
    srcs = glob([
        "third_party/intel/lib/TritonIntelGPUTransforms/*.cpp",
        "third_party/intel/lib/TritonIntelGPUTransforms/Pipeliner/*.cpp",
    ]),
    hdrs = glob([
        "third_party/intel/include/Dialect/TritonIntelGPU/Transforms/*.h",
        "third_party/intel/lib/TritonIntelGPUTransforms/*.h",
        "third_party/intel/lib/TritonIntelGPUTransforms/Pipeliner/*.h",
    ]),
    textual_hdrs = glob(["include/triton/**/*.h"]),
    deps = _mlir_common_deps + [
        ":TritonGENIR",
        ":TritonIntelAnalysis",
        ":TritonIntelGPUIR",
        ":TritonIntelUtils",
        ":triton_intel_gpu_transforms_inc_gen",
        "@llvm-project//mlir:SCFTransforms",
        "@triton//:TritonDialects",
        "@triton//:TritonGPUTransforms",
        "@triton//:TritonTransforms",
    ],
)

genrule(
    name = "intel_reduce_op_to_llvm_stub_gen",
    outs = ["third_party/intel/lib/TritonIntelGPUToLLVM/ReduceOpToLLVMStub.cpp"],
    cmd = "cat > $@ <<'EOF'\n" +
          "#include \"PatternTritonGPUOpToLLVM.h\"\n" +
          "namespace mlir::triton::intel {\n" +
          "void populateReduceOpToLLVMPatterns(LLVMTypeConverter&, RewritePatternSet&, const TargetInfoBase&, PatternBenefit) {}\n" +
          "}  // namespace mlir::triton::intel\n" +
          "EOF\n",
)

cc_library(
    name = "TritonIntelGPUToLLVM",
    copts = _intel_copts + ["-Iexternal/intel_xpu_triton/include"],
    includes = _intel_includes,
    srcs = glob([
        "third_party/intel/lib/TritonIntelGPUToLLVM/*.cpp",
        "third_party/intel/lib/TritonIntelGPUToLLVM/DotOpToLLVM/*.cpp",
    ], exclude = [
        "third_party/intel/lib/TritonIntelGPUToLLVM/ReduceOpToLLVM.cpp",
        "third_party/intel/lib/TritonIntelGPUToLLVM/XeAsmFormat.cpp",
    ]) + [":intel_reduce_op_to_llvm_stub_gen"],
    hdrs = glob(["third_party/intel/include/TritonIntelGPUToLLVM/*.h"]),
    textual_hdrs = glob(["include/triton/**/*.h"]),
    deps = _mlir_common_deps + [
        ":GPUToTritonGEN",
        ":TritonGENIR",
        ":TritonGENToLLVM",
        ":TritonGENToSPIRV",
        ":TritonIntelGPUIR",
        ":TritonIntelUtils",
        ":TritonIntelXeAsm",
        ":triton_intel_gpu_to_llvm_pass_inc_gen",
        "@llvm-project//mlir:ArithToLLVM",
        "@llvm-project//mlir:ControlFlowToLLVM",
        "@llvm-project//mlir:GPUToLLVMSPVTransforms",
        "@llvm-project//mlir:IndexToLLVM",
        "@llvm-project//mlir:MathToLLVM",
        "@llvm-project//mlir:SCFToControlFlow",
        "@llvm-project//mlir:SPIRVToLLVM",
        "@llvm-project//mlir:UBToLLVM",
        "@triton//:TritonDialects",
        "@triton//:TritonGPUToLLVM",
        "@triton//:TritonInstrumentTransforms",
    ],
)

cc_library(
    name = "TritonIntelLLVMIR",
    copts = _intel_copts,
    includes = _intel_includes,
    srcs = glob(["third_party/intel/lib/Target/LLVMIR/*.cpp"]),
    hdrs = glob(["third_party/intel/include/Target/LLVMIR/*.h"]),
    deps = _mlir_common_deps + [
        "@llvm-project//llvm:Analysis",
        "@llvm-project//llvm:InstCombine",
        "@llvm-project//llvm:Passes",
        "@llvm-project//llvm:TransformUtils",
    ],
)

cc_library(
    name = "TritonGENToLLVMIRTranslation",
    copts = _intel_copts,
    includes = _intel_includes,
    srcs = ["third_party/intel/lib/Target/LLVMIR/Dialect/TritonGEN/TritonGENToLLVMIRTranslation.cpp"],
    hdrs = glob(["third_party/intel/include/Target/LLVMIR/Dialect/TritonGEN/*.h"]),
    deps = _mlir_common_deps + [
        ":TritonGENIR",
        "@llvm-project//mlir:ToLLVMIRTranslation",
    ],
)

cc_library(
    name = "TritonSPIRV",
    copts = _intel_copts,
    includes = _intel_includes,
    # XLA lowers Triton-generated LLVM with its existing IntelGpuCompiler SPIR-V
    # path. Intel's standalone translator currently requires newer
    # SPIRV-LLVM-Translator extension enums than XLA pins, so keep this target
    # as a compatibility placeholder for now.
    srcs = [],
    hdrs = glob(["third_party/intel/include/Target/SPIRV/*.h"]),
    deps = _mlir_common_deps + [":TritonIntelLLVMIR"],
)

cc_library(
    name = "TritonXPUBackend",
    copts = _intel_copts,
    includes = _intel_includes,
    deps = _mlir_common_deps + [
        ":GPUToTritonGEN",
        ":TritonAnnotateModule",
        ":TritonGENIR",
        ":TritonGENToLLVM",
        ":TritonGENToLLVMIRTranslation",
        ":TritonGENToSPIRV",
        ":TritonIntelAnalysis",
        ":TritonIntelGPUIR",
        ":TritonIntelGPUToLLVM",
        ":TritonIntelGPUTransforms",
        ":TritonIntelLLVMIR",
        ":TritonIntelTransforms",
        ":TritonIntelUtils",
        ":TritonIntelXeAsm",
        ":TritonSPIRV",
    ],
)
"""

def repo():
    native.new_local_repository(
        name = "intel_xpu_triton",
        path = "/home/steeve/intel-xpu-backend-for-triton",
        build_file_content = _BUILD_FILE_CONTENT,
    )
