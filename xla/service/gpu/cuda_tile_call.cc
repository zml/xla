#include "xla/service/gpu/cuda_tile_call.h"

#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/AsmParser/AsmParser.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/LLVM.h"
#include "xla/mlir/utils/error_util.h"
#include "xla/tsl/platform/statusor.h"

namespace xla::gpu {
namespace {

// The documented get_tile_block_id limit, per axis.
constexpr int64_t kMaxGridAxis = (int64_t{1} << 24) - 1;

// Keys the PTX and Triton configs carry that have no Tile IR meaning; a caller
// who copies one of those schemas is told, not silently mis-served.
constexpr absl::string_view kRejectedKeys[] = {
    "block_x",        "block_y",       "block_z",
    "shared_mem_bytes", "num_warps",   "num_stages",
    "num_ctas",       "is_tma_allowed", "global_scratch_memory_size",
    "zeroed_outputs", "zeroed_args",   "cluster_x",
    "cluster_y",      "cluster_z",     "kernel_data",
};

llvm::StringRef ToStringRef(absl::string_view s) {
  return llvm::StringRef(s.data(), s.size());
}

absl::StatusOr<std::string> GetRequiredString(mlir::DictionaryAttr attrs,
                                              absl::string_view key) {
  auto attr = attrs.getAs<mlir::StringAttr>(ToStringRef(key));
  if (!attr) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Missing required string field '", key, "' in backend_config"));
  }
  return attr.getValue().str();
}

absl::StatusOr<int32_t> GetGridAxis(mlir::DictionaryAttr attrs,
                                    absl::string_view key) {
  auto attr = attrs.getAs<mlir::IntegerAttr>(ToStringRef(key));
  if (!attr) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Missing required integer field '", key, "' in backend_config"));
  }
  // getSExtValue asserts on an attribute wider than 64 bits.
  std::optional<int64_t> maybe_value = attr.getValue().trySExtValue();
  if (!maybe_value.has_value()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "backend_config field '", key, "' does not fit in 64 bits"));
  }
  int64_t value = *maybe_value;
  if (value < 1 || value > kMaxGridAxis) {
    return absl::InvalidArgumentError(
        absl::StrCat("backend_config field '", key, "' is ", value,
                     " but a Tile IR grid axis must be in [1, ", kMaxGridAxis,
                     "]"));
  }
  return static_cast<int32_t>(value);
}

absl::Status ParseIrVersion(absl::string_view text, uint8_t* major,
                            uint8_t* minor) {
  std::vector<absl::string_view> parts = absl::StrSplit(text, '.');
  int major_value = 0;
  int minor_value = 0;
  if (parts.size() != 2 || !absl::SimpleAtoi(parts[0], &major_value) ||
      !absl::SimpleAtoi(parts[1], &minor_value) || major_value < 0 ||
      major_value > 255 || minor_value < 0 || minor_value > 255) {
    return absl::InvalidArgumentError(absl::StrCat(
        "backend_config field 'ir_version' must be \"MAJOR.MINOR\", got \"",
        text, "\""));
  }
  *major = static_cast<uint8_t>(major_value);
  *minor = static_cast<uint8_t>(minor_value);
  return absl::OkStatus();
}

}  // namespace

absl::StatusOr<CudaTileCall> CudaTileCall::Parse(
    absl::string_view backend_config, mlir::MLIRContext* mlir_context) {
  if (backend_config.empty()) {
    return absl::InvalidArgumentError(
        "CUDA Tile IR custom call backend_config is empty");
  }
  mlir::DictionaryAttr attrs;
  {
    // Keeps the parser's diagnostic in the status instead of on stderr.
    mlir::BaseScopedDiagnosticHandler diagnostics(mlir_context);
    attrs = mlir::dyn_cast_or_null<mlir::DictionaryAttr>(
        mlir::parseAttribute(ToStringRef(backend_config), mlir_context));
    absl::Status parse_status = diagnostics.ConsumeStatus();
    if (!attrs) {
      return absl::InvalidArgumentError(absl::StrCat(
          "CUDA Tile IR custom call backend_config is not an MLIR dictionary "
          "attribute",
          parse_status.ok() ? "" : ": ", parse_status.message()));
    }
  }

  for (absl::string_view key : kRejectedKeys) {
    if (attrs.contains(ToStringRef(key))) {
      return absl::InvalidArgumentError(absl::StrCat(
          "backend_config field '", key,
          "' has no meaning for a CUDA Tile IR kernel: the launch is grid-only "
          "with a (1,1,1) block and no dynamic shared memory, and warp/CTA "
          "tuning is expressed as optimization_hints on the cuda_tile.entry"));
    }
  }

  CudaTileCall call;
  ABSL_ASSIGN_OR_RETURN(call.name, GetRequiredString(attrs, "name"));

  ABSL_ASSIGN_OR_RETURN(std::string kernel_type,
                        GetRequiredString(attrs, "kernel_type"));
  if (kernel_type != "cuda_tile") {
    return absl::InvalidArgumentError(absl::StrCat(
        "backend_config field 'kernel_type' is \"", kernel_type,
        "\"; only \"cuda_tile\" (textual cuda_tile MLIR) is supported"));
  }

  ABSL_ASSIGN_OR_RETURN(call.ir, GetRequiredString(attrs, "ir"));
  ABSL_ASSIGN_OR_RETURN(call.grid_x, GetGridAxis(attrs, "grid_x"));
  ABSL_ASSIGN_OR_RETURN(call.grid_y, GetGridAxis(attrs, "grid_y"));
  ABSL_ASSIGN_OR_RETURN(call.grid_z, GetGridAxis(attrs, "grid_z"));

  if (mlir::Attribute version = attrs.get("ir_version")) {
    auto text = mlir::dyn_cast<mlir::StringAttr>(version);
    if (!text) {
      return absl::InvalidArgumentError(
          "backend_config field 'ir_version' must be a \"MAJOR.MINOR\" string");
    }
    ABSL_RETURN_IF_ERROR(ParseIrVersion(
        absl::string_view(text.getValue().data(), text.getValue().size()),
        &call.bytecode_major, &call.bytecode_minor));
  }

  if (mlir::Attribute raw = attrs.get("output_indices")) {
    auto output_indices = mlir::dyn_cast<mlir::ArrayAttr>(raw);
    if (!output_indices) {
      return absl::InvalidArgumentError(
          "backend_config field 'output_indices' must be an array of "
          "integers");
    }
    for (const mlir::Attribute& index : output_indices) {
      auto int_attr = mlir::dyn_cast<mlir::IntegerAttr>(index);
      if (!int_attr) {
        return absl::InvalidArgumentError(
            "Invalid output_indices: all elements must be integers");
      }
      std::optional<int64_t> value = int_attr.getValue().trySExtValue();
      if (!value.has_value() || *value < 0 ||
          *value > std::numeric_limits<int32_t>::max()) {
        return absl::InvalidArgumentError(
            "Invalid output_indices: every element must be a non-negative "
            "32-bit position");
      }
      call.output_indices.push_back(static_cast<int32_t>(*value));
    }
  }

  return call;
}

std::string CudaTileKernelFingerprint(absl::string_view backend_config,
                                      int num_args) {
  return absl::StrCat(backend_config, "|nargs=", num_args);
}

}  // namespace xla::gpu
