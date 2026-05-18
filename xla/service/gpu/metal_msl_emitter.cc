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

#include "xla/service/gpu/metal_msl_emitter.h"

#include <algorithm>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Operator.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"
#include "xla/tsl/platform/status_macros.h"

namespace xla {
namespace gpu {
namespace {

constexpr size_t kMaxDirectMetalBufferArguments = 31;

struct BlockStopKey {
  const llvm::BasicBlock* block;
  const llvm::BasicBlock* stop;

  template <typename H>
  friend H AbslHashValue(H h, const BlockStopKey& key) {
    return H::combine(std::move(h), key.block, key.stop);
  }

  bool operator==(const BlockStopKey& other) const {
    return block == other.block && stop == other.stop;
  }
};

std::string PrintLlvm(const llvm::Value& value) {
  std::string result;
  llvm::raw_string_ostream os(result);
  value.print(os);
  return os.str();
}

std::string PrintLlvmType(const llvm::Type& type) {
  std::string result;
  llvm::raw_string_ostream os(result);
  type.print(os);
  return os.str();
}

std::string Indent(int depth) { return std::string(depth * 2, ' '); }

std::string WideVectorStructDefinitions() {
  std::string output;
  for (int elements = 5; elements <= 16; ++elements) {
    absl::StrAppend(&output, "struct xla_metal_vec", elements,
                    "_char { char elements[", elements, "]; };\n");
    absl::StrAppend(&output, "struct xla_metal_vec", elements,
                    "_uchar { uchar elements[", elements, "]; };\n");
  }
  return output;
}

bool IsPointerType(const llvm::Value& value) {
  return value.getType()->isPointerTy();
}

bool IsAggregateType(llvm::Type* type) {
  return type->isStructTy() || type->isArrayTy();
}

bool IsComplexFloatStruct(llvm::Type* type) {
  auto* struct_type = llvm::dyn_cast<llvm::StructType>(type);
  return struct_type != nullptr && struct_type->getNumElements() == 2 &&
         struct_type->getElementType(0)->isFloatTy() &&
         struct_type->getElementType(1)->isFloatTy();
}

bool IsNonMaterializedAggregateType(llvm::Type* type) {
  return IsAggregateType(type) && !IsComplexFloatStruct(type);
}

absl::Status Unsupported(const llvm::Function& function, absl::string_view what,
                         const llvm::Value& value) {
  return absl::UnimplementedError(absl::StrCat(
      "Metal MSL emission for kernel '", function.getName().str(),
      "' does not support ", what, ": ", PrintLlvm(value)));
}

std::string MslIdentifier(absl::string_view name, absl::string_view fallback) {
  std::string result;
  result.reserve(name.size() + fallback.size());
  for (char c : name) {
    if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
        (c >= '0' && c <= '9') || c == '_') {
      result.push_back(c);
    } else {
      result.push_back('_');
    }
  }
  if (result.empty()) result = std::string(fallback);
  if (result[0] >= '0' && result[0] <= '9') result.insert(result.begin(), '_');
  return result;
}

bool IsKernelFunction(const llvm::Function& function) {
  if (function.isDeclaration()) return false;

  const llvm::Module* module = function.getParent();
  if (auto* annotations = module->getNamedMetadata("nvvm.annotations")) {
    for (llvm::MDNode* node : annotations->operands()) {
      if (node->getNumOperands() < 2) continue;
      auto* value_md =
          llvm::dyn_cast<llvm::ValueAsMetadata>(node->getOperand(0));
      if (value_md == nullptr || value_md->getValue() != &function) continue;

      auto* string_md = llvm::dyn_cast<llvm::MDString>(node->getOperand(1));
      if (string_md != nullptr && string_md->getString() == "kernel") {
        return true;
      }
    }
    return function.getLinkage() == llvm::GlobalValue::ExternalLinkage;
  }

  return function.getLinkage() == llvm::GlobalValue::ExternalLinkage;
}

absl::StatusOr<std::string> IntegerLiteral(const llvm::APInt& value,
                                           bool is_signed) {
  if (value.getBitWidth() == 1) {
    return value.isZero() ? "false" : "true";
  }
  llvm::SmallString<32> text;
  value.toString(text, /*Radix=*/10, is_signed);
  return std::string(text.str());
}

absl::StatusOr<std::string> ZeroInitializer(llvm::Type* type);

absl::StatusOr<std::string> MslScalarType(llvm::Type* type,
                                          bool unsigned_integer = false) {
  if (type->isIntegerTy(1)) return "bool";
  if (type->isIntegerTy(8)) return unsigned_integer ? "uchar" : "char";
  if (type->isIntegerTy(16)) return unsigned_integer ? "ushort" : "short";
  if (type->isIntegerTy(32)) return unsigned_integer ? "uint" : "int";
  if (type->isIntegerTy(64)) return unsigned_integer ? "ulong" : "long";
  if (type->isBFloatTy()) return "ushort";
  if (type->isHalfTy()) return "half";
  if (type->isFloatTy()) return "float";
  if (type->isDoubleTy()) return "double";
  return absl::UnimplementedError(
      absl::StrCat("Unsupported LLVM scalar type for MSL: ",
                   PrintLlvmType(*type)));
}

std::optional<std::string> WideVectorTypeName(
    llvm::FixedVectorType* vector_type, bool unsigned_integer = false) {
  int elements = vector_type->getNumElements();
  if (elements <= 4 || elements > 16) return std::nullopt;
  if (!vector_type->getElementType()->isIntegerTy(8)) return std::nullopt;
  return absl::StrCat("xla_metal_vec", elements, "_",
                      unsigned_integer ? "uchar" : "char");
}

absl::StatusOr<std::string> MslType(llvm::Type* type,
                                    bool unsigned_integer = false) {
  if (auto* vector_type = llvm::dyn_cast<llvm::FixedVectorType>(type)) {
    int elements = vector_type->getNumElements();
    if (std::optional<std::string> wide_type =
            WideVectorTypeName(vector_type, unsigned_integer)) {
      return *wide_type;
    }
    if (elements < 2 || elements > 4) {
      return absl::UnimplementedError(absl::StrCat(
          "Metal MSL emission supports vector widths 2, 3, and 4, got ",
          elements, " for ", PrintLlvmType(*type)));
    }
    TF_ASSIGN_OR_RETURN(std::string element_type,
                        MslScalarType(vector_type->getElementType(),
                                      unsigned_integer));
    return absl::StrCat(element_type, elements);
  }
  if (IsComplexFloatStruct(type)) return "float2";
  return MslScalarType(type, unsigned_integer);
}

absl::StatusOr<std::string> ZeroInitializer(llvm::Type* type) {
  if (type->isIntegerTy(1)) return "false";
  if (type->isIntegerTy()) return "0";
  if (type->isBFloatTy()) return "0";
  if (type->isHalfTy()) return "half(0.0)";
  if (type->isFloatTy()) return "0.0f";
  if (type->isDoubleTy()) return "0.0";
  if (auto* vector_type = llvm::dyn_cast<llvm::FixedVectorType>(type)) {
    TF_ASSIGN_OR_RETURN(std::string msl_type, MslType(type));
    if (WideVectorTypeName(vector_type).has_value()) {
      return absl::StrCat(msl_type, "{}");
    }
    TF_ASSIGN_OR_RETURN(std::string zero,
                        ZeroInitializer(vector_type->getElementType()));
    std::vector<std::string> values(vector_type->getNumElements(), zero);
    return absl::StrCat(msl_type, "(", absl::StrJoin(values, ", "), ")");
  }
  if (IsComplexFloatStruct(type)) return "float2(0.0f, 0.0f)";
  return absl::UnimplementedError(
      absl::StrCat("Unsupported zero initializer type for MSL: ",
                   PrintLlvmType(*type)));
}

std::optional<int64_t> FixedTypeSize(const llvm::DataLayout& data_layout,
                                     llvm::Type* type) {
  llvm::TypeSize size = data_layout.getTypeAllocSize(type);
  if (size.isScalable()) return std::nullopt;
  return size.getFixedValue();
}

absl::StatusOr<int64_t> FlattenedElementCount(
    const llvm::DataLayout& data_layout, llvm::Type* type,
    llvm::Type* target_element_type) {
  if (type == target_element_type) return 1;

  if (auto* array_type = llvm::dyn_cast<llvm::ArrayType>(type)) {
    TF_ASSIGN_OR_RETURN(int64_t nested,
                        FlattenedElementCount(data_layout,
                                              array_type->getElementType(),
                                              target_element_type));
    return nested * array_type->getNumElements();
  }

  if (auto* vector_type = llvm::dyn_cast<llvm::FixedVectorType>(type)) {
    TF_ASSIGN_OR_RETURN(int64_t nested,
                        FlattenedElementCount(data_layout,
                                              vector_type->getElementType(),
                                              target_element_type));
    return nested * vector_type->getNumElements();
  }

  if (auto* struct_type = llvm::dyn_cast<llvm::StructType>(type)) {
    int64_t total = 0;
    for (llvm::Type* element_type : struct_type->elements()) {
      TF_ASSIGN_OR_RETURN(
          int64_t nested,
          FlattenedElementCount(data_layout, element_type, target_element_type));
      total += nested;
    }
    return total;
  }

  std::optional<int64_t> type_size = FixedTypeSize(data_layout, type);
  std::optional<int64_t> target_size =
      FixedTypeSize(data_layout, target_element_type);
  if (!type_size.has_value() || !target_size.has_value() ||
      *target_size == 0 || *type_size % *target_size != 0) {
    return absl::UnimplementedError(absl::StrCat(
        "Cannot express GEP type ", PrintLlvmType(*type),
        " as a multiple of ", PrintLlvmType(*target_element_type)));
  }
  return *type_size / *target_size;
}

absl::StatusOr<llvm::Type*> TypeAfterGepIndex(llvm::Type* current_type,
                                              const llvm::Value* index,
                                              bool first_index) {
  if (first_index) return current_type;

  if (auto* array_type = llvm::dyn_cast<llvm::ArrayType>(current_type)) {
    return array_type->getElementType();
  }
  if (auto* vector_type = llvm::dyn_cast<llvm::FixedVectorType>(current_type)) {
    return vector_type->getElementType();
  }
  if (auto* struct_type = llvm::dyn_cast<llvm::StructType>(current_type)) {
    auto* constant_index = llvm::dyn_cast<llvm::ConstantInt>(index);
    if (constant_index == nullptr) {
      return absl::UnimplementedError(
          "Metal MSL emission requires constant struct GEP indices.");
    }
    uint64_t field = constant_index->getZExtValue();
    if (field >= struct_type->getNumElements()) {
      return absl::InvalidArgumentError(
          absl::StrCat("Struct GEP field index ", field, " is out of range."));
    }
    return struct_type->getElementType(field);
  }
  return absl::UnimplementedError(
      absl::StrCat("Cannot index into LLVM type: ",
                   PrintLlvmType(*current_type)));
}

absl::StatusOr<llvm::Type*> TypeScaledByGepIndex(llvm::Type* current_type,
                                                 bool first_index) {
  if (first_index) return current_type;

  if (auto* array_type = llvm::dyn_cast<llvm::ArrayType>(current_type)) {
    return array_type->getElementType();
  }
  if (auto* vector_type = llvm::dyn_cast<llvm::FixedVectorType>(current_type)) {
    return vector_type->getElementType();
  }
  if (llvm::isa<llvm::StructType>(current_type)) {
    return absl::InvalidArgumentError(
        "Struct GEP indices are lowered through field offsets.");
  }
  return absl::UnimplementedError(
      absl::StrCat("Cannot index into LLVM type: ",
                   PrintLlvmType(*current_type)));
}

template <typename GepT>
absl::StatusOr<llvm::Type*> GepResultElementType(const GepT& gep) {
  llvm::Type* current_type = gep.getSourceElementType();
  bool first_index = true;
  for (const llvm::Use& use : gep.indices()) {
    TF_ASSIGN_OR_RETURN(
        current_type,
        TypeAfterGepIndex(current_type, use.get(), first_index));
    first_index = false;
  }
  return current_type;
}

bool IsZeroConstant(const llvm::Value* value) {
  auto* constant = llvm::dyn_cast<llvm::ConstantInt>(value);
  return constant != nullptr && constant->isZero();
}

std::string AddIndex(std::string lhs, std::string rhs) {
  if (lhs == "0") return rhs;
  if (rhs == "0") return lhs;
  return absl::StrCat("(", lhs, " + ", rhs, ")");
}

std::string ScaleIndex(std::string index, int64_t scale) {
  if (index == "0" || scale == 0) return "0";
  if (scale == 1) return index;
  return absl::StrCat("(", index, " * ", scale, ")");
}

class FunctionEmitter {
 public:
  explicit FunctionEmitter(const llvm::Function& function)
      : function_(function),
        data_layout_(function.getParent()->getDataLayout()) {}

  absl::StatusOr<std::string> Emit() {
    if (!function_.getReturnType()->isVoidTy()) {
      return absl::UnimplementedError(absl::StrCat(
          "Metal kernel '", function_.getName().str(),
          "' must return void, got ", PrintLlvmType(*function_.getReturnType())));
    }

    InferPointerElementTypes();
    AssignNames();
    TF_ASSIGN_OR_RETURN(std::string kernel_name, KernelName());

    std::string output;
    if (UseArgumentBuffer()) {
      RETURN_IF_ERROR(EmitArgumentBufferStruct(&output, kernel_name));
    }
    RETURN_IF_ERROR(EmitSignature(&output, kernel_name));
    absl::StrAppend(&output, " {\n");
    RETURN_IF_ERROR(EmitArgumentAliases(&output));
    RETURN_IF_ERROR(EmitDeclarations(&output));
    RETURN_IF_ERROR(EmitBody(&output));
    absl::StrAppend(&output, "}\n");
    return output;
  }

 private:
  struct PtrExpr {
    std::string base;
    std::string index;
    llvm::Type* element_type;
    std::string address_space;
  };

  struct StoredInlineValue {
    std::string expression;
    llvm::Type* type;
  };

  void InferPointerElementTypes() {
    for (const llvm::BasicBlock& block : function_) {
      for (const llvm::Instruction& instruction : block) {
        if (auto* gep = llvm::dyn_cast<llvm::GetElementPtrInst>(&instruction)) {
          if (auto element_type = GepResultElementType(*gep);
              element_type.ok()) {
            RecordPointerElementType(gep, *element_type);
          }
        }
      }
    }

    for (const llvm::BasicBlock& block : function_) {
      for (const llvm::Instruction& instruction : block) {
        if (auto* load = llvm::dyn_cast<llvm::LoadInst>(&instruction)) {
          RecordPointerElementType(load->getPointerOperand(), load->getType());
        } else if (auto* store =
                       llvm::dyn_cast<llvm::StoreInst>(&instruction)) {
          RecordPointerElementType(store->getPointerOperand(),
                                   store->getValueOperand()->getType());
        } else if (auto* call = llvm::dyn_cast<llvm::CallInst>(&instruction)) {
          RecordInlinePointerElementTypes(*call);
        }
      }
    }
  }

  void RecordInlinePointerElementTypes(const llvm::CallInst& call) {
    const llvm::Function* callee = call.getCalledFunction();
    if (callee == nullptr || callee->isDeclaration()) return;

    for (const llvm::BasicBlock& block : *callee) {
      for (const llvm::Instruction& instruction : block) {
        if (auto* load = llvm::dyn_cast<llvm::LoadInst>(&instruction)) {
          RecordInlinePointerElementType(call, load->getPointerOperand(),
                                         load->getType());
        } else if (auto* store =
                       llvm::dyn_cast<llvm::StoreInst>(&instruction)) {
          RecordInlinePointerElementType(
              call, store->getPointerOperand(),
              store->getValueOperand()->getType());
        }
      }
    }
  }

  void RecordInlinePointerElementType(const llvm::CallInst& call,
                                      const llvm::Value* pointer,
                                      llvm::Type* element_type) {
    const llvm::Value* root = PointerRoot(pointer);
    if (root == nullptr) return;
    auto* argument = llvm::dyn_cast<llvm::Argument>(root);
    if (argument == nullptr) return;
    RecordPointerElementType(call.getArgOperand(argument->getArgNo()),
                             element_type);
  }

  void RecordPointerElementType(const llvm::Value* pointer,
                                llvm::Type* element_type) {
    const llvm::Value* root = PointerRoot(pointer);
    if (root == nullptr) return;
    auto [it, inserted] = pointer_element_types_.try_emplace(root, element_type);
    if (!inserted && it->second != element_type) {
      if (auto* vector_type =
              llvm::dyn_cast<llvm::FixedVectorType>(it->second);
          vector_type != nullptr &&
          vector_type->getElementType() == element_type) {
        it->second = element_type;
      }
      return;
    }
  }

  const llvm::Value* PointerRoot(const llvm::Value* value) const {
    while (value != nullptr) {
      if (llvm::isa<llvm::Argument>(value)) return value;
      if (auto* op = llvm::dyn_cast<llvm::Operator>(value)) {
        switch (op->getOpcode()) {
          case llvm::Instruction::GetElementPtr:
          case llvm::Instruction::BitCast:
          case llvm::Instruction::AddrSpaceCast:
            value = op->getOperand(0);
            continue;
          default:
            return value;
        }
      }
      return value;
    }
    return nullptr;
  }

  void AssignNames() {
    int arg_index = 0;
    for (const llvm::Argument& arg : function_.args()) {
      names_[&arg] = absl::StrCat("arg", arg_index++);
    }

    int value_index = 0;
    int local_index = 0;
    int cmpxchg_index = 0;
    for (const llvm::BasicBlock& block : function_) {
      for (const llvm::Instruction& instruction : block) {
        if (instruction.getType()->isVoidTy()) continue;
        if (llvm::isa<llvm::AllocaInst>(&instruction)) {
          names_[&instruction] = absl::StrCat("local", local_index++);
          continue;
        }
        if (llvm::isa<llvm::AtomicCmpXchgInst>(&instruction)) {
          cmpxchg_old_names_[&instruction] =
              absl::StrCat("cmpxchg_old", cmpxchg_index);
          cmpxchg_success_names_[&instruction] =
              absl::StrCat("cmpxchg_success", cmpxchg_index);
          ++cmpxchg_index;
          continue;
        }
        if (IsPointerType(instruction)) continue;
        if (IsNonMaterializedAggregateType(instruction.getType())) continue;
        names_[&instruction] = absl::StrCat("v", value_index++);
      }
    }
  }

  std::string Name(const llvm::Value* value) const {
    auto it = names_.find(value);
    if (it != names_.end()) return it->second;
    return MslIdentifier(value->getName().str(), "unnamed");
  }

  bool UseArgumentBuffer() const {
    return function_.arg_size() > kMaxDirectMetalBufferArguments;
  }

  absl::StatusOr<std::string> KernelName() const {
    std::string kernel_name =
        MslIdentifier(function_.getName().str(), "xla_metal_kernel");
    if (kernel_name != function_.getName().str()) {
      return absl::UnimplementedError(absl::StrCat(
          "Metal kernel name '", function_.getName().str(),
          "' is not a valid MSL identifier after XLA sanitization."));
    }
    return kernel_name;
  }

  absl::Status EmitArgumentBufferStruct(std::string* output,
                                        absl::string_view kernel_name) {
    absl::StrAppend(output, "struct ", kernel_name, "_args {\n");
    int arg_index = 0;
    for (const llvm::Argument& arg : function_.args()) {
      std::string name = Name(&arg);
      if (arg.getType()->isPointerTy()) {
        llvm::Type* element_type = PointerElementType(&arg);
        TF_ASSIGN_OR_RETURN(std::string msl_type, MslType(element_type));
        absl::StrAppend(output, "  device ", msl_type, "* ", name, " [[id(",
                        arg_index, ")]];\n");
      } else {
        TF_ASSIGN_OR_RETURN(std::string msl_type, MslType(arg.getType()));
        absl::StrAppend(output, "  constant ", msl_type, "& ", name,
                        " [[id(", arg_index, ")]];\n");
      }
      ++arg_index;
    }
    absl::StrAppend(output, "};\n\n");
    return absl::OkStatus();
  }

  absl::Status EmitSignature(std::string* output,
                             absl::string_view kernel_name) {

    absl::StrAppend(output, "kernel void ", kernel_name, "(");
    std::vector<std::string> params;
    if (UseArgumentBuffer()) {
      params.push_back(
          absl::StrFormat("device %s_args& args [[buffer(0)]]", kernel_name));
    } else {
      int arg_index = 0;
      for (const llvm::Argument& arg : function_.args()) {
        std::string name = Name(&arg);
        if (arg.getType()->isPointerTy()) {
          llvm::Type* element_type = PointerElementType(&arg);
          TF_ASSIGN_OR_RETURN(std::string msl_type, MslType(element_type));
          params.push_back(absl::StrFormat("device %s* %s [[buffer(%d)]]",
                                           msl_type, name, arg_index));
        } else {
          TF_ASSIGN_OR_RETURN(std::string msl_type, MslType(arg.getType()));
          params.push_back(absl::StrFormat("constant %s& %s [[buffer(%d)]]",
                                           msl_type, name, arg_index));
        }
        ++arg_index;
      }
    }
    params.push_back("uint3 metal_tid [[thread_position_in_threadgroup]]");
    params.push_back("uint3 metal_bid [[threadgroup_position_in_grid]]");
    params.push_back("uint3 metal_ntid [[threads_per_threadgroup]]");
    params.push_back("uint3 metal_nbid [[threadgroups_per_grid]]");
    params.push_back("uint3 metal_gid [[thread_position_in_grid]]");
    absl::StrAppend(output, absl::StrJoin(params, ", "), ")");
    return absl::OkStatus();
  }

  absl::Status EmitArgumentAliases(std::string* output) {
    if (!UseArgumentBuffer()) return absl::OkStatus();

    for (const llvm::Argument& arg : function_.args()) {
      std::string name = Name(&arg);
      if (arg.getType()->isPointerTy()) {
        llvm::Type* element_type = PointerElementType(&arg);
        TF_ASSIGN_OR_RETURN(std::string msl_type, MslType(element_type));
        absl::StrAppend(output, "  device ", msl_type, "* ", name,
                        " = args.", name, ";\n");
      } else {
        TF_ASSIGN_OR_RETURN(std::string msl_type, MslType(arg.getType()));
        absl::StrAppend(output, "  ", msl_type, " ", name, " = args.", name,
                        ";\n");
      }
    }
    return absl::OkStatus();
  }

  llvm::Type* PointerElementType(const llvm::Value* pointer) const {
    const llvm::Value* root = PointerRoot(pointer);
    auto it = pointer_element_types_.find(root);
    if (it != pointer_element_types_.end()) return it->second;
    return llvm::Type::getInt8Ty(function_.getContext());
  }

  struct ArrayInfo {
    llvm::Type* element_type;
    uint64_t element_count;
  };

  absl::Status EmitDeclarations(std::string* output) {
    RETURN_IF_ERROR(EmitThreadgroupGlobals(output));
    for (const llvm::BasicBlock& block : function_) {
      for (const llvm::Instruction& instruction : block) {
        if (auto* alloca = llvm::dyn_cast<llvm::AllocaInst>(&instruction)) {
          TF_ASSIGN_OR_RETURN(std::string declaration,
                              AllocaDeclaration(*alloca));
          absl::StrAppend(output, "  ", declaration, "\n");
          continue;
        }
        if (instruction.getType()->isVoidTy()) continue;
        if (IsPointerType(instruction)) continue;
        if (IsNonMaterializedAggregateType(instruction.getType())) continue;
        TF_ASSIGN_OR_RETURN(std::string type, MslType(instruction.getType()));
        TF_ASSIGN_OR_RETURN(std::string zero,
                            ZeroInitializer(instruction.getType()));
        absl::StrAppend(output, "  ", type, " ", Name(&instruction), " = ",
                        zero, ";\n");
      }
    }
    return absl::OkStatus();
  }

  absl::Status EmitThreadgroupGlobals(std::string* output) {
    for (const llvm::GlobalVariable& global : function_.getParent()->globals()) {
      if (global.getAddressSpace() != 3) continue;
      TF_ASSIGN_OR_RETURN(std::string declaration,
                          ThreadgroupGlobalDeclaration(global));
      absl::StrAppend(output, "  ", declaration, "\n");
    }
    return absl::OkStatus();
  }

  absl::StatusOr<std::string> ThreadgroupGlobalDeclaration(
      const llvm::GlobalVariable& global) {
    TF_ASSIGN_OR_RETURN(ArrayInfo array,
                        FlattenedArrayInfo(global.getValueType()));
    TF_ASSIGN_OR_RETURN(std::string type, MslType(array.element_type));
    return absl::StrCat("threadgroup ", type, " ", Name(&global), "[",
                        array.element_count, "];");
  }

  absl::StatusOr<std::string> AllocaDeclaration(
      const llvm::AllocaInst& alloca) {
    auto* constant_count = llvm::dyn_cast<llvm::ConstantInt>(
        alloca.getArraySize());
    if (constant_count == nullptr) {
      return Unsupported(function_, "dynamic stack allocation", alloca);
    }

    uint64_t count = constant_count->getZExtValue();
    if (count == 0) {
      return absl::InvalidArgumentError(
          "Metal MSL emission cannot allocate a zero-sized stack slot.");
    }

    TF_ASSIGN_OR_RETURN(std::string type,
                        MslType(alloca.getAllocatedType()));
    TF_ASSIGN_OR_RETURN(std::string zero,
                        ZeroInitializer(alloca.getAllocatedType()));
    std::vector<std::string> values(count, zero);
    return absl::StrCat(type, " ", Name(&alloca), "[", count, "] = {",
                        absl::StrJoin(values, ", "), "};");
  }

  absl::StatusOr<ArrayInfo> FlattenedArrayInfo(llvm::Type* type) {
    if (auto* array_type = llvm::dyn_cast<llvm::ArrayType>(type)) {
      TF_ASSIGN_OR_RETURN(ArrayInfo nested,
                          FlattenedArrayInfo(array_type->getElementType()));
      return ArrayInfo{nested.element_type,
                       nested.element_count * array_type->getNumElements()};
    }
    return ArrayInfo{type, 1};
  }

  absl::StatusOr<PtrExpr> BytePointerExpr(const PtrExpr& pointer) {
    std::optional<int64_t> element_size =
        FixedTypeSize(data_layout_, pointer.element_type);
    if (!element_size.has_value()) {
      return absl::UnimplementedError(absl::StrCat(
          "Cannot express MSL pointer ", pointer.base,
          " as a byte pointer because element type ",
          PrintLlvmType(*pointer.element_type), " has no fixed size."));
    }
    return PtrExpr{
        absl::StrCat("reinterpret_cast<", pointer.address_space, " char*>(",
                     pointer.base, ")"),
        ScaleIndex(pointer.index, *element_size),
        llvm::Type::getInt8Ty(function_.getContext()), pointer.address_space};
  }

  absl::StatusOr<int64_t> ScaleForGepIndex(llvm::Type* scaled_type,
                                           PtrExpr* base) {
    absl::StatusOr<int64_t> scale =
        FlattenedElementCount(data_layout_, scaled_type, base->element_type);
    if (scale.ok()) return *scale;

    std::optional<int64_t> scaled_size =
        FixedTypeSize(data_layout_, scaled_type);
    std::optional<int64_t> base_size =
        FixedTypeSize(data_layout_, base->element_type);
    if (scaled_size.has_value() && base_size.has_value() &&
        *scaled_size < *base_size) {
      TF_ASSIGN_OR_RETURN(*base, BytePointerExpr(*base));
      return *scaled_size;
    }
    return scale.status();
  }

  absl::Status EmitBody(std::string* output) {
    if (function_.empty()) return absl::OkStatus();
    absl::flat_hash_set<const llvm::BasicBlock*> visited;
    return EmitStructuredBlock(&function_.getEntryBlock(), /*stop_block=*/nullptr,
                               output, /*indent=*/1, &visited);
  }

  absl::Status EmitStructuredBlock(
      const llvm::BasicBlock* block, const llvm::BasicBlock* stop_block,
      std::string* output, int indent,
      absl::flat_hash_set<const llvm::BasicBlock*>* visited) {
    while (block != nullptr && block != stop_block) {
      if (!visited->insert(block).second) {
        return Unsupported(function_, "cyclic or reused control flow",
                           *block->getTerminator());
      }

      if (const auto* branch =
              llvm::dyn_cast<llvm::CondBrInst>(block->getTerminator());
          branch != nullptr &&
          (branch->getSuccessor(0) == block ||
           branch->getSuccessor(1) == block)) {
        const bool continue_on_true = branch->getSuccessor(0) == block;
        const llvm::BasicBlock* exit_block =
            continue_on_true ? branch->getSuccessor(1)
                             : branch->getSuccessor(0);
        absl::StrAppend(output, Indent(indent), "do {\n");
        RETURN_IF_ERROR(EmitBlockStatements(*block, output, indent + 1));
        RETURN_IF_ERROR(EmitPhiAssignments(*block, block, output, indent + 1));
        TF_ASSIGN_OR_RETURN(std::string condition, Expr(branch->getCondition()));
        if (!continue_on_true) condition = absl::StrCat("!(", condition, ")");
        absl::StrAppend(output, Indent(indent), "} while (", condition,
                        ");\n");
        block = exit_block;
        continue;
      }

      if (const auto* branch =
              llvm::dyn_cast<llvm::CondBrInst>(block->getTerminator());
          branch != nullptr) {
        const llvm::BasicBlock* true_block = branch->getSuccessor(0);
        const llvm::BasicBlock* false_block = branch->getSuccessor(1);
        if (CanEmitStructuredBlockToStop(true_block, block)) {
          absl::StrAppend(output, Indent(indent), "while (true) {\n");
          RETURN_IF_ERROR(EmitBlockStatements(*block, output, indent + 1));
          TF_ASSIGN_OR_RETURN(std::string condition,
                              Expr(branch->getCondition()));
          absl::StrAppend(output, Indent(indent + 1), "if (!(", condition,
                          ")) {\n", Indent(indent + 2), "break;\n",
                          Indent(indent + 1), "}\n");
          RETURN_IF_ERROR(EmitStructuredBlock(true_block, block, output,
                                              indent + 1, visited));
          absl::StrAppend(output, Indent(indent), "}\n");
          RETURN_IF_ERROR(EmitPhiAssignments(*false_block, block, output,
                                             indent));
          block = false_block;
          continue;
        }
        if (CanEmitStructuredBlockToStop(false_block, block)) {
          absl::StrAppend(output, Indent(indent), "while (true) {\n");
          RETURN_IF_ERROR(EmitBlockStatements(*block, output, indent + 1));
          TF_ASSIGN_OR_RETURN(std::string condition,
                              Expr(branch->getCondition()));
          absl::StrAppend(output, Indent(indent + 1), "if (", condition,
                          ") {\n", Indent(indent + 2), "break;\n",
                          Indent(indent + 1), "}\n");
          RETURN_IF_ERROR(EmitStructuredBlock(false_block, block, output,
                                              indent + 1, visited));
          absl::StrAppend(output, Indent(indent), "}\n");
          RETURN_IF_ERROR(EmitPhiAssignments(*true_block, block, output,
                                             indent));
          block = true_block;
          continue;
        }
      }

      RETURN_IF_ERROR(EmitBlockStatements(*block, output, indent));

      const llvm::Instruction* terminator = block->getTerminator();
      if (llvm::isa<llvm::ReturnInst>(terminator)) return absl::OkStatus();

      if (const auto* branch = llvm::dyn_cast<llvm::UncondBrInst>(terminator)) {
        const llvm::BasicBlock* successor = branch->getSuccessor(0);
        RETURN_IF_ERROR(
            EmitPhiAssignments(*successor, block, output, indent));
        block = successor;
        continue;
      }

      const auto* branch = llvm::dyn_cast<llvm::CondBrInst>(terminator);
      if (branch == nullptr) {
        return Unsupported(function_, "terminator", *terminator);
      }

      const llvm::BasicBlock* true_block = branch->getSuccessor(0);
      const llvm::BasicBlock* false_block = branch->getSuccessor(1);
      const llvm::BasicBlock* true_successor =
          UnconditionalBranchSuccessor(*true_block);
      const llvm::BasicBlock* false_successor =
          UnconditionalBranchSuccessor(*false_block);

      TF_ASSIGN_OR_RETURN(std::string condition, Expr(branch->getCondition()));

      if (true_successor != nullptr && true_successor == false_successor) {
        absl::StrAppend(output, Indent(indent), "if (", condition, ") {\n");
        RETURN_IF_ERROR(EmitStructuredBlock(true_block, true_successor, output,
                                            indent + 1, visited));
        absl::StrAppend(output, Indent(indent), "} else {\n");
        RETURN_IF_ERROR(EmitStructuredBlock(false_block, false_successor, output,
                                            indent + 1, visited));
        absl::StrAppend(output, Indent(indent), "}\n");
        block = true_successor;
        continue;
      }

      if (true_successor == false_block) {
        absl::StrAppend(output, Indent(indent), "if (", condition, ") {\n");
        RETURN_IF_ERROR(EmitStructuredBlock(true_block, false_block, output,
                                            indent + 1, visited));
        absl::StrAppend(output, Indent(indent), "}\n");
        block = false_block;
        continue;
      }

      if (false_successor == true_block) {
        absl::StrAppend(output, Indent(indent), "if (!(", condition, ")) {\n");
        RETURN_IF_ERROR(EmitStructuredBlock(false_block, true_block, output,
                                            indent + 1, visited));
        absl::StrAppend(output, Indent(indent), "}\n");
        block = true_block;
        continue;
      }

      if (CanEmitStructuredBlockToStop(true_block, false_block)) {
        absl::StrAppend(output, Indent(indent), "if (", condition, ") {\n");
        RETURN_IF_ERROR(EmitStructuredBlock(true_block, false_block, output,
                                            indent + 1, visited));
        absl::StrAppend(output, Indent(indent), "}\n");
        block = false_block;
        continue;
      }

      if (CanEmitStructuredBlockToStop(false_block, true_block)) {
        absl::StrAppend(output, Indent(indent), "if (!(", condition, ")) {\n");
        RETURN_IF_ERROR(EmitStructuredBlock(false_block, true_block, output,
                                            indent + 1, visited));
        absl::StrAppend(output, Indent(indent), "}\n");
        block = true_block;
        continue;
      }

      if (stop_block != nullptr &&
          CanEmitStructuredBlockToStop(true_block, stop_block) &&
          CanEmitStructuredBlockToStop(false_block, stop_block)) {
        absl::StrAppend(output, Indent(indent), "if (", condition, ") {\n");
        absl::flat_hash_set<const llvm::BasicBlock*> true_visited = *visited;
        RETURN_IF_ERROR(EmitStructuredBlock(true_block, stop_block, output,
                                            indent + 1, &true_visited));
        absl::StrAppend(output, Indent(indent), "} else {\n");
        absl::flat_hash_set<const llvm::BasicBlock*> false_visited = *visited;
        RETURN_IF_ERROR(EmitStructuredBlock(false_block, stop_block, output,
                                            indent + 1, &false_visited));
        absl::StrAppend(output, Indent(indent), "}\n");
        visited->insert(true_visited.begin(), true_visited.end());
        visited->insert(false_visited.begin(), false_visited.end());
        block = stop_block;
        continue;
      }

      if (const llvm::BasicBlock* continuation =
              FindCommonStructuredContinuation(true_block, false_block,
                                               stop_block)) {
        absl::StrAppend(output, Indent(indent), "if (", condition, ") {\n");
        absl::flat_hash_set<const llvm::BasicBlock*> true_visited = *visited;
        RETURN_IF_ERROR(EmitStructuredBlock(true_block, continuation, output,
                                            indent + 1, &true_visited));
        absl::StrAppend(output, Indent(indent), "} else {\n");
        absl::flat_hash_set<const llvm::BasicBlock*> false_visited = *visited;
        RETURN_IF_ERROR(EmitStructuredBlock(false_block, continuation, output,
                                            indent + 1, &false_visited));
        absl::StrAppend(output, Indent(indent), "}\n");
        visited->insert(true_visited.begin(), true_visited.end());
        visited->insert(false_visited.begin(), false_visited.end());
        block = continuation;
        continue;
      }

      return Unsupported(function_, "control flow shape", *terminator);
    }

    return absl::OkStatus();
  }

  absl::flat_hash_map<const llvm::BasicBlock*, int> ReachableBlockDistances(
      const llvm::BasicBlock* start) const {
    absl::flat_hash_map<const llvm::BasicBlock*, int> distances;
    if (start == nullptr) return distances;

    std::vector<const llvm::BasicBlock*> worklist = {start};
    distances.insert_or_assign(start, 0);
    for (size_t i = 0; i < worklist.size(); ++i) {
      const llvm::BasicBlock* block = worklist[i];
      int next_distance = distances.at(block) + 1;
      const llvm::Instruction* terminator = block->getTerminator();
      if (llvm::isa<llvm::ReturnInst>(terminator)) continue;

      auto add_successor = [&](const llvm::BasicBlock* successor) {
        if (successor == nullptr || distances.contains(successor)) return;
        distances.insert_or_assign(successor, next_distance);
        worklist.push_back(successor);
      };

      if (const auto* branch =
              llvm::dyn_cast<llvm::UncondBrInst>(terminator)) {
        add_successor(branch->getSuccessor(0));
        continue;
      }
      if (const auto* branch = llvm::dyn_cast<llvm::CondBrInst>(terminator)) {
        add_successor(branch->getSuccessor(0));
        add_successor(branch->getSuccessor(1));
      }
    }
    return distances;
  }

  const llvm::BasicBlock* FindCommonStructuredContinuation(
      const llvm::BasicBlock* true_block, const llvm::BasicBlock* false_block,
      const llvm::BasicBlock* stop_block) const {
    absl::flat_hash_map<const llvm::BasicBlock*, int> true_distances =
        ReachableBlockDistances(true_block);
    absl::flat_hash_map<const llvm::BasicBlock*, int> false_distances =
        ReachableBlockDistances(false_block);

    const llvm::BasicBlock* best = nullptr;
    int best_score = 0;
    for (const llvm::BasicBlock& candidate_ref : function_) {
      const llvm::BasicBlock* candidate = &candidate_ref;
      if (candidate == true_block || candidate == false_block ||
          candidate == stop_block) {
        continue;
      }
      auto true_it = true_distances.find(candidate);
      auto false_it = false_distances.find(candidate);
      if (true_it == true_distances.end() ||
          false_it == false_distances.end()) {
        continue;
      }
      if (!CanEmitStructuredBlockToStop(true_block, candidate) ||
          !CanEmitStructuredBlockToStop(false_block, candidate)) {
        continue;
      }
      if (stop_block != nullptr &&
          !CanEmitStructuredBlockToStop(candidate, stop_block)) {
        continue;
      }

      int score = std::max(true_it->second, false_it->second) * 1000 +
                  true_it->second + false_it->second;
      if (best == nullptr || score < best_score) {
        best = candidate;
        best_score = score;
      }
    }
    return best;
  }

  bool CanEmitStructuredBlockToStop(const llvm::BasicBlock* block,
                                    const llvm::BasicBlock* stop_block) const {
    absl::flat_hash_map<BlockStopKey, bool> memo;
    absl::flat_hash_set<BlockStopKey> active;
    return CanEmitStructuredBlockToStop(block, stop_block, &memo, &active);
  }

  bool CanEmitStructuredBlockToStop(
      const llvm::BasicBlock* block, const llvm::BasicBlock* stop_block,
      absl::flat_hash_map<BlockStopKey, bool>* memo,
      absl::flat_hash_set<BlockStopKey>* active) const {
    if (block == stop_block) return true;
    if (block == nullptr) return false;

    BlockStopKey key{block, stop_block};
    if (auto it = memo->find(key); it != memo->end()) {
      return it->second;
    }
    if (!active->insert(key).second) return false;
    auto finish = [&](bool result) {
      active->erase(key);
      memo->insert_or_assign(key, result);
      return result;
    };

    const llvm::Instruction* terminator = block->getTerminator();
    if (llvm::isa<llvm::ReturnInst>(terminator)) return finish(false);

    if (const auto* branch = llvm::dyn_cast<llvm::UncondBrInst>(terminator)) {
      return finish(CanEmitStructuredBlockToStop(
          branch->getSuccessor(0), stop_block, memo, active));
    }

    const auto* branch = llvm::dyn_cast<llvm::CondBrInst>(terminator);
    if (branch == nullptr) return finish(false);

    const llvm::BasicBlock* true_block = branch->getSuccessor(0);
    const llvm::BasicBlock* false_block = branch->getSuccessor(1);
    const llvm::BasicBlock* true_successor =
        UnconditionalBranchSuccessor(*true_block);
    const llvm::BasicBlock* false_successor =
        UnconditionalBranchSuccessor(*false_block);

    if (true_block == block) {
      return finish(
          CanEmitStructuredBlockToStop(false_block, stop_block, memo, active));
    }
    if (false_block == block) {
      return finish(
          CanEmitStructuredBlockToStop(true_block, stop_block, memo, active));
    }

    if (CanEmitStructuredBlockToStop(true_block, block, memo, active) &&
        CanEmitStructuredBlockToStop(false_block, stop_block, memo, active)) {
      return finish(true);
    }

    if (CanEmitStructuredBlockToStop(false_block, block, memo, active) &&
        CanEmitStructuredBlockToStop(true_block, stop_block, memo, active)) {
      return finish(true);
    }

    if (true_successor != nullptr && true_successor == false_successor) {
      return finish(CanEmitStructuredBlockToStop(true_block, true_successor,
                                                 memo, active) &&
                    CanEmitStructuredBlockToStop(false_block, false_successor,
                                                 memo, active) &&
                    CanEmitStructuredBlockToStop(true_successor, stop_block,
                                                 memo, active));
    }

    if (CanEmitStructuredBlockToStop(true_block, false_block, memo, active) &&
        CanEmitStructuredBlockToStop(false_block, stop_block, memo, active)) {
      return finish(true);
    }

    if (CanEmitStructuredBlockToStop(false_block, true_block,
                                     memo, active) &&
        CanEmitStructuredBlockToStop(true_block, stop_block, memo, active)) {
      return finish(true);
    }

    if (stop_block != nullptr) {
      return finish(CanEmitStructuredBlockToStop(true_block, stop_block, memo,
                                                 active) &&
                    CanEmitStructuredBlockToStop(false_block, stop_block, memo,
                                                 active));
    }

    return finish(false);
  }

  absl::StatusOr<bool> TryEmitIfElseDiamond(const llvm::CondBrInst& branch,
                                            std::string* output) {
    const llvm::BasicBlock& entry = function_.getEntryBlock();
    const llvm::BasicBlock* true_block = branch.getSuccessor(0);
    const llvm::BasicBlock* false_block = branch.getSuccessor(1);

    const llvm::BasicBlock* true_successor =
        UnconditionalBranchSuccessor(*true_block);
    const llvm::BasicBlock* false_successor =
        UnconditionalBranchSuccessor(*false_block);
    if (true_successor == nullptr || true_successor != false_successor) {
      return false;
    }

    const llvm::BasicBlock* merge_block = true_successor;
    const llvm::BasicBlock* continuation_block = nullptr;
    const llvm::Instruction* merge_terminator = merge_block->getTerminator();
    if (llvm::isa<llvm::ReturnInst>(merge_terminator)) {
      continuation_block = merge_block;
    } else {
      continuation_block = UnconditionalBranchSuccessor(*merge_block);
      if (continuation_block == nullptr ||
          !llvm::isa<llvm::ReturnInst>(continuation_block->getTerminator())) {
        return false;
      }
    }

    RETURN_IF_ERROR(EmitBlockStatements(entry, output, 1));
    TF_ASSIGN_OR_RETURN(std::string condition, Expr(branch.getCondition()));
    absl::StrAppend(output, "  if (", condition, ") {\n");
    RETURN_IF_ERROR(EmitBlockStatements(*true_block, output, 2));
    RETURN_IF_ERROR(EmitPhiAssignments(*merge_block, true_block, output, 2));
    absl::StrAppend(output, "  } else {\n");
    RETURN_IF_ERROR(EmitBlockStatements(*false_block, output, 2));
    RETURN_IF_ERROR(EmitPhiAssignments(*merge_block, false_block, output, 2));
    absl::StrAppend(output, "  }\n");

    RETURN_IF_ERROR(EmitBlockStatements(*merge_block, output, 1));
    if (continuation_block != merge_block) {
      RETURN_IF_ERROR(EmitBlockStatements(*continuation_block, output, 1));
    }
    return true;
  }

  const llvm::BasicBlock* UnconditionalBranchSuccessor(
      const llvm::BasicBlock& block) const {
    const auto* branch = llvm::dyn_cast<llvm::UncondBrInst>(block.getTerminator());
    if (branch == nullptr) return nullptr;
    return branch->getSuccessor(0);
  }

  absl::Status EmitPhiAssignments(const llvm::BasicBlock& block,
                                  const llvm::BasicBlock* incoming_block,
                                  std::string* output, int indent) {
    for (const llvm::Instruction& instruction : block) {
      const auto* phi = llvm::dyn_cast<llvm::PHINode>(&instruction);
      if (phi == nullptr) break;
      const llvm::Value* incoming = phi->getIncomingValueForBlock(incoming_block);
      if (incoming == nullptr) {
        return absl::InvalidArgumentError(
            "Metal MSL emission could not find phi incoming value.");
      }
      TF_ASSIGN_OR_RETURN(std::string value, Expr(incoming));
      absl::StrAppend(output, Indent(indent), Name(phi), " = ", value, ";\n");
    }
    return absl::OkStatus();
  }

  bool IsReturnOnlyBlock(const llvm::BasicBlock& block) const {
    for (const llvm::Instruction& instruction : block) {
      if (llvm::isa<llvm::ReturnInst>(&instruction)) return true;
      return false;
    }
    return false;
  }

  bool IsBodyBlock(const llvm::BasicBlock& block,
                   const llvm::BasicBlock* exit_block) const {
    const llvm::Instruction* terminator = block.getTerminator();
    if (llvm::isa<llvm::ReturnInst>(terminator)) return true;
    const auto* branch = llvm::dyn_cast<llvm::UncondBrInst>(terminator);
    return branch != nullptr && branch->getSuccessor(0) == exit_block;
  }

  absl::Status EmitBlockStatements(const llvm::BasicBlock& block,
                                   std::string* output, int indent) {
    for (const llvm::Instruction& instruction : block) {
      if (llvm::isa<llvm::PHINode>(&instruction)) continue;
      if (instruction.isTerminator()) continue;
      RETURN_IF_ERROR(EmitInstruction(instruction, output, indent));
    }
    return absl::OkStatus();
  }

  absl::Status EmitInstruction(const llvm::Instruction& instruction,
                               std::string* output, int indent) {
    if (llvm::isa<llvm::AllocaInst>(&instruction)) {
      return absl::OkStatus();
    }
    if (llvm::isa<llvm::GetElementPtrInst>(&instruction) &&
        IsPointerType(instruction)) {
      return absl::OkStatus();
    }
    if (auto* cast = llvm::dyn_cast<llvm::CastInst>(&instruction);
        cast != nullptr && IsPointerType(instruction)) {
      return absl::OkStatus();
    }

    if (auto* store = llvm::dyn_cast<llvm::StoreInst>(&instruction)) {
      TF_ASSIGN_OR_RETURN(PtrExpr pointer,
                          PointerExprFor(store->getPointerOperand()));
      TF_ASSIGN_OR_RETURN(std::string value, Expr(store->getValueOperand()));
      llvm::Type* value_type = store->getValueOperand()->getType();
      if (value_type == pointer.element_type) {
        absl::StrAppend(output, Indent(indent), pointer.base, "[",
                        pointer.index, "] = ", value, ";\n");
      } else {
        TF_ASSIGN_OR_RETURN(std::string value_msl_type, MslType(value_type));
        absl::StrAppend(output, Indent(indent), "*reinterpret_cast<",
                        pointer.address_space, " ", value_msl_type, "*>(&",
                        pointer.base, "[", pointer.index, "]) = ", value,
                        ";\n");
      }
      return absl::OkStatus();
    }

    if (auto* atomic = llvm::dyn_cast<llvm::AtomicRMWInst>(&instruction)) {
      TF_ASSIGN_OR_RETURN(std::string statement, AtomicRmwStatement(*atomic));
      absl::StrAppend(output, Indent(indent), statement, "\n");
      return absl::OkStatus();
    }

    if (auto* cmpxchg = llvm::dyn_cast<llvm::AtomicCmpXchgInst>(&instruction)) {
      RETURN_IF_ERROR(EmitAtomicCmpXchg(*cmpxchg, output, indent));
      return absl::OkStatus();
    }

    if (auto* call = llvm::dyn_cast<llvm::CallInst>(&instruction)) {
      if (call->getType()->isVoidTy()) {
        TF_ASSIGN_OR_RETURN(std::string statement,
                            VoidCallStatementOrInline(*call));
        if (!statement.empty()) {
          absl::StrAppend(output, Indent(indent), statement, "\n");
        }
        return absl::OkStatus();
      }
    }

    if (IsNonMaterializedAggregateType(instruction.getType())) {
      return absl::OkStatus();
    }

    if (instruction.getType()->isVoidTy()) {
      return Unsupported(function_, "void instruction", instruction);
    }

    TF_ASSIGN_OR_RETURN(std::string expression, InstructionExpr(instruction));
    absl::StrAppend(output, Indent(indent), Name(&instruction), " = ",
                    expression, ";\n");
    return absl::OkStatus();
  }

  absl::StatusOr<std::string> Expr(const llvm::Value* value) {
    if (auto* constant_int = llvm::dyn_cast<llvm::ConstantInt>(value)) {
      return IntegerLiteral(constant_int->getValue(), /*is_signed=*/true);
    }
    if (auto* constant_fp = llvm::dyn_cast<llvm::ConstantFP>(value)) {
      const llvm::APFloat& as_apfloat = constant_fp->getValueAPF();
      if (as_apfloat.isInfinity() || as_apfloat.isNaN()) {
        std::string sign = as_apfloat.isNegative() ? "-" : "";
        std::string literal = as_apfloat.isInfinity() ? "INFINITY" : "NAN";
        if (value->getType()->isBFloatTy()) {
          return absl::StrCat("xla_metal_f32_to_bf16(float(", sign, literal,
                              "))");
        }
        if (value->getType()->isHalfTy()) {
          return absl::StrCat("half(", sign, literal, ")");
        }
        if (value->getType()->isFloatTy()) {
          return absl::StrCat("float(", sign, literal, ")");
        }
        return absl::StrCat("double(", sign, literal, ")");
      }
      double as_double = as_apfloat.convertToDouble();
      if (value->getType()->isBFloatTy()) {
        return absl::StrFormat("xla_metal_f32_to_bf16(float(%.9g))",
                               as_double);
      }
      if (value->getType()->isHalfTy()) {
        return absl::StrFormat("half(%.9g)", as_double);
      }
      if (value->getType()->isFloatTy()) {
        return absl::StrFormat("float(%.9g)", as_double);
      }
      return absl::StrFormat("%.17g", as_double);
    }
    if (auto* constant_struct = llvm::dyn_cast<llvm::ConstantStruct>(value);
        constant_struct != nullptr && IsComplexFloatStruct(value->getType())) {
      TF_ASSIGN_OR_RETURN(std::string real,
                          Expr(constant_struct->getOperand(0)));
      TF_ASSIGN_OR_RETURN(std::string imag,
                          Expr(constant_struct->getOperand(1)));
      return absl::StrCat("float2(", real, ", ", imag, ")");
    }
    if (llvm::isa<llvm::ConstantAggregateZero>(value) ||
        llvm::isa<llvm::UndefValue>(value) ||
        llvm::isa<llvm::PoisonValue>(value)) {
      return ZeroInitializer(value->getType());
    }
    if (auto* constant_vector = llvm::dyn_cast<llvm::ConstantDataVector>(value)) {
      TF_ASSIGN_OR_RETURN(std::string type, MslType(value->getType()));
      std::vector<std::string> elements;
      elements.reserve(constant_vector->getNumElements());
      for (int i = 0; i < constant_vector->getNumElements(); ++i) {
        TF_ASSIGN_OR_RETURN(std::string element,
                            Expr(constant_vector->getElementAsConstant(i)));
        elements.push_back(std::move(element));
      }
      return absl::StrCat(type, "(", absl::StrJoin(elements, ", "), ")");
    }
    if (auto* argument = llvm::dyn_cast<llvm::Argument>(value)) {
      return Name(argument);
    }
    if (auto* instruction = llvm::dyn_cast<llvm::Instruction>(value)) {
      if (IsPointerType(*instruction)) {
        return Unsupported(function_, "pointer value as scalar expression",
                           *instruction);
      }
      return Name(instruction);
    }
    if (auto* constant = llvm::dyn_cast<llvm::ConstantExpr>(value)) {
      return ConstantExpr(*constant);
    }
    return Unsupported(function_, "value expression", *value);
  }

  absl::StatusOr<std::string> ConstantExpr(const llvm::ConstantExpr& constant) {
    switch (constant.getOpcode()) {
      case llvm::Instruction::Trunc:
      case llvm::Instruction::ZExt:
      case llvm::Instruction::SExt:
      case llvm::Instruction::FPTrunc:
      case llvm::Instruction::FPExt:
      case llvm::Instruction::SIToFP:
      case llvm::Instruction::UIToFP:
      case llvm::Instruction::FPToSI:
      case llvm::Instruction::FPToUI:
      case llvm::Instruction::BitCast: {
        TF_ASSIGN_OR_RETURN(std::string operand, Expr(constant.getOperand(0)));
        TF_ASSIGN_OR_RETURN(std::string type, MslType(constant.getType()));
        if (constant.getOpcode() == llvm::Instruction::BitCast) {
          return absl::StrCat("as_type<", type, ">(", operand, ")");
        }
        return absl::StrCat("static_cast<", type, ">(", operand, ")");
      }
      default:
        return absl::UnimplementedError(absl::StrCat(
            "Unsupported LLVM constant expression for MSL: ",
            PrintLlvm(constant)));
    }
  }

  absl::StatusOr<std::string> InstructionExpr(
      const llvm::Instruction& instruction) {
    if (auto* load = llvm::dyn_cast<llvm::LoadInst>(&instruction)) {
      TF_ASSIGN_OR_RETURN(PtrExpr pointer,
                          PointerExprFor(load->getPointerOperand()));
      return LoadFromPointer(pointer, load->getType());
    }
    if (auto* binary = llvm::dyn_cast<llvm::BinaryOperator>(&instruction)) {
      return BinaryExpr(*binary);
    }
    if (instruction.getOpcode() == llvm::Instruction::FNeg) {
      TF_ASSIGN_OR_RETURN(std::string operand, Expr(instruction.getOperand(0)));
      return absl::StrCat("(-", operand, ")");
    }
    if (auto* icmp = llvm::dyn_cast<llvm::ICmpInst>(&instruction)) {
      return ICmpExpr(*icmp);
    }
    if (auto* fcmp = llvm::dyn_cast<llvm::FCmpInst>(&instruction)) {
      return FCmpExpr(*fcmp);
    }
    if (auto* select = llvm::dyn_cast<llvm::SelectInst>(&instruction)) {
      TF_ASSIGN_OR_RETURN(std::string condition, Expr(select->getCondition()));
      TF_ASSIGN_OR_RETURN(std::string true_value, Expr(select->getTrueValue()));
      TF_ASSIGN_OR_RETURN(std::string false_value,
                          Expr(select->getFalseValue()));
      return absl::StrCat("(", condition, " ? ", true_value, " : ",
                          false_value, ")");
    }
    if (auto* cast = llvm::dyn_cast<llvm::CastInst>(&instruction)) {
      return CastExpr(*cast);
    }
    if (auto* call = llvm::dyn_cast<llvm::CallInst>(&instruction)) {
      return CallExpr(*call);
    }
    if (auto* insert = llvm::dyn_cast<llvm::InsertValueInst>(&instruction)) {
      return InsertValueExpr(*insert);
    }
    if (auto* extract = llvm::dyn_cast<llvm::ExtractValueInst>(&instruction)) {
      return ExtractValueExpr(*extract);
    }
    if (auto* extract =
            llvm::dyn_cast<llvm::ExtractElementInst>(&instruction)) {
      TF_ASSIGN_OR_RETURN(std::string vector, Expr(extract->getVectorOperand()));
      TF_ASSIGN_OR_RETURN(std::string index, Expr(extract->getIndexOperand()));
      if (auto* vector_type = llvm::dyn_cast<llvm::FixedVectorType>(
              extract->getVectorOperand()->getType());
          vector_type != nullptr && WideVectorTypeName(vector_type).has_value()) {
        return absl::StrCat(vector, ".elements[", index, "]");
      }
      return absl::StrCat(vector, "[", index, "]");
    }
    if (auto* insert =
            llvm::dyn_cast<llvm::InsertElementInst>(&instruction)) {
      return InsertElementExpr(*insert);
    }
    if (auto* shuffle =
            llvm::dyn_cast<llvm::ShuffleVectorInst>(&instruction)) {
      return ShuffleVectorExpr(*shuffle);
    }
    return Unsupported(function_, "instruction", instruction);
  }

  absl::StatusOr<std::string> VectorElementExpr(const llvm::Value* vector,
                                                int index) {
    auto* vector_type = llvm::cast<llvm::FixedVectorType>(vector->getType());
    llvm::Type* element_type = vector_type->getElementType();
    if (llvm::isa<llvm::UndefValue>(vector) ||
        llvm::isa<llvm::PoisonValue>(vector) ||
        llvm::isa<llvm::ConstantAggregateZero>(vector)) {
      return ZeroInitializer(element_type);
    }
    if (auto* constant_data =
            llvm::dyn_cast<llvm::ConstantDataVector>(vector)) {
      return Expr(constant_data->getElementAsConstant(index));
    }
    TF_ASSIGN_OR_RETURN(std::string expression, Expr(vector));
    if (WideVectorTypeName(vector_type).has_value()) {
      return absl::StrCat(expression, ".elements[", index, "]");
    }
    return absl::StrCat(expression, "[", index, "]");
  }

  absl::StatusOr<std::string> InsertElementExpr(
      const llvm::InsertElementInst& insert) {
    auto* vector_type =
        llvm::dyn_cast<llvm::FixedVectorType>(insert.getType());
    if (vector_type == nullptr || vector_type->getNumElements() > 4) {
      return Unsupported(function_, "vector insertion", insert);
    }
    auto* index = llvm::dyn_cast<llvm::ConstantInt>(insert.getOperand(2));
    if (index == nullptr) {
      return Unsupported(function_, "dynamic vector insertion", insert);
    }
    uint64_t inserted_index = index->getZExtValue();
    if (inserted_index >= vector_type->getNumElements()) {
      return Unsupported(function_, "out-of-bounds vector insertion", insert);
    }

    TF_ASSIGN_OR_RETURN(std::string type, MslType(insert.getType()));
    TF_ASSIGN_OR_RETURN(std::string inserted_value,
                        Expr(insert.getOperand(1)));
    std::vector<std::string> elements;
    elements.reserve(vector_type->getNumElements());
    for (int i = 0; i < vector_type->getNumElements(); ++i) {
      if (i == inserted_index) {
        elements.push_back(inserted_value);
      } else {
        TF_ASSIGN_OR_RETURN(std::string element,
                            VectorElementExpr(insert.getOperand(0), i));
        elements.push_back(std::move(element));
      }
    }
    return absl::StrCat(type, "(", absl::StrJoin(elements, ", "), ")");
  }

  absl::StatusOr<std::string> ShuffleVectorExpr(
      const llvm::ShuffleVectorInst& shuffle) {
    auto* result_type =
        llvm::dyn_cast<llvm::FixedVectorType>(shuffle.getType());
    auto* lhs_type = llvm::dyn_cast<llvm::FixedVectorType>(
        shuffle.getOperand(0)->getType());
    if (result_type == nullptr || lhs_type == nullptr ||
        result_type->getNumElements() > 4) {
      return Unsupported(function_, "vector shuffle", shuffle);
    }

    TF_ASSIGN_OR_RETURN(std::string type, MslType(shuffle.getType()));
    std::vector<int> mask = shuffle.getShuffleMask();
    std::vector<std::string> elements;
    elements.reserve(mask.size());
    for (int lane : mask) {
      if (lane < 0) {
        TF_ASSIGN_OR_RETURN(
            std::string zero,
            ZeroInitializer(result_type->getElementType()));
        elements.push_back(std::move(zero));
        continue;
      }
      const llvm::Value* source = shuffle.getOperand(0);
      int source_lane = lane;
      if (lane >= lhs_type->getNumElements()) {
        source = shuffle.getOperand(1);
        source_lane = lane - lhs_type->getNumElements();
      }
      TF_ASSIGN_OR_RETURN(std::string element,
                          VectorElementExpr(source, source_lane));
      elements.push_back(std::move(element));
    }
    return absl::StrCat(type, "(", absl::StrJoin(elements, ", "), ")");
  }

  absl::StatusOr<std::string> BinaryExpr(const llvm::BinaryOperator& binary) {
    TF_ASSIGN_OR_RETURN(std::string lhs, Expr(binary.getOperand(0)));
    TF_ASSIGN_OR_RETURN(std::string rhs, Expr(binary.getOperand(1)));
    return BinaryExprWithOperands(binary, lhs, rhs);
  }

  absl::StatusOr<std::string> ExtractValueExpr(
      const llvm::ExtractValueInst& extract) {
    if (extract.getNumIndices() != 1) {
      return Unsupported(function_, "nested aggregate extraction", extract);
    }
    const llvm::Value* aggregate = extract.getAggregateOperand();
    unsigned field = *extract.idx_begin();
    if (IsComplexFloatStruct(aggregate->getType())) {
      TF_ASSIGN_OR_RETURN(std::string value, Expr(aggregate));
      if (field == 0) return absl::StrCat(value, ".x");
      if (field == 1) return absl::StrCat(value, ".y");
      return absl::InvalidArgumentError(
          absl::StrCat("complex field ", field, " is out of range."));
    }
    if (auto* cmpxchg = llvm::dyn_cast<llvm::AtomicCmpXchgInst>(aggregate)) {
      if (field == 0) return CmpXchgOldName(*cmpxchg);
      if (field == 1) return CmpXchgSuccessName(*cmpxchg);
      return absl::InvalidArgumentError(
          absl::StrCat("cmpxchg field ", field, " is out of range."));
    }
    if (auto* call = llvm::dyn_cast<llvm::CallInst>(aggregate)) {
      return InlineSimpleAggregateElement(*call, field);
    }
    return AggregateElementValue(aggregate, /*outer_call=*/nullptr, field);
  }

  absl::StatusOr<std::string> InsertValueExpr(
      const llvm::InsertValueInst& insert) {
    if (!IsComplexFloatStruct(insert.getType()) || insert.getNumIndices() != 1) {
      return Unsupported(function_, "aggregate insertion", insert);
    }

    unsigned field = *insert.idx_begin();
    if (field > 1) {
      return absl::InvalidArgumentError(
          absl::StrCat("complex field ", field, " is out of range."));
    }

    std::string components[2] = {"0.0f", "0.0f"};
    const llvm::Value* aggregate = insert.getAggregateOperand();
    if (!llvm::isa<llvm::UndefValue>(aggregate) &&
        !llvm::isa<llvm::PoisonValue>(aggregate)) {
      TF_ASSIGN_OR_RETURN(components[0], ComplexComponentExpr(aggregate, 0));
      TF_ASSIGN_OR_RETURN(components[1], ComplexComponentExpr(aggregate, 1));
    }
    TF_ASSIGN_OR_RETURN(std::string inserted,
                        Expr(insert.getInsertedValueOperand()));
    components[field] = std::move(inserted);
    return absl::StrCat("float2(", components[0], ", ", components[1], ")");
  }

  absl::StatusOr<std::string> ComplexComponentExpr(const llvm::Value* value,
                                                   unsigned field) {
    if (field > 1) {
      return absl::InvalidArgumentError(
          absl::StrCat("complex field ", field, " is out of range."));
    }
    if (llvm::isa<llvm::UndefValue>(value) ||
        llvm::isa<llvm::PoisonValue>(value)) {
      return "0.0f";
    }
    if (auto* insert = llvm::dyn_cast<llvm::InsertValueInst>(value)) {
      if (insert->getNumIndices() != 1) {
        return Unsupported(function_, "aggregate insertion", *insert);
      }
      if (*insert->idx_begin() == field) {
        return Expr(insert->getInsertedValueOperand());
      }
      return ComplexComponentExpr(insert->getAggregateOperand(), field);
    }
    TF_ASSIGN_OR_RETURN(std::string expression, Expr(value));
    return absl::StrCat(expression, field == 0 ? ".x" : ".y");
  }

  absl::StatusOr<std::string> BinaryExprWithOperands(
      const llvm::BinaryOperator& binary, absl::string_view lhs,
      absl::string_view rhs) {
    switch (binary.getOpcode()) {
      case llvm::Instruction::Add:
      case llvm::Instruction::FAdd:
        return absl::StrCat("(", lhs, " + ", rhs, ")");
      case llvm::Instruction::Sub:
      case llvm::Instruction::FSub:
        return absl::StrCat("(", lhs, " - ", rhs, ")");
      case llvm::Instruction::Mul:
      case llvm::Instruction::FMul:
        return absl::StrCat("(", lhs, " * ", rhs, ")");
      case llvm::Instruction::SDiv:
      case llvm::Instruction::FDiv:
        return absl::StrCat("(", lhs, " / ", rhs, ")");
      case llvm::Instruction::SRem:
        return absl::StrCat("(", lhs, " % ", rhs, ")");
      case llvm::Instruction::FRem:
        return absl::StrCat("fmod(", lhs, ", ", rhs, ")");
      case llvm::Instruction::UDiv:
      case llvm::Instruction::URem: {
        TF_ASSIGN_OR_RETURN(std::string type,
                            MslType(binary.getType(), /*unsigned_integer=*/true));
        std::string op =
            binary.getOpcode() == llvm::Instruction::UDiv ? " / " : " % ";
        return absl::StrCat("(static_cast<", type, ">(", lhs, ")", op,
                            "static_cast<", type, ">(", rhs, "))");
      }
      case llvm::Instruction::Shl:
        return absl::StrCat("(", lhs, " << ", rhs, ")");
      case llvm::Instruction::LShr: {
        TF_ASSIGN_OR_RETURN(std::string type, MslType(binary.getType()));
        TF_ASSIGN_OR_RETURN(
            std::string unsigned_type,
            MslType(binary.getType(), /*unsigned_integer=*/true));
        return absl::StrCat("(as_type<", type, ">(static_cast<",
                            unsigned_type, ">(", lhs, ") >> ", rhs, "))");
      }
      case llvm::Instruction::AShr:
        return absl::StrCat("(", lhs, " >> ", rhs, ")");
      case llvm::Instruction::And:
        return absl::StrCat("(", lhs, " & ", rhs, ")");
      case llvm::Instruction::Or:
        return absl::StrCat("(", lhs, " | ", rhs, ")");
      case llvm::Instruction::Xor:
        return absl::StrCat("(", lhs, " ^ ", rhs, ")");
      default:
        return Unsupported(function_, "binary operator", binary);
    }
  }

  absl::StatusOr<std::string> CastForUnsignedCompare(
      const llvm::Value* value, bool unsigned_compare) {
    TF_ASSIGN_OR_RETURN(std::string expression, Expr(value));
    return CastForUnsignedCompare(value->getType(), expression,
                                  unsigned_compare);
  }

  absl::StatusOr<std::string> CastForUnsignedCompare(
      llvm::Type* type, absl::string_view expression, bool unsigned_compare) {
    if (!unsigned_compare || !type->isIntegerTy()) {
      return std::string(expression);
    }
    TF_ASSIGN_OR_RETURN(std::string msl_type,
                        MslType(type, /*unsigned_integer=*/true));
    return absl::StrCat("static_cast<", msl_type, ">(", expression, ")");
  }

  absl::StatusOr<std::string> ICmpExpr(const llvm::ICmpInst& icmp) {
    bool unsigned_compare = icmp.isUnsigned();
    TF_ASSIGN_OR_RETURN(std::string lhs,
                        CastForUnsignedCompare(icmp.getOperand(0),
                                               unsigned_compare));
    TF_ASSIGN_OR_RETURN(std::string rhs,
                        CastForUnsignedCompare(icmp.getOperand(1),
                                               unsigned_compare));
    return ICmpExprWithOperands(icmp, lhs, rhs);
  }

  absl::StatusOr<std::string> ICmpExprWithOperands(
      const llvm::ICmpInst& icmp, absl::string_view lhs,
      absl::string_view rhs) {
    switch (icmp.getPredicate()) {
      case llvm::CmpInst::ICMP_EQ:
        return absl::StrCat("(", lhs, " == ", rhs, ")");
      case llvm::CmpInst::ICMP_NE:
        return absl::StrCat("(", lhs, " != ", rhs, ")");
      case llvm::CmpInst::ICMP_UGT:
      case llvm::CmpInst::ICMP_SGT:
        return absl::StrCat("(", lhs, " > ", rhs, ")");
      case llvm::CmpInst::ICMP_UGE:
      case llvm::CmpInst::ICMP_SGE:
        return absl::StrCat("(", lhs, " >= ", rhs, ")");
      case llvm::CmpInst::ICMP_ULT:
      case llvm::CmpInst::ICMP_SLT:
        return absl::StrCat("(", lhs, " < ", rhs, ")");
      case llvm::CmpInst::ICMP_ULE:
      case llvm::CmpInst::ICMP_SLE:
        return absl::StrCat("(", lhs, " <= ", rhs, ")");
      default:
        return Unsupported(function_, "integer comparison", icmp);
    }
  }

  absl::StatusOr<std::string> CastForFloatCompare(
      llvm::Type* type, absl::string_view expression) {
    if (type->isBFloatTy()) {
      return absl::StrCat("xla_metal_bf16_to_f32(", expression, ")");
    }
    return std::string(expression);
  }

  absl::StatusOr<std::string> FCmpExpr(const llvm::FCmpInst& fcmp) {
    TF_ASSIGN_OR_RETURN(std::string lhs, Expr(fcmp.getOperand(0)));
    TF_ASSIGN_OR_RETURN(std::string rhs, Expr(fcmp.getOperand(1)));
    return FCmpExprWithOperands(fcmp, lhs, rhs);
  }

  absl::StatusOr<std::string> FCmpExprWithOperands(
      const llvm::FCmpInst& fcmp, absl::string_view lhs,
      absl::string_view rhs) {
    TF_ASSIGN_OR_RETURN(std::string lhs_expr,
                        CastForFloatCompare(fcmp.getOperand(0)->getType(),
                                            lhs));
    TF_ASSIGN_OR_RETURN(std::string rhs_expr,
                        CastForFloatCompare(fcmp.getOperand(1)->getType(),
                                            rhs));
    switch (fcmp.getPredicate()) {
      case llvm::CmpInst::FCMP_OEQ:
        return absl::StrCat("(!isnan(", lhs_expr, ") && !isnan(", rhs_expr,
                            ") && (", lhs_expr, " == ", rhs_expr, "))");
      case llvm::CmpInst::FCMP_UEQ:
        return absl::StrCat("(isnan(", lhs_expr, ") || isnan(", rhs_expr,
                            ") || (", lhs_expr, " == ", rhs_expr, "))");
      case llvm::CmpInst::FCMP_ONE:
        return absl::StrCat("(!isnan(", lhs_expr, ") && !isnan(", rhs_expr,
                            ") && (", lhs_expr, " != ", rhs_expr, "))");
      case llvm::CmpInst::FCMP_UNE:
        return absl::StrCat("(isnan(", lhs_expr, ") || isnan(", rhs_expr,
                            ") || (", lhs_expr, " != ", rhs_expr, "))");
      case llvm::CmpInst::FCMP_OGT:
        return absl::StrCat("(!isnan(", lhs_expr, ") && !isnan(", rhs_expr,
                            ") && (", lhs_expr, " > ", rhs_expr, "))");
      case llvm::CmpInst::FCMP_UGT:
        return absl::StrCat("(isnan(", lhs_expr, ") || isnan(", rhs_expr,
                            ") || (", lhs_expr, " > ", rhs_expr, "))");
      case llvm::CmpInst::FCMP_OGE:
        return absl::StrCat("(!isnan(", lhs_expr, ") && !isnan(", rhs_expr,
                            ") && (", lhs_expr, " >= ", rhs_expr, "))");
      case llvm::CmpInst::FCMP_UGE:
        return absl::StrCat("(isnan(", lhs_expr, ") || isnan(", rhs_expr,
                            ") || (", lhs_expr, " >= ", rhs_expr, "))");
      case llvm::CmpInst::FCMP_OLT:
        return absl::StrCat("(!isnan(", lhs_expr, ") && !isnan(", rhs_expr,
                            ") && (", lhs_expr, " < ", rhs_expr, "))");
      case llvm::CmpInst::FCMP_ULT:
        return absl::StrCat("(isnan(", lhs_expr, ") || isnan(", rhs_expr,
                            ") || (", lhs_expr, " < ", rhs_expr, "))");
      case llvm::CmpInst::FCMP_OLE:
        return absl::StrCat("(!isnan(", lhs_expr, ") && !isnan(", rhs_expr,
                            ") && (", lhs_expr, " <= ", rhs_expr, "))");
      case llvm::CmpInst::FCMP_ULE:
        return absl::StrCat("(isnan(", lhs_expr, ") || isnan(", rhs_expr,
                            ") || (", lhs_expr, " <= ", rhs_expr, "))");
      case llvm::CmpInst::FCMP_FALSE:
        return "false";
      case llvm::CmpInst::FCMP_TRUE:
        return "true";
      case llvm::CmpInst::FCMP_ORD:
        return absl::StrCat("(!isnan(", lhs_expr, ") && !isnan(", rhs_expr,
                            "))");
      case llvm::CmpInst::FCMP_UNO:
        return absl::StrCat("(isnan(", lhs_expr, ") || isnan(", rhs_expr,
                            "))");
      default:
        return Unsupported(function_, "floating-point comparison", fcmp);
    }
  }

  absl::StatusOr<std::string> CastExpr(const llvm::CastInst& cast) {
    if (cast.getOpcode() == llvm::Instruction::PtrToInt) {
      return PointerToIntegerExpr(cast);
    }
    TF_ASSIGN_OR_RETURN(std::string operand, Expr(cast.getOperand(0)));
    return CastExprWithOperand(cast, operand);
  }

  absl::StatusOr<std::string> CastExprWithOperand(const llvm::CastInst& cast,
                                                  absl::string_view operand) {
    TF_ASSIGN_OR_RETURN(std::string type, MslType(cast.getType()));
    if (cast.getOpcode() == llvm::Instruction::FPTrunc &&
        cast.getType()->isBFloatTy() &&
        cast.getOperand(0)->getType()->isFloatTy()) {
      return absl::StrCat("xla_metal_f32_to_bf16(", operand, ")");
    }
    if (cast.getOpcode() == llvm::Instruction::FPExt &&
        cast.getOperand(0)->getType()->isBFloatTy() &&
        cast.getType()->isFloatTy()) {
      return absl::StrCat("xla_metal_bf16_to_f32(", operand, ")");
    }
    if ((cast.getOpcode() == llvm::Instruction::SIToFP ||
         cast.getOpcode() == llvm::Instruction::UIToFP) &&
        cast.getType()->isBFloatTy()) {
      return absl::StrCat("xla_metal_f32_to_bf16(static_cast<float>(",
                          operand, "))");
    }
    if ((cast.getOpcode() == llvm::Instruction::FPToSI ||
         cast.getOpcode() == llvm::Instruction::FPToUI) &&
        cast.getOperand(0)->getType()->isBFloatTy()) {
      return absl::StrCat("static_cast<", type, ">(xla_metal_bf16_to_f32(",
                          operand, "))");
    }
    switch (cast.getOpcode()) {
      case llvm::Instruction::Trunc:
      case llvm::Instruction::ZExt:
      case llvm::Instruction::SExt:
      case llvm::Instruction::FPTrunc:
      case llvm::Instruction::FPExt:
      case llvm::Instruction::SIToFP:
      case llvm::Instruction::UIToFP:
      case llvm::Instruction::FPToSI:
      case llvm::Instruction::FPToUI:
        return absl::StrCat("static_cast<", type, ">(", operand, ")");
      case llvm::Instruction::BitCast:
        return absl::StrCat("as_type<", type, ">(", operand, ")");
      default:
        return Unsupported(function_, "cast", cast);
    }
  }

  absl::StatusOr<std::string> PointerToIntegerExpr(
      const llvm::CastInst& cast) {
    TF_ASSIGN_OR_RETURN(PtrExpr pointer, PointerExprFor(cast.getOperand(0)));
    TF_ASSIGN_OR_RETURN(std::string type, MslType(cast.getType()));
    std::optional<int64_t> element_size =
        FixedTypeSize(data_layout_, pointer.element_type);
    if (!element_size.has_value()) {
      return Unsupported(function_, "pointer-to-integer", cast);
    }
    return absl::StrCat("static_cast<", type, ">(",
                        ScaleIndex(pointer.index, *element_size), ")");
  }

  absl::StatusOr<std::string> CallExpr(const llvm::CallInst& call) {
    const llvm::Function* callee = call.getCalledFunction();
    if (callee == nullptr) return Unsupported(function_, "indirect call", call);
    if (!callee->isDeclaration()) {
      return InlineSimpleFunctionCall(call);
    }
    llvm::StringRef name = callee->getName();

    if (name == "llvm.nvvm.read.ptx.sreg.tid.x") return "metal_tid.x";
    if (name == "llvm.nvvm.read.ptx.sreg.tid.y") return "metal_tid.y";
    if (name == "llvm.nvvm.read.ptx.sreg.tid.z") return "metal_tid.z";
    if (name == "llvm.nvvm.read.ptx.sreg.ctaid.x") return "metal_bid.x";
    if (name == "llvm.nvvm.read.ptx.sreg.ctaid.y") return "metal_bid.y";
    if (name == "llvm.nvvm.read.ptx.sreg.ctaid.z") return "metal_bid.z";
    if (name == "llvm.nvvm.read.ptx.sreg.ntid.x") return "metal_ntid.x";
    if (name == "llvm.nvvm.read.ptx.sreg.ntid.y") return "metal_ntid.y";
    if (name == "llvm.nvvm.read.ptx.sreg.ntid.z") return "metal_ntid.z";
    if (name == "llvm.nvvm.read.ptx.sreg.nctaid.x") return "metal_nbid.x";
    if (name == "llvm.nvvm.read.ptx.sreg.nctaid.y") return "metal_nbid.y";
    if (name == "llvm.nvvm.read.ptx.sreg.nctaid.z") return "metal_nbid.z";
    if (name == "llvm.nvvm.shfl.sync.down.f32" ||
        name == "llvm.nvvm.shfl.sync.down.i32") {
      if (call.arg_size() != 4) {
        return Unsupported(function_, "shuffle-down call", call);
      }
      TF_ASSIGN_OR_RETURN(std::string value, Expr(call.getArgOperand(1)));
      TF_ASSIGN_OR_RETURN(std::string delta, Expr(call.getArgOperand(2)));
      return absl::StrCat("simd_shuffle_down(", value, ", ", delta, ")");
    }

    if (name.starts_with("llvm.fabs.")) return UnaryMathCall("fabs", call);
    if (name.starts_with("llvm.sqrt.")) return UnaryMathCall("sqrt", call);
    if (name.starts_with("llvm.sin.")) return UnaryMathCall("sin", call);
    if (name.starts_with("llvm.cos.")) return UnaryMathCall("cos", call);
    if (name.starts_with("llvm.exp.")) return UnaryMathCall("exp", call);
    if (name.starts_with("llvm.log.")) return UnaryMathCall("log", call);
    if (name.starts_with("llvm.floor.")) return UnaryMathCall("floor", call);
    if (name.starts_with("llvm.ceil.")) return UnaryMathCall("ceil", call);
    if (name.starts_with("llvm.round.")) return UnaryMathCall("round", call);
    if (name.starts_with("llvm.pow.")) return BinaryMathCall("pow", call);
    if (name.starts_with("llvm.copysign.")) {
      return BinaryMathCall("copysign", call);
    }
    if (name.starts_with("llvm.ctpop.")) {
      return IntegerBitCountCall("popcount", call);
    }
    if (name.starts_with("llvm.ctlz.")) {
      return IntegerBitCountCall("clz", call);
    }
    if (name.starts_with("llvm.maximum.")) return BinaryMathCall("max", call);
    if (name.starts_with("llvm.minimum.")) return BinaryMathCall("min", call);
    if (name.starts_with("llvm.smax.")) return BinaryMathCall("max", call);
    if (name.starts_with("llvm.smin.")) return BinaryMathCall("min", call);
    if (name.starts_with("llvm.umax.")) {
      return UnsignedIntegerBinaryCall("max", call);
    }
    if (name.starts_with("llvm.umin.")) {
      return UnsignedIntegerBinaryCall("min", call);
    }

    if (name == "__nv_fabsf") return UnaryMathCall("fabs", call);
    if (name == "__nv_sqrtf") return UnaryMathCall("sqrt", call);
    if (name == "__nv_sinf") return UnaryMathCall("sin", call);
    if (name == "__nv_cosf") return UnaryMathCall("cos", call);
    if (name == "__nv_expf") return UnaryMathCall("exp", call);
    if (name == "__nv_logf") return UnaryMathCall("log", call);
    if (name == "__nv_log1pf") return Log1pCall(call);
    if (name == "__nv_floorf") return UnaryMathCall("floor", call);
    if (name == "__nv_ceilf") return UnaryMathCall("ceil", call);
    if (name == "__nv_roundf") return UnaryMathCall("round", call);
    if (name == "__nv_rintf") return UnaryMathCall("rint", call);
    if (name == "__nv_powf") return BinaryMathCall("pow", call);
    if (name == "__nv_atan2f") return BinaryMathCall("atan2", call);
    if (name == "__nv_copysignf") return BinaryMathCall("copysign", call);
    if (name == "__nv_fmodf") return BinaryMathCall("fmod", call);
    if (name == "__nv_erff") return UnaryMathCall("erf", call);
    if (name == "__nv_tanhf") return UnaryMathCall("tanh", call);
    if (name == "__nv_fmaf") return TernaryMathCall("fma", call);

    return Unsupported(function_, "call", call);
  }

  absl::StatusOr<std::string> InlineSimpleFunctionCall(
      const llvm::CallInst& call) {
    const llvm::Function* callee = call.getCalledFunction();
    if (callee == nullptr || callee->isDeclaration()) {
      return Unsupported(function_, "inline call", call);
    }
    if (callee->arg_size() != call.arg_size()) {
      return Unsupported(function_, "inline function shape", call);
    }
    if (callee->size() != 1) {
      if (absl::StatusOr<std::string> expression =
              InlineIfElsePhiFunctionCall(call);
          expression.ok()) {
        return *expression;
      }
      TF_ASSIGN_OR_RETURN(const llvm::ReturnInst* ret,
                          SingleReturnInst(*callee, call));
      return InlineSimpleValue(ret->getReturnValue(), call);
    }
    const llvm::BasicBlock& block = callee->getEntryBlock();
    const auto* ret = llvm::dyn_cast<llvm::ReturnInst>(block.getTerminator());
    if (ret == nullptr || ret->getReturnValue() == nullptr) {
      return Unsupported(function_, "inline function return", call);
    }
    return InlineSimpleValue(ret->getReturnValue(), call);
  }

  absl::StatusOr<std::string> InlineIfElsePhiFunctionCall(
      const llvm::CallInst& call) {
    const llvm::Function* callee = call.getCalledFunction();
    if (callee == nullptr || callee->isDeclaration()) {
      return Unsupported(function_, "inline call", call);
    }
    const llvm::BasicBlock& entry = callee->getEntryBlock();
    const auto* branch =
        llvm::dyn_cast<llvm::CondBrInst>(entry.getTerminator());
    if (branch == nullptr) {
      return Unsupported(function_, "inline function shape", call);
    }

    TF_ASSIGN_OR_RETURN(const llvm::PHINode* phi, ReturnPhi(*callee, call));
    const llvm::BasicBlock* merge_block = phi->getParent();

    TF_ASSIGN_OR_RETURN(std::string condition,
                        InlineSimpleValue(branch->getCondition(), call));
    TF_ASSIGN_OR_RETURN(
        std::string true_value,
        InlinePhiIncomingForPath(branch->getSuccessor(0), merge_block, *phi,
                                 call));
    TF_ASSIGN_OR_RETURN(
        std::string false_value,
        InlinePhiIncomingForPath(branch->getSuccessor(1), merge_block, *phi,
                                 call));
    return absl::StrCat("(", condition, " ? ", true_value, " : ",
                        false_value, ")");
  }

  absl::StatusOr<const llvm::PHINode*> ReturnPhi(
      const llvm::Function& callee, const llvm::CallInst& call) {
    TF_ASSIGN_OR_RETURN(const llvm::ReturnInst* ret,
                        SingleReturnInst(callee, call));
    const auto* phi = llvm::dyn_cast<llvm::PHINode>(ret->getReturnValue());
    if (phi == nullptr) {
      return Unsupported(function_, "inline function return", call);
    }
    return phi;
  }

  absl::StatusOr<const llvm::ReturnInst*> SingleReturnInst(
      const llvm::Function& callee, const llvm::CallInst& call) {
    const llvm::ReturnInst* ret = nullptr;
    for (const llvm::BasicBlock& block : callee) {
      if (const auto* block_ret =
              llvm::dyn_cast<llvm::ReturnInst>(block.getTerminator())) {
        if (ret != nullptr) {
          return Unsupported(function_, "inline function return", call);
        }
        ret = block_ret;
      }
    }
    if (ret == nullptr || ret->getReturnValue() == nullptr) {
      return Unsupported(function_, "inline function return", call);
    }
    return ret;
  }

  absl::StatusOr<const llvm::Value*> PhiIncomingValueOnLinearPath(
      const llvm::BasicBlock* block, const llvm::BasicBlock* merge_block,
      const llvm::PHINode& phi, const llvm::CallInst& call) {
    while (block != nullptr && block != merge_block) {
      int index = phi.getBasicBlockIndex(block);
      if (index >= 0) return phi.getIncomingValue(index);
      const auto* branch =
          llvm::dyn_cast<llvm::UncondBrInst>(block->getTerminator());
      if (branch == nullptr) break;
      block = branch->getSuccessor(0);
    }
    return Unsupported(function_, "inline function return", call);
  }

  absl::StatusOr<std::string> InlinePhiIncomingForPath(
      const llvm::BasicBlock* block, const llvm::BasicBlock* merge_block,
      const llvm::PHINode& phi, const llvm::CallInst& call) {
    if (absl::StatusOr<const llvm::Value*> incoming =
            PhiIncomingValueOnLinearPath(block, merge_block, phi, call);
        incoming.ok()) {
      return InlineSimpleValue(*incoming, call);
    }

    const auto* branch =
        llvm::dyn_cast<llvm::CondBrInst>(block->getTerminator());
    if (branch == nullptr) {
      return Unsupported(function_, "inline function shape", call);
    }
    const llvm::BasicBlock* true_successor =
        UnconditionalBranchSuccessor(*branch->getSuccessor(0));
    const llvm::BasicBlock* false_successor =
        UnconditionalBranchSuccessor(*branch->getSuccessor(1));
    if (true_successor == nullptr || true_successor != false_successor) {
      return Unsupported(function_, "inline function shape", call);
    }

    const llvm::BasicBlock* local_merge = true_successor;
    TF_ASSIGN_OR_RETURN(
        const llvm::Value* incoming,
        PhiIncomingValueOnLinearPath(local_merge, merge_block, phi, call));
    const auto* local_phi = llvm::dyn_cast<llvm::PHINode>(incoming);
    if (local_phi == nullptr || local_phi->getParent() != local_merge) {
      return InlineSimpleValue(incoming, call);
    }

    TF_ASSIGN_OR_RETURN(std::string condition,
                        InlineSimpleValue(branch->getCondition(), call));
    TF_ASSIGN_OR_RETURN(
        std::string true_value,
        InlinePhiIncomingForPath(branch->getSuccessor(0), local_merge,
                                 *local_phi, call));
    TF_ASSIGN_OR_RETURN(
        std::string false_value,
        InlinePhiIncomingForPath(branch->getSuccessor(1), local_merge,
                                 *local_phi, call));
    return absl::StrCat("(", condition, " ? ", true_value, " : ",
                        false_value, ")");
  }

  absl::StatusOr<std::string> InlineSimplePhiValue(
      const llvm::PHINode& phi, const llvm::CallInst& call) {
    const llvm::BasicBlock* merge_block = phi.getParent();
    const llvm::Function* function = merge_block->getParent();
    for (const llvm::BasicBlock& block : *function) {
      const auto* branch =
          llvm::dyn_cast<llvm::CondBrInst>(block.getTerminator());
      if (branch == nullptr) continue;
      absl::StatusOr<std::string> true_value = InlinePhiIncomingForPath(
          branch->getSuccessor(0), merge_block, phi, call);
      if (!true_value.ok()) continue;
      absl::StatusOr<std::string> false_value = InlinePhiIncomingForPath(
          branch->getSuccessor(1), merge_block, phi, call);
      if (!false_value.ok()) continue;
      TF_ASSIGN_OR_RETURN(std::string condition,
                          InlineSimpleValue(branch->getCondition(), call));
      return absl::StrCat("(", condition, " ? ", *true_value, " : ",
                          *false_value, ")");
    }
    return Unsupported(function_, "inline phi", phi);
  }

  absl::StatusOr<std::string> InlineSimpleAggregateElement(
      const llvm::CallInst& call, unsigned field) {
    const llvm::Function* callee = call.getCalledFunction();
    if (callee == nullptr || callee->isDeclaration()) {
      return Unsupported(function_, "inline aggregate call", call);
    }
    if (callee->arg_size() != call.arg_size() || callee->size() != 1) {
      return Unsupported(function_, "inline aggregate function shape", call);
    }
    const llvm::BasicBlock& block = callee->getEntryBlock();
    const auto* ret = llvm::dyn_cast<llvm::ReturnInst>(block.getTerminator());
    if (ret == nullptr || ret->getReturnValue() == nullptr) {
      return Unsupported(function_, "inline aggregate return", call);
    }
    return AggregateElementValue(ret->getReturnValue(), &call, field);
  }

  absl::StatusOr<std::string> AggregateElementValue(
      const llvm::Value* value, const llvm::CallInst* outer_call,
      unsigned field) {
    if (auto* insert = llvm::dyn_cast<llvm::InsertValueInst>(value)) {
      if (insert->getNumIndices() != 1) {
        return Unsupported(function_, "nested aggregate insertion", *insert);
      }
      if (*insert->idx_begin() == field) {
        if (outer_call == nullptr) {
          return Expr(insert->getInsertedValueOperand());
        }
        return InlineSimpleValue(insert->getInsertedValueOperand(), *outer_call);
      }
      return AggregateElementValue(insert->getAggregateOperand(), outer_call,
                                   field);
    }

    if (auto* constant = llvm::dyn_cast<llvm::ConstantStruct>(value)) {
      if (field >= constant->getNumOperands()) {
        return absl::InvalidArgumentError(
            absl::StrCat("Aggregate field ", field, " is out of range."));
      }
      return Expr(constant->getOperand(field));
    }

    if (llvm::isa<llvm::ConstantAggregateZero>(value) ||
        llvm::isa<llvm::UndefValue>(value) ||
        llvm::isa<llvm::PoisonValue>(value)) {
      auto* struct_type = llvm::dyn_cast<llvm::StructType>(value->getType());
      if (struct_type == nullptr || field >= struct_type->getNumElements()) {
        return Unsupported(function_, "aggregate initializer", *value);
      }
      return ZeroInitializer(struct_type->getElementType(field));
    }

    return Unsupported(function_, "aggregate value", *value);
  }

  absl::StatusOr<std::string> InlineSimpleValue(
      const llvm::Value* value, const llvm::CallInst& call) {
    if (auto* argument = llvm::dyn_cast<llvm::Argument>(value)) {
      return Expr(call.getArgOperand(argument->getArgNo()));
    }
    if (llvm::isa<llvm::Constant>(value)) {
      return Expr(value);
    }
    if (auto* phi = llvm::dyn_cast<llvm::PHINode>(value)) {
      return InlineSimplePhiValue(*phi, call);
    }
    if (auto* binary = llvm::dyn_cast<llvm::BinaryOperator>(value)) {
      TF_ASSIGN_OR_RETURN(std::string lhs,
                          InlineSimpleValue(binary->getOperand(0), call));
      TF_ASSIGN_OR_RETURN(std::string rhs,
                          InlineSimpleValue(binary->getOperand(1), call));
      return BinaryExprWithOperands(*binary, lhs, rhs);
    }
    if (auto* instruction = llvm::dyn_cast<llvm::Instruction>(value);
        instruction != nullptr &&
        instruction->getOpcode() == llvm::Instruction::FNeg) {
      TF_ASSIGN_OR_RETURN(std::string operand,
                          InlineSimpleValue(instruction->getOperand(0), call));
      return absl::StrCat("(-", operand, ")");
    }
    if (auto* icmp = llvm::dyn_cast<llvm::ICmpInst>(value)) {
      bool unsigned_compare = icmp->isUnsigned();
      TF_ASSIGN_OR_RETURN(std::string lhs,
                          InlineSimpleValue(icmp->getOperand(0), call));
      TF_ASSIGN_OR_RETURN(std::string rhs,
                          InlineSimpleValue(icmp->getOperand(1), call));
      TF_ASSIGN_OR_RETURN(lhs,
                          CastForUnsignedCompare(icmp->getOperand(0)->getType(),
                                                 lhs, unsigned_compare));
      TF_ASSIGN_OR_RETURN(rhs,
                          CastForUnsignedCompare(icmp->getOperand(1)->getType(),
                                                 rhs, unsigned_compare));
      return ICmpExprWithOperands(*icmp, lhs, rhs);
    }
    if (auto* fcmp = llvm::dyn_cast<llvm::FCmpInst>(value)) {
      TF_ASSIGN_OR_RETURN(std::string lhs,
                          InlineSimpleValue(fcmp->getOperand(0), call));
      TF_ASSIGN_OR_RETURN(std::string rhs,
                          InlineSimpleValue(fcmp->getOperand(1), call));
      return FCmpExprWithOperands(*fcmp, lhs, rhs);
    }
    if (auto* select = llvm::dyn_cast<llvm::SelectInst>(value)) {
      TF_ASSIGN_OR_RETURN(std::string condition,
                          InlineSimpleValue(select->getCondition(), call));
      TF_ASSIGN_OR_RETURN(std::string true_value,
                          InlineSimpleValue(select->getTrueValue(), call));
      TF_ASSIGN_OR_RETURN(std::string false_value,
                          InlineSimpleValue(select->getFalseValue(), call));
      return absl::StrCat("(", condition, " ? ", true_value, " : ",
                          false_value, ")");
    }
    if (auto* insert = llvm::dyn_cast<llvm::InsertValueInst>(value)) {
      return InlineComplexInsertValueExpr(*insert, call);
    }
    if (auto* extract = llvm::dyn_cast<llvm::ExtractValueInst>(value)) {
      return InlineExtractValueExpr(*extract, call);
    }
    if (auto* load = llvm::dyn_cast<llvm::LoadInst>(value)) {
      return InlineSimpleLoadExpr(*load, call);
    }
    if (auto* cast = llvm::dyn_cast<llvm::CastInst>(value)) {
      TF_ASSIGN_OR_RETURN(std::string operand,
                          InlineSimpleValue(cast->getOperand(0), call));
      return CastExprWithOperand(*cast, operand);
    }
    if (auto* nested_call = llvm::dyn_cast<llvm::CallInst>(value)) {
      return InlineSimpleCall(*nested_call, call);
    }
    return Unsupported(function_, "inline value", *value);
  }

  absl::StatusOr<std::string> InlineComplexInsertValueExpr(
      const llvm::InsertValueInst& insert, const llvm::CallInst& call) {
    if (!IsComplexFloatStruct(insert.getType()) || insert.getNumIndices() != 1) {
      return Unsupported(function_, "inline aggregate insertion", insert);
    }

    unsigned field = *insert.idx_begin();
    if (field > 1) {
      return absl::InvalidArgumentError(
          absl::StrCat("complex field ", field, " is out of range."));
    }

    std::string components[2] = {"0.0f", "0.0f"};
    const llvm::Value* aggregate = insert.getAggregateOperand();
    if (!llvm::isa<llvm::UndefValue>(aggregate) &&
        !llvm::isa<llvm::PoisonValue>(aggregate)) {
      TF_ASSIGN_OR_RETURN(components[0],
                          InlineComplexComponentExpr(aggregate, call, 0));
      TF_ASSIGN_OR_RETURN(components[1],
                          InlineComplexComponentExpr(aggregate, call, 1));
    }
    TF_ASSIGN_OR_RETURN(
        std::string inserted,
        InlineSimpleValue(insert.getInsertedValueOperand(), call));
    components[field] = std::move(inserted);
    return absl::StrCat("float2(", components[0], ", ", components[1], ")");
  }

  absl::StatusOr<std::string> InlineComplexComponentExpr(
      const llvm::Value* value, const llvm::CallInst& call, unsigned field) {
    if (field > 1) {
      return absl::InvalidArgumentError(
          absl::StrCat("complex field ", field, " is out of range."));
    }
    if (llvm::isa<llvm::UndefValue>(value) ||
        llvm::isa<llvm::PoisonValue>(value)) {
      return "0.0f";
    }
    if (auto* insert = llvm::dyn_cast<llvm::InsertValueInst>(value)) {
      if (insert->getNumIndices() != 1) {
        return Unsupported(function_, "inline aggregate insertion", *insert);
      }
      if (*insert->idx_begin() == field) {
        return InlineSimpleValue(insert->getInsertedValueOperand(), call);
      }
      return InlineComplexComponentExpr(insert->getAggregateOperand(), call,
                                        field);
    }
    TF_ASSIGN_OR_RETURN(std::string expression, InlineSimpleValue(value, call));
    return absl::StrCat(expression, field == 0 ? ".x" : ".y");
  }

  absl::StatusOr<std::string> InlineExtractValueExpr(
      const llvm::ExtractValueInst& extract, const llvm::CallInst& call) {
    if (extract.getNumIndices() != 1) {
      return Unsupported(function_, "inline nested aggregate extraction",
                         extract);
    }
    const llvm::Value* aggregate = extract.getAggregateOperand();
    unsigned field = *extract.idx_begin();
    if (IsComplexFloatStruct(aggregate->getType())) {
      TF_ASSIGN_OR_RETURN(std::string value,
                          InlineSimpleValue(aggregate, call));
      if (field == 0) return absl::StrCat(value, ".x");
      if (field == 1) return absl::StrCat(value, ".y");
      return absl::InvalidArgumentError(
          absl::StrCat("complex field ", field, " is out of range."));
    }
    if (auto* nested_call = llvm::dyn_cast<llvm::CallInst>(aggregate)) {
      return AggregateElementValue(nested_call, &call, field);
    }
    return AggregateElementValue(aggregate, &call, field);
  }

  absl::StatusOr<std::string> InlineSimpleLoadExpr(
      const llvm::LoadInst& load, const llvm::CallInst& call) {
    const absl::flat_hash_map<const llvm::Value*, StoredInlineValue> values;
    const absl::flat_hash_map<const llvm::Value*, StoredInlineValue> memory;
    return InlineLoadExpr(load, call, values, memory);
  }

  absl::StatusOr<std::string> InlineSimpleCall(
      const llvm::CallInst& nested_call, const llvm::CallInst& outer_call) {
    const llvm::Function* callee = nested_call.getCalledFunction();
    if (callee == nullptr) {
      return Unsupported(function_, "inline call", nested_call);
    }
    if (!callee->isDeclaration()) {
      return InlineNestedSimpleFunctionCall(nested_call, outer_call);
    }
    llvm::StringRef name = callee->getName();
    if (name.starts_with("llvm.maximum.") || name.starts_with("llvm.smax.")) {
      return InlineSimpleBinaryCall("max", nested_call, outer_call);
    }
    if (name.starts_with("llvm.minimum.") || name.starts_with("llvm.smin.")) {
      return InlineSimpleBinaryCall("min", nested_call, outer_call);
    }
    if (name.starts_with("llvm.fabs.")) {
      return InlineSimpleUnaryCall("fabs", nested_call, outer_call);
    }
    if (name.starts_with("llvm.sqrt.")) {
      return InlineSimpleUnaryCall("sqrt", nested_call, outer_call);
    }
    if (name.starts_with("llvm.floor.")) {
      return InlineSimpleUnaryCall("floor", nested_call, outer_call);
    }
    if (name.starts_with("llvm.ceil.")) {
      return InlineSimpleUnaryCall("ceil", nested_call, outer_call);
    }
    if (name.starts_with("llvm.umax.")) {
      return InlineSimpleUnsignedBinaryCall("max", nested_call, outer_call);
    }
    if (name.starts_with("llvm.umin.")) {
      return InlineSimpleUnsignedBinaryCall("min", nested_call, outer_call);
    }
    if (name == "__nv_fabsf") {
      return InlineSimpleUnaryCall("fabs", nested_call, outer_call);
    }
    if (name == "__nv_sqrtf") {
      return InlineSimpleUnaryCall("sqrt", nested_call, outer_call);
    }
    if (name == "__nv_floorf") {
      return InlineSimpleUnaryCall("floor", nested_call, outer_call);
    }
    if (name == "__nv_ceilf") {
      return InlineSimpleUnaryCall("ceil", nested_call, outer_call);
    }
    return Unsupported(function_, "inline call", nested_call);
  }

  absl::StatusOr<std::string> InlineNestedSimpleFunctionCall(
      const llvm::CallInst& nested_call, const llvm::CallInst& outer_call) {
    const llvm::Function* callee = nested_call.getCalledFunction();
    if (callee == nullptr || callee->isDeclaration()) {
      return Unsupported(function_, "nested inline call", nested_call);
    }
    if (callee->arg_size() != nested_call.arg_size()) {
      return Unsupported(function_, "nested inline function shape",
                         nested_call);
    }
    if (callee->size() != 1) {
      if (absl::StatusOr<std::string> expression =
              InlineNestedIfElsePhiFunctionCall(nested_call, outer_call);
          expression.ok()) {
        return *expression;
      }
      TF_ASSIGN_OR_RETURN(const llvm::ReturnInst* ret,
                          SingleReturnInst(*callee, nested_call));
      return InlineNestedSimpleValue(ret->getReturnValue(), nested_call,
                                     outer_call);
    }
    const llvm::BasicBlock& block = callee->getEntryBlock();
    const auto* ret = llvm::dyn_cast<llvm::ReturnInst>(block.getTerminator());
    if (ret == nullptr || ret->getReturnValue() == nullptr) {
      return Unsupported(function_, "nested inline function return",
                         nested_call);
    }
    return InlineNestedSimpleValue(ret->getReturnValue(), nested_call,
                                   outer_call);
  }

  absl::StatusOr<std::string> InlineNestedIfElsePhiFunctionCall(
      const llvm::CallInst& nested_call, const llvm::CallInst& outer_call) {
    const llvm::Function* callee = nested_call.getCalledFunction();
    if (callee == nullptr || callee->isDeclaration()) {
      return Unsupported(function_, "nested inline call", nested_call);
    }
    const llvm::BasicBlock& entry = callee->getEntryBlock();
    const auto* branch =
        llvm::dyn_cast<llvm::CondBrInst>(entry.getTerminator());
    if (branch == nullptr) {
      return Unsupported(function_, "nested inline function shape",
                         nested_call);
    }

    const llvm::BasicBlock* true_block = branch->getSuccessor(0);
    const llvm::BasicBlock* false_block = branch->getSuccessor(1);
    const llvm::BasicBlock* true_successor =
        UnconditionalBranchSuccessor(*true_block);
    const llvm::BasicBlock* false_successor =
        UnconditionalBranchSuccessor(*false_block);
    if (true_successor == nullptr || true_successor != false_successor) {
      return Unsupported(function_, "nested inline function shape",
                         nested_call);
    }

    const llvm::BasicBlock* merge_block = true_successor;
    const llvm::ReturnInst* ret =
        llvm::dyn_cast<llvm::ReturnInst>(merge_block->getTerminator());
    if (ret == nullptr) {
      const llvm::BasicBlock* return_block =
          UnconditionalBranchSuccessor(*merge_block);
      if (return_block == nullptr) {
        return Unsupported(function_, "nested inline function shape",
                           nested_call);
      }
      ret = llvm::dyn_cast<llvm::ReturnInst>(return_block->getTerminator());
    }
    if (ret == nullptr || ret->getReturnValue() == nullptr) {
      return Unsupported(function_, "nested inline function return",
                         nested_call);
    }

    const auto* phi = llvm::dyn_cast<llvm::PHINode>(ret->getReturnValue());
    if (phi == nullptr || phi->getParent() != merge_block) {
      return Unsupported(function_, "nested inline function return",
                         nested_call);
    }
    int true_index = phi->getBasicBlockIndex(true_block);
    int false_index = phi->getBasicBlockIndex(false_block);
    if (true_index < 0 || false_index < 0) {
      return Unsupported(function_, "nested inline function return",
                         nested_call);
    }

    TF_ASSIGN_OR_RETURN(
        std::string condition,
        InlineNestedSimpleValue(branch->getCondition(), nested_call,
                                outer_call));
    TF_ASSIGN_OR_RETURN(
        std::string true_value,
        InlineNestedSimpleValue(phi->getIncomingValue(true_index), nested_call,
                                outer_call));
    TF_ASSIGN_OR_RETURN(
        std::string false_value,
        InlineNestedSimpleValue(phi->getIncomingValue(false_index), nested_call,
                                outer_call));
    return absl::StrCat("(", condition, " ? ", true_value, " : ",
                        false_value, ")");
  }

  absl::StatusOr<std::string> InlineNestedSimpleValue(
      const llvm::Value* value, const llvm::CallInst& nested_call,
      const llvm::CallInst& outer_call) {
    if (auto* argument = llvm::dyn_cast<llvm::Argument>(value)) {
      return InlineSimpleValue(nested_call.getArgOperand(argument->getArgNo()),
                               outer_call);
    }
    if (llvm::isa<llvm::Constant>(value)) return Expr(value);
    if (auto* phi = llvm::dyn_cast<llvm::PHINode>(value)) {
      return InlineNestedPhiValue(*phi, nested_call, outer_call);
    }
    if (auto* binary = llvm::dyn_cast<llvm::BinaryOperator>(value)) {
      TF_ASSIGN_OR_RETURN(
          std::string lhs,
          InlineNestedSimpleValue(binary->getOperand(0), nested_call,
                                  outer_call));
      TF_ASSIGN_OR_RETURN(
          std::string rhs,
          InlineNestedSimpleValue(binary->getOperand(1), nested_call,
                                  outer_call));
      return BinaryExprWithOperands(*binary, lhs, rhs);
    }
    if (auto* instruction = llvm::dyn_cast<llvm::Instruction>(value);
        instruction != nullptr &&
        instruction->getOpcode() == llvm::Instruction::FNeg) {
      TF_ASSIGN_OR_RETURN(
          std::string operand,
          InlineNestedSimpleValue(instruction->getOperand(0), nested_call,
                                  outer_call));
      return absl::StrCat("(-", operand, ")");
    }
    if (auto* icmp = llvm::dyn_cast<llvm::ICmpInst>(value)) {
      bool unsigned_compare = icmp->isUnsigned();
      TF_ASSIGN_OR_RETURN(
          std::string lhs,
          InlineNestedSimpleValue(icmp->getOperand(0), nested_call,
                                  outer_call));
      TF_ASSIGN_OR_RETURN(
          std::string rhs,
          InlineNestedSimpleValue(icmp->getOperand(1), nested_call,
                                  outer_call));
      TF_ASSIGN_OR_RETURN(lhs,
                          CastForUnsignedCompare(icmp->getOperand(0)->getType(),
                                                 lhs, unsigned_compare));
      TF_ASSIGN_OR_RETURN(rhs,
                          CastForUnsignedCompare(icmp->getOperand(1)->getType(),
                                                 rhs, unsigned_compare));
      return ICmpExprWithOperands(*icmp, lhs, rhs);
    }
    if (auto* fcmp = llvm::dyn_cast<llvm::FCmpInst>(value)) {
      TF_ASSIGN_OR_RETURN(
          std::string lhs,
          InlineNestedSimpleValue(fcmp->getOperand(0), nested_call,
                                  outer_call));
      TF_ASSIGN_OR_RETURN(
          std::string rhs,
          InlineNestedSimpleValue(fcmp->getOperand(1), nested_call,
                                  outer_call));
      return FCmpExprWithOperands(*fcmp, lhs, rhs);
    }
    if (auto* select = llvm::dyn_cast<llvm::SelectInst>(value)) {
      TF_ASSIGN_OR_RETURN(
          std::string condition,
          InlineNestedSimpleValue(select->getCondition(), nested_call,
                                  outer_call));
      TF_ASSIGN_OR_RETURN(
          std::string true_value,
          InlineNestedSimpleValue(select->getTrueValue(), nested_call,
                                  outer_call));
      TF_ASSIGN_OR_RETURN(
          std::string false_value,
          InlineNestedSimpleValue(select->getFalseValue(), nested_call,
                                  outer_call));
      return absl::StrCat("(", condition, " ? ", true_value, " : ",
                          false_value, ")");
    }
    if (auto* cast = llvm::dyn_cast<llvm::CastInst>(value)) {
      TF_ASSIGN_OR_RETURN(
          std::string operand,
          InlineNestedSimpleValue(cast->getOperand(0), nested_call,
                                  outer_call));
      return CastExprWithOperand(*cast, operand);
    }
    if (auto* load = llvm::dyn_cast<llvm::LoadInst>(value)) {
      return InlineNestedLoadExpr(*load, nested_call, outer_call);
    }
    if (auto* call = llvm::dyn_cast<llvm::CallInst>(value)) {
      return InlineNestedSimpleCall(*call, nested_call, outer_call);
    }
    return Unsupported(function_, "nested inline value", *value);
  }

  absl::StatusOr<std::string> InlineNestedSimpleCall(
      const llvm::CallInst& call, const llvm::CallInst& nested_call,
      const llvm::CallInst& outer_call) {
    const llvm::Function* callee = call.getCalledFunction();
    if (callee == nullptr) {
      return Unsupported(function_, "nested inline call", call);
    }
    if (!callee->isDeclaration()) {
      return InlineNestedSimpleFunctionCallThrough(call, nested_call,
                                                   outer_call);
    }
    llvm::StringRef name = callee->getName();
    if (name.starts_with("llvm.maximum.") || name.starts_with("llvm.smax.")) {
      return InlineNestedBinaryCall("max", call, nested_call, outer_call);
    }
    if (name.starts_with("llvm.minimum.") || name.starts_with("llvm.smin.")) {
      return InlineNestedBinaryCall("min", call, nested_call, outer_call);
    }
    if (name.starts_with("llvm.fabs.")) {
      return InlineNestedUnaryCall("fabs", call, nested_call, outer_call);
    }
    if (name.starts_with("llvm.sqrt.")) {
      return InlineNestedUnaryCall("sqrt", call, nested_call, outer_call);
    }
    if (name.starts_with("llvm.floor.")) {
      return InlineNestedUnaryCall("floor", call, nested_call, outer_call);
    }
    if (name.starts_with("llvm.ceil.")) {
      return InlineNestedUnaryCall("ceil", call, nested_call, outer_call);
    }
    if (name.starts_with("llvm.umax.")) {
      return InlineNestedUnsignedBinaryCall("max", call, nested_call,
                                            outer_call);
    }
    if (name.starts_with("llvm.umin.")) {
      return InlineNestedUnsignedBinaryCall("min", call, nested_call,
                                            outer_call);
    }
    if (name == "__nv_fabsf") {
      return InlineNestedUnaryCall("fabs", call, nested_call, outer_call);
    }
    if (name == "__nv_sqrtf") {
      return InlineNestedUnaryCall("sqrt", call, nested_call, outer_call);
    }
    if (name == "__nv_floorf") {
      return InlineNestedUnaryCall("floor", call, nested_call, outer_call);
    }
    if (name == "__nv_ceilf") {
      return InlineNestedUnaryCall("ceil", call, nested_call, outer_call);
    }
    return Unsupported(function_, "nested inline call", call);
  }

  absl::StatusOr<std::string> InlineNestedSimpleFunctionCallThrough(
      const llvm::CallInst& call, const llvm::CallInst& nested_call,
      const llvm::CallInst& outer_call) {
    const llvm::Function* callee = call.getCalledFunction();
    if (callee == nullptr || callee->isDeclaration()) {
      return Unsupported(function_, "nested inline call", call);
    }
    if (callee->arg_size() != call.arg_size()) {
      return Unsupported(function_, "nested inline function shape", call);
    }
    const llvm::ReturnInst* ret = nullptr;
    if (callee->size() == 1) {
      const llvm::BasicBlock& block = callee->getEntryBlock();
      ret = llvm::dyn_cast<llvm::ReturnInst>(block.getTerminator());
      if (ret == nullptr || ret->getReturnValue() == nullptr) {
        return Unsupported(function_, "nested inline function return", call);
      }
    } else {
      TF_ASSIGN_OR_RETURN(ret, SingleReturnInst(*callee, call));
    }
    return InlineDoubleNestedSimpleValue(ret->getReturnValue(), call,
                                         nested_call, outer_call);
  }

  absl::StatusOr<std::string> InlineNestedPhiValue(
      const llvm::PHINode& phi, const llvm::CallInst& nested_call,
      const llvm::CallInst& outer_call) {
    const llvm::BasicBlock* merge_block = phi.getParent();
    const llvm::Function* function = merge_block->getParent();
    for (const llvm::BasicBlock& block : *function) {
      const auto* branch =
          llvm::dyn_cast<llvm::CondBrInst>(block.getTerminator());
      if (branch == nullptr) continue;
      absl::StatusOr<std::string> true_value =
          InlineNestedPhiIncomingForPath(branch->getSuccessor(0), merge_block,
                                        phi, nested_call, outer_call);
      if (!true_value.ok()) continue;
      absl::StatusOr<std::string> false_value =
          InlineNestedPhiIncomingForPath(branch->getSuccessor(1), merge_block,
                                        phi, nested_call, outer_call);
      if (!false_value.ok()) continue;
      TF_ASSIGN_OR_RETURN(
          std::string condition,
          InlineNestedSimpleValue(branch->getCondition(), nested_call,
                                  outer_call));
      return absl::StrCat("(", condition, " ? ", *true_value, " : ",
                          *false_value, ")");
    }
    return Unsupported(function_, "nested inline phi", phi);
  }

  absl::StatusOr<std::string> InlineNestedPhiIncomingForPath(
      const llvm::BasicBlock* block, const llvm::BasicBlock* merge_block,
      const llvm::PHINode& phi, const llvm::CallInst& nested_call,
      const llvm::CallInst& outer_call) {
    if (absl::StatusOr<const llvm::Value*> incoming =
            PhiIncomingValueOnLinearPath(block, merge_block, phi, nested_call);
        incoming.ok()) {
      return InlineNestedSimpleValue(*incoming, nested_call, outer_call);
    }

    const auto* branch =
        llvm::dyn_cast<llvm::CondBrInst>(block->getTerminator());
    if (branch == nullptr) {
      return Unsupported(function_, "nested inline function shape",
                         nested_call);
    }
    const llvm::BasicBlock* true_successor =
        UnconditionalBranchSuccessor(*branch->getSuccessor(0));
    const llvm::BasicBlock* false_successor =
        UnconditionalBranchSuccessor(*branch->getSuccessor(1));
    if (true_successor == nullptr || true_successor != false_successor) {
      return Unsupported(function_, "nested inline function shape",
                         nested_call);
    }

    const llvm::BasicBlock* local_merge = true_successor;
    TF_ASSIGN_OR_RETURN(const llvm::Value* incoming,
                        PhiIncomingValueOnLinearPath(
                            local_merge, merge_block, phi, nested_call));
    const auto* local_phi = llvm::dyn_cast<llvm::PHINode>(incoming);
    if (local_phi == nullptr || local_phi->getParent() != local_merge) {
      return InlineNestedSimpleValue(incoming, nested_call, outer_call);
    }

    TF_ASSIGN_OR_RETURN(
        std::string condition,
        InlineNestedSimpleValue(branch->getCondition(), nested_call,
                                outer_call));
    TF_ASSIGN_OR_RETURN(
        std::string true_value,
        InlineNestedPhiIncomingForPath(branch->getSuccessor(0), local_merge,
                                      *local_phi, nested_call, outer_call));
    TF_ASSIGN_OR_RETURN(
        std::string false_value,
        InlineNestedPhiIncomingForPath(branch->getSuccessor(1), local_merge,
                                      *local_phi, nested_call, outer_call));
    return absl::StrCat("(", condition, " ? ", true_value, " : ",
                        false_value, ")");
  }

  absl::StatusOr<std::string> InlineNestedBinaryCall(
      absl::string_view msl_name, const llvm::CallInst& call,
      const llvm::CallInst& nested_call, const llvm::CallInst& outer_call) {
    if (call.arg_size() != 2) {
      return Unsupported(function_, "nested inline binary call", call);
    }
    TF_ASSIGN_OR_RETURN(
        std::string lhs,
        InlineNestedSimpleValue(call.getArgOperand(0), nested_call,
                                outer_call));
    TF_ASSIGN_OR_RETURN(
        std::string rhs,
        InlineNestedSimpleValue(call.getArgOperand(1), nested_call,
                                outer_call));
    return absl::StrCat(msl_name, "(", lhs, ", ", rhs, ")");
  }

  absl::StatusOr<std::string> InlineNestedUnaryCall(
      absl::string_view msl_name, const llvm::CallInst& call,
      const llvm::CallInst& nested_call, const llvm::CallInst& outer_call) {
    if (call.arg_size() != 1) {
      return Unsupported(function_, "nested inline unary call", call);
    }
    TF_ASSIGN_OR_RETURN(
        std::string operand,
        InlineNestedSimpleValue(call.getArgOperand(0), nested_call,
                                outer_call));
    return absl::StrCat(msl_name, "(", operand, ")");
  }

  absl::StatusOr<std::string> InlineNestedUnsignedBinaryCall(
      absl::string_view msl_name, const llvm::CallInst& call,
      const llvm::CallInst& nested_call, const llvm::CallInst& outer_call) {
    if (call.arg_size() != 2) {
      return Unsupported(function_, "nested inline binary call", call);
    }
    TF_ASSIGN_OR_RETURN(
        std::string lhs,
        InlineNestedSimpleValue(call.getArgOperand(0), nested_call,
                                outer_call));
    TF_ASSIGN_OR_RETURN(
        std::string rhs,
        InlineNestedSimpleValue(call.getArgOperand(1), nested_call,
                                outer_call));
    return UnsignedIntegerBinaryExpr(msl_name, call.getType(), lhs, rhs);
  }

  absl::StatusOr<std::string> InlineDoubleNestedSimpleValue(
      const llvm::Value* value, const llvm::CallInst& call,
      const llvm::CallInst& nested_call, const llvm::CallInst& outer_call) {
    if (auto* argument = llvm::dyn_cast<llvm::Argument>(value)) {
      return InlineNestedSimpleValue(call.getArgOperand(argument->getArgNo()),
                                     nested_call, outer_call);
    }
    if (llvm::isa<llvm::Constant>(value)) return Expr(value);
    if (auto* phi = llvm::dyn_cast<llvm::PHINode>(value)) {
      return InlineDoubleNestedPhiValue(*phi, call, nested_call, outer_call);
    }
    if (auto* binary = llvm::dyn_cast<llvm::BinaryOperator>(value)) {
      TF_ASSIGN_OR_RETURN(std::string lhs,
                          InlineDoubleNestedSimpleValue(
                              binary->getOperand(0), call, nested_call,
                              outer_call));
      TF_ASSIGN_OR_RETURN(std::string rhs,
                          InlineDoubleNestedSimpleValue(
                              binary->getOperand(1), call, nested_call,
                              outer_call));
      return BinaryExprWithOperands(*binary, lhs, rhs);
    }
    if (auto* icmp = llvm::dyn_cast<llvm::ICmpInst>(value)) {
      bool unsigned_compare = icmp->isUnsigned();
      TF_ASSIGN_OR_RETURN(std::string lhs,
                          InlineDoubleNestedSimpleValue(
                              icmp->getOperand(0), call, nested_call,
                              outer_call));
      TF_ASSIGN_OR_RETURN(std::string rhs,
                          InlineDoubleNestedSimpleValue(
                              icmp->getOperand(1), call, nested_call,
                              outer_call));
      TF_ASSIGN_OR_RETURN(lhs,
                          CastForUnsignedCompare(icmp->getOperand(0)->getType(),
                                                 lhs, unsigned_compare));
      TF_ASSIGN_OR_RETURN(rhs,
                          CastForUnsignedCompare(icmp->getOperand(1)->getType(),
                                                 rhs, unsigned_compare));
      return ICmpExprWithOperands(*icmp, lhs, rhs);
    }
    if (auto* fcmp = llvm::dyn_cast<llvm::FCmpInst>(value)) {
      TF_ASSIGN_OR_RETURN(std::string lhs,
                          InlineDoubleNestedSimpleValue(
                              fcmp->getOperand(0), call, nested_call,
                              outer_call));
      TF_ASSIGN_OR_RETURN(std::string rhs,
                          InlineDoubleNestedSimpleValue(
                              fcmp->getOperand(1), call, nested_call,
                              outer_call));
      return FCmpExprWithOperands(*fcmp, lhs, rhs);
    }
    if (auto* cast = llvm::dyn_cast<llvm::CastInst>(value)) {
      TF_ASSIGN_OR_RETURN(std::string operand,
                          InlineDoubleNestedSimpleValue(
                              cast->getOperand(0), call, nested_call,
                              outer_call));
      return CastExprWithOperand(*cast, operand);
    }
    if (auto* load = llvm::dyn_cast<llvm::LoadInst>(value)) {
      return InlineDoubleNestedLoadExpr(*load, call, nested_call, outer_call);
    }
    return Unsupported(function_, "double nested inline value", *value);
  }

  absl::StatusOr<std::string> InlineDoubleNestedPhiValue(
      const llvm::PHINode& phi, const llvm::CallInst& call,
      const llvm::CallInst& nested_call, const llvm::CallInst& outer_call) {
    const llvm::BasicBlock* merge_block = phi.getParent();
    const llvm::Function* function = merge_block->getParent();
    for (const llvm::BasicBlock& block : *function) {
      const auto* branch =
          llvm::dyn_cast<llvm::CondBrInst>(block.getTerminator());
      if (branch == nullptr) continue;
      absl::StatusOr<std::string> true_value =
          InlineDoubleNestedPhiIncomingForPath(
              branch->getSuccessor(0), merge_block, phi, call, nested_call,
              outer_call);
      if (!true_value.ok()) continue;
      absl::StatusOr<std::string> false_value =
          InlineDoubleNestedPhiIncomingForPath(
              branch->getSuccessor(1), merge_block, phi, call, nested_call,
              outer_call);
      if (!false_value.ok()) continue;
      TF_ASSIGN_OR_RETURN(
          std::string condition,
          InlineDoubleNestedSimpleValue(branch->getCondition(), call,
                                        nested_call, outer_call));
      return absl::StrCat("(", condition, " ? ", *true_value, " : ",
                          *false_value, ")");
    }
    return Unsupported(function_, "double nested inline phi", phi);
  }

  absl::StatusOr<std::string> InlineDoubleNestedPhiIncomingForPath(
      const llvm::BasicBlock* block, const llvm::BasicBlock* merge_block,
      const llvm::PHINode& phi, const llvm::CallInst& call,
      const llvm::CallInst& nested_call, const llvm::CallInst& outer_call) {
    if (absl::StatusOr<const llvm::Value*> incoming =
            PhiIncomingValueOnLinearPath(block, merge_block, phi, call);
        incoming.ok()) {
      return InlineDoubleNestedSimpleValue(*incoming, call, nested_call,
                                           outer_call);
    }

    const auto* branch =
        llvm::dyn_cast<llvm::CondBrInst>(block->getTerminator());
    if (branch == nullptr) {
      return Unsupported(function_, "nested inline function shape", call);
    }
    const llvm::BasicBlock* true_successor =
        UnconditionalBranchSuccessor(*branch->getSuccessor(0));
    const llvm::BasicBlock* false_successor =
        UnconditionalBranchSuccessor(*branch->getSuccessor(1));
    if (true_successor == nullptr || true_successor != false_successor) {
      return Unsupported(function_, "nested inline function shape", call);
    }

    const llvm::BasicBlock* local_merge = true_successor;
    TF_ASSIGN_OR_RETURN(const llvm::Value* incoming,
                        PhiIncomingValueOnLinearPath(local_merge, merge_block,
                                                     phi, call));
    const auto* local_phi = llvm::dyn_cast<llvm::PHINode>(incoming);
    if (local_phi == nullptr || local_phi->getParent() != local_merge) {
      return InlineDoubleNestedSimpleValue(incoming, call, nested_call,
                                           outer_call);
    }

    TF_ASSIGN_OR_RETURN(
        std::string condition,
        InlineDoubleNestedSimpleValue(branch->getCondition(), call,
                                      nested_call, outer_call));
    TF_ASSIGN_OR_RETURN(
        std::string true_value,
        InlineDoubleNestedPhiIncomingForPath(
            branch->getSuccessor(0), local_merge, *local_phi, call,
            nested_call, outer_call));
    TF_ASSIGN_OR_RETURN(
        std::string false_value,
        InlineDoubleNestedPhiIncomingForPath(
            branch->getSuccessor(1), local_merge, *local_phi, call,
            nested_call, outer_call));
    return absl::StrCat("(", condition, " ? ", true_value, " : ",
                        false_value, ")");
  }

  absl::StatusOr<std::string> InlineDoubleNestedLoadExpr(
      const llvm::LoadInst& load, const llvm::CallInst& call,
      const llvm::CallInst& nested_call, const llvm::CallInst& outer_call) {
    TF_ASSIGN_OR_RETURN(
        PtrExpr pointer,
        InlineDoubleNestedPointerExprFor(load.getPointerOperand(), call,
                                         nested_call, outer_call));
    return LoadFromPointer(pointer, load.getType());
  }

  absl::StatusOr<PtrExpr> InlineDoubleNestedPointerExprFor(
      const llvm::Value* value, const llvm::CallInst& call,
      const llvm::CallInst& nested_call, const llvm::CallInst& outer_call) {
    if (auto* argument = llvm::dyn_cast<llvm::Argument>(value)) {
      return InlineNestedPointerExprFor(call.getArgOperand(argument->getArgNo()),
                                        nested_call, outer_call);
    }
    if (llvm::isa<llvm::GlobalVariable>(value) ||
        llvm::isa<llvm::AllocaInst>(value)) {
      return PointerExprFor(value);
    }
    if (auto* op = llvm::dyn_cast<llvm::Operator>(value)) {
      switch (op->getOpcode()) {
        case llvm::Instruction::BitCast:
        case llvm::Instruction::AddrSpaceCast:
          return InlineDoubleNestedPointerExprFor(op->getOperand(0), call,
                                                  nested_call, outer_call);
        case llvm::Instruction::GetElementPtr:
          return InlineDoubleNestedGepExpr(*llvm::cast<llvm::GEPOperator>(op),
                                           call, nested_call, outer_call);
        default:
          break;
      }
    }
    return InlineNestedPointerExprFor(value, nested_call, outer_call);
  }

  absl::StatusOr<PtrExpr> InlineDoubleNestedGepExpr(
      const llvm::GEPOperator& gep, const llvm::CallInst& call,
      const llvm::CallInst& nested_call, const llvm::CallInst& outer_call) {
    TF_ASSIGN_OR_RETURN(PtrExpr base,
                        InlineDoubleNestedPointerExprFor(
                            gep.getPointerOperand(), call, nested_call,
                            outer_call));
    llvm::Type* current_type = gep.getSourceElementType();
    std::string offset = "0";
    bool first_index = true;

    for (const llvm::Use& use : gep.indices()) {
      const llvm::Value* index = use.get();

      if (!IsZeroConstant(index)) {
        if (auto* struct_type = llvm::dyn_cast<llvm::StructType>(current_type);
            struct_type != nullptr && !first_index) {
          auto* constant_index = llvm::dyn_cast<llvm::ConstantInt>(index);
          if (constant_index == nullptr) {
            return absl::UnimplementedError(
                "Metal MSL emission requires constant struct GEP indices.");
          }
          uint64_t field = constant_index->getZExtValue();
          int64_t field_offset = 0;
          for (uint64_t i = 0; i < field; ++i) {
            TF_ASSIGN_OR_RETURN(
                int64_t count,
                FlattenedElementCount(data_layout_,
                                      struct_type->getElementType(i),
                                      base.element_type));
            field_offset += count;
          }
          offset = AddIndex(offset, absl::StrCat(field_offset));
        } else {
          TF_ASSIGN_OR_RETURN(llvm::Type* scaled_type,
                              TypeScaledByGepIndex(current_type, first_index));
          TF_ASSIGN_OR_RETURN(int64_t scale,
                              ScaleForGepIndex(scaled_type, &base));
          TF_ASSIGN_OR_RETURN(std::string index_expr,
                              InlineDoubleNestedSimpleValue(index, call,
                                                            nested_call,
                                                            outer_call));
          offset = AddIndex(offset, ScaleIndex(index_expr, scale));
        }
      }

      TF_ASSIGN_OR_RETURN(current_type,
                          TypeAfterGepIndex(current_type, index, first_index));
      first_index = false;
    }

    return PtrExpr{base.base, AddIndex(base.index, offset), base.element_type,
                   base.address_space};
  }

  absl::StatusOr<std::string> InlineNestedLoadExpr(
      const llvm::LoadInst& load, const llvm::CallInst& nested_call,
      const llvm::CallInst& outer_call) {
    TF_ASSIGN_OR_RETURN(PtrExpr pointer,
                        InlineNestedPointerExprFor(load.getPointerOperand(),
                                                   nested_call, outer_call));
    return LoadFromPointer(pointer, load.getType());
  }

  absl::StatusOr<PtrExpr> InlineNestedPointerExprFor(
      const llvm::Value* value, const llvm::CallInst& nested_call,
      const llvm::CallInst& outer_call) {
    if (auto* argument = llvm::dyn_cast<llvm::Argument>(value)) {
      return InlinePointerExprFor(nested_call.getArgOperand(argument->getArgNo()),
                                  outer_call);
    }
    if (llvm::isa<llvm::GlobalVariable>(value) ||
        llvm::isa<llvm::AllocaInst>(value)) {
      return PointerExprFor(value);
    }
    if (auto* op = llvm::dyn_cast<llvm::Operator>(value)) {
      switch (op->getOpcode()) {
        case llvm::Instruction::BitCast:
        case llvm::Instruction::AddrSpaceCast:
          return InlineNestedPointerExprFor(op->getOperand(0), nested_call,
                                            outer_call);
        case llvm::Instruction::GetElementPtr:
          return InlineNestedGepExpr(*llvm::cast<llvm::GEPOperator>(op),
                                     nested_call, outer_call);
        default:
          break;
      }
    }
    return InlinePointerExprFor(value, outer_call);
  }

  absl::StatusOr<PtrExpr> InlineNestedGepExpr(
      const llvm::GEPOperator& gep, const llvm::CallInst& nested_call,
      const llvm::CallInst& outer_call) {
    TF_ASSIGN_OR_RETURN(
        PtrExpr base,
        InlineNestedPointerExprFor(gep.getPointerOperand(), nested_call,
                                   outer_call));
    llvm::Type* current_type = gep.getSourceElementType();
    std::string offset = "0";
    bool first_index = true;

    for (const llvm::Use& use : gep.indices()) {
      const llvm::Value* index = use.get();

      if (!IsZeroConstant(index)) {
        if (auto* struct_type = llvm::dyn_cast<llvm::StructType>(current_type);
            struct_type != nullptr && !first_index) {
          auto* constant_index = llvm::dyn_cast<llvm::ConstantInt>(index);
          if (constant_index == nullptr) {
            return absl::UnimplementedError(
                "Metal MSL emission requires constant struct GEP indices.");
          }
          uint64_t field = constant_index->getZExtValue();
          int64_t field_offset = 0;
          for (uint64_t i = 0; i < field; ++i) {
            TF_ASSIGN_OR_RETURN(
                int64_t count,
                FlattenedElementCount(data_layout_,
                                      struct_type->getElementType(i),
                                      base.element_type));
            field_offset += count;
          }
          offset = AddIndex(offset, absl::StrCat(field_offset));
        } else {
          TF_ASSIGN_OR_RETURN(llvm::Type* scaled_type,
                              TypeScaledByGepIndex(current_type, first_index));
          TF_ASSIGN_OR_RETURN(int64_t scale,
                              ScaleForGepIndex(scaled_type, &base));
          TF_ASSIGN_OR_RETURN(std::string index_expr,
                              InlineNestedSimpleValue(index, nested_call,
                                                      outer_call));
          offset = AddIndex(offset, ScaleIndex(index_expr, scale));
        }
      }

      TF_ASSIGN_OR_RETURN(current_type,
                          TypeAfterGepIndex(current_type, index, first_index));
      first_index = false;
    }

    return PtrExpr{base.base, AddIndex(base.index, offset), base.element_type,
                   base.address_space};
  }

  absl::StatusOr<std::string> InlineSimpleBinaryCall(
      absl::string_view msl_name, const llvm::CallInst& nested_call,
      const llvm::CallInst& outer_call) {
    if (nested_call.arg_size() != 2) {
      return Unsupported(function_, "inline binary call", nested_call);
    }
    TF_ASSIGN_OR_RETURN(
        std::string lhs,
        InlineSimpleValue(nested_call.getArgOperand(0), outer_call));
    TF_ASSIGN_OR_RETURN(
        std::string rhs,
        InlineSimpleValue(nested_call.getArgOperand(1), outer_call));
    return absl::StrCat(msl_name, "(", lhs, ", ", rhs, ")");
  }

  absl::StatusOr<std::string> InlineSimpleUnaryCall(
      absl::string_view msl_name, const llvm::CallInst& nested_call,
      const llvm::CallInst& outer_call) {
    if (nested_call.arg_size() != 1) {
      return Unsupported(function_, "inline unary call", nested_call);
    }
    TF_ASSIGN_OR_RETURN(
        std::string operand,
        InlineSimpleValue(nested_call.getArgOperand(0), outer_call));
    return absl::StrCat(msl_name, "(", operand, ")");
  }

  absl::StatusOr<std::string> InlineSimpleUnsignedBinaryCall(
      absl::string_view msl_name, const llvm::CallInst& nested_call,
      const llvm::CallInst& outer_call) {
    if (nested_call.arg_size() != 2) {
      return Unsupported(function_, "inline binary call", nested_call);
    }
    TF_ASSIGN_OR_RETURN(
        std::string lhs,
        InlineSimpleValue(nested_call.getArgOperand(0), outer_call));
    TF_ASSIGN_OR_RETURN(
        std::string rhs,
        InlineSimpleValue(nested_call.getArgOperand(1), outer_call));
    return UnsignedIntegerBinaryExpr(msl_name, nested_call.getType(), lhs, rhs);
  }

  absl::StatusOr<std::string> UnaryMathCall(absl::string_view msl_name,
                                            const llvm::CallInst& call) {
    if (call.arg_size() != 1) return Unsupported(function_, "math call", call);
    TF_ASSIGN_OR_RETURN(std::string operand, Expr(call.getArgOperand(0)));
    return absl::StrCat(msl_name, "(", operand, ")");
  }

  absl::StatusOr<std::string> IntegerBitCountCall(
      absl::string_view msl_name, const llvm::CallInst& call) {
    if (call.arg_size() < 1) {
      return Unsupported(function_, "integer bit-count call", call);
    }
    TF_ASSIGN_OR_RETURN(std::string operand, Expr(call.getArgOperand(0)));
    llvm::Type* operand_type = call.getArgOperand(0)->getType();
    TF_ASSIGN_OR_RETURN(std::string unsigned_operand_type,
                        MslType(operand_type, /*unsigned_integer=*/true));
    TF_ASSIGN_OR_RETURN(std::string result_type, MslType(call.getType()));
    return absl::StrCat("static_cast<", result_type, ">(", msl_name,
                        "(static_cast<", unsigned_operand_type, ">(", operand,
                        ")))");
  }

  absl::StatusOr<std::string> Log1pCall(const llvm::CallInst& call) {
    if (call.arg_size() != 1) return Unsupported(function_, "math call", call);
    TF_ASSIGN_OR_RETURN(std::string operand, Expr(call.getArgOperand(0)));
    return absl::StrCat("log((1.0f + ", operand, "))");
  }

  absl::StatusOr<std::string> BinaryMathCall(absl::string_view msl_name,
                                             const llvm::CallInst& call) {
    if (call.arg_size() != 2) return Unsupported(function_, "math call", call);
    TF_ASSIGN_OR_RETURN(std::string lhs, Expr(call.getArgOperand(0)));
    TF_ASSIGN_OR_RETURN(std::string rhs, Expr(call.getArgOperand(1)));
    return absl::StrCat(msl_name, "(", lhs, ", ", rhs, ")");
  }

  absl::StatusOr<std::string> UnsignedIntegerBinaryCall(
      absl::string_view msl_name, const llvm::CallInst& call) {
    if (call.arg_size() != 2) return Unsupported(function_, "math call", call);
    TF_ASSIGN_OR_RETURN(std::string lhs, Expr(call.getArgOperand(0)));
    TF_ASSIGN_OR_RETURN(std::string rhs, Expr(call.getArgOperand(1)));
    return UnsignedIntegerBinaryExpr(msl_name, call.getType(), lhs, rhs);
  }

  absl::StatusOr<std::string> UnsignedIntegerBinaryExpr(
      absl::string_view msl_name, llvm::Type* result_type,
      absl::string_view lhs, absl::string_view rhs) {
    TF_ASSIGN_OR_RETURN(std::string unsigned_type,
                        MslType(result_type, /*unsigned_integer=*/true));
    TF_ASSIGN_OR_RETURN(std::string signed_type, MslType(result_type));
    return absl::StrCat("static_cast<", signed_type, ">(", msl_name,
                        "(static_cast<", unsigned_type, ">(", lhs,
                        "), static_cast<", unsigned_type, ">(", rhs, ")))");
  }

  absl::StatusOr<std::string> TernaryMathCall(absl::string_view msl_name,
                                              const llvm::CallInst& call) {
    if (call.arg_size() != 3) return Unsupported(function_, "math call", call);
    TF_ASSIGN_OR_RETURN(std::string arg0, Expr(call.getArgOperand(0)));
    TF_ASSIGN_OR_RETURN(std::string arg1, Expr(call.getArgOperand(1)));
    TF_ASSIGN_OR_RETURN(std::string arg2, Expr(call.getArgOperand(2)));
    return absl::StrCat(msl_name, "(", arg0, ", ", arg1, ", ", arg2, ")");
  }

  absl::StatusOr<std::string> AtomicRmwStatement(
      const llvm::AtomicRMWInst& atomic) {
    TF_ASSIGN_OR_RETURN(PtrExpr pointer,
                        PointerExprFor(atomic.getPointerOperand()));
    TF_ASSIGN_OR_RETURN(std::string value, Expr(atomic.getValOperand()));
    TF_ASSIGN_OR_RETURN(std::string expression,
                        AtomicRmwExpr(atomic, pointer, value));
    if (atomic.use_empty()) return absl::StrCat(expression, ";");
    return absl::StrCat(Name(&atomic), " = ", expression, ";");
  }

  absl::StatusOr<std::string> AtomicRmwExpr(const llvm::AtomicRMWInst& atomic,
                                            const PtrExpr& pointer,
                                            absl::string_view value) {
    TF_ASSIGN_OR_RETURN(std::string atomic_type,
                        MslAtomicType(atomic.getValOperand()->getType(),
                                      AtomicOpUsesUnsignedType(atomic)));
    TF_ASSIGN_OR_RETURN(std::string function, AtomicRmwFunction(atomic));
    return absl::StrCat(function, "(reinterpret_cast<device ", atomic_type,
                        "*>(&", pointer.base, "[", pointer.index, "]), ",
                        value, ", memory_order_relaxed)");
  }

  absl::Status EmitAtomicCmpXchg(const llvm::AtomicCmpXchgInst& cmpxchg,
                                 std::string* output, int indent) {
    TF_ASSIGN_OR_RETURN(PtrExpr pointer,
                        PointerExprFor(cmpxchg.getPointerOperand()));
    TF_ASSIGN_OR_RETURN(std::string compare, Expr(cmpxchg.getCompareOperand()));
    TF_ASSIGN_OR_RETURN(std::string new_value, Expr(cmpxchg.getNewValOperand()));
    llvm::Type* value_type = cmpxchg.getCompareOperand()->getType();
    TF_ASSIGN_OR_RETURN(std::string value_msl_type, MslType(value_type));
    TF_ASSIGN_OR_RETURN(std::string atomic_type,
                        MslAtomicType(value_type, /*unsigned_integer=*/false));
    std::string old_name = CmpXchgOldName(cmpxchg);
    std::string success_name = CmpXchgSuccessName(cmpxchg);

    absl::StrAppend(output, Indent(indent), value_msl_type, " ", old_name,
                    " = ", compare, ";\n");
    absl::StrAppend(
        output, Indent(indent), "bool ", success_name,
        " = atomic_compare_exchange_weak_explicit(reinterpret_cast<device ",
        atomic_type, "*>(&", pointer.base, "[", pointer.index, "]), &",
        old_name, ", ", new_value,
        ", memory_order_relaxed, memory_order_relaxed);\n");
    return absl::OkStatus();
  }

  std::string CmpXchgOldName(const llvm::AtomicCmpXchgInst& cmpxchg) const {
    auto it = cmpxchg_old_names_.find(&cmpxchg);
    if (it != cmpxchg_old_names_.end()) return it->second;
    return absl::StrCat(Name(&cmpxchg), "_old");
  }

  std::string CmpXchgSuccessName(
      const llvm::AtomicCmpXchgInst& cmpxchg) const {
    auto it = cmpxchg_success_names_.find(&cmpxchg);
    if (it != cmpxchg_success_names_.end()) return it->second;
    return absl::StrCat(Name(&cmpxchg), "_success");
  }

  bool AtomicOpUsesUnsignedType(const llvm::AtomicRMWInst& atomic) const {
    switch (atomic.getOperation()) {
      case llvm::AtomicRMWInst::UMax:
      case llvm::AtomicRMWInst::UMin:
        return true;
      default:
        return false;
    }
  }

  absl::StatusOr<std::string> MslAtomicType(llvm::Type* type,
                                            bool unsigned_integer) {
    if (type->isFloatTy()) return "atomic_float";
    if (type->isIntegerTy(32)) {
      return unsigned_integer ? "atomic_uint" : "atomic_int";
    }
    return absl::UnimplementedError(
        absl::StrCat("Unsupported LLVM atomic type for MSL: ",
                     PrintLlvmType(*type)));
  }

  absl::StatusOr<std::string> AtomicRmwFunction(
      const llvm::AtomicRMWInst& atomic) {
    switch (atomic.getOperation()) {
      case llvm::AtomicRMWInst::Add:
      case llvm::AtomicRMWInst::FAdd:
        return "atomic_fetch_add_explicit";
      case llvm::AtomicRMWInst::Sub:
        return "atomic_fetch_sub_explicit";
      case llvm::AtomicRMWInst::And:
        return "atomic_fetch_and_explicit";
      case llvm::AtomicRMWInst::Or:
        return "atomic_fetch_or_explicit";
      case llvm::AtomicRMWInst::Xor:
        return "atomic_fetch_xor_explicit";
      case llvm::AtomicRMWInst::Max:
      case llvm::AtomicRMWInst::UMax:
        return "atomic_fetch_max_explicit";
      case llvm::AtomicRMWInst::Min:
      case llvm::AtomicRMWInst::UMin:
        return "atomic_fetch_min_explicit";
      case llvm::AtomicRMWInst::Xchg:
        return "atomic_exchange_explicit";
      default:
        return Unsupported(function_, "atomic read-modify-write", atomic);
    }
  }

  absl::StatusOr<std::string> VoidCallStatementOrInline(
      const llvm::CallInst& call) {
    const llvm::Function* callee = call.getCalledFunction();
    if (callee != nullptr && !callee->isDeclaration()) {
      return InlineVoidFunctionCall(call);
    }
    return VoidCallStatement(call);
  }

  absl::StatusOr<std::string> VoidCallStatement(const llvm::CallInst& call) {
    const llvm::Function* callee = call.getCalledFunction();
    if (callee == nullptr) return Unsupported(function_, "indirect call", call);
    llvm::StringRef name = callee->getName();
    if (name == "llvm.nvvm.barrier.cta.sync.aligned.all" ||
        name == "llvm.cuda.syncthreads") {
      return "threadgroup_barrier(mem_flags::mem_threadgroup);";
    }
    if (name == "llvm.assume") return "";
    return Unsupported(function_, "void call", call);
  }

  absl::StatusOr<std::string> InlineVoidFunctionCall(
      const llvm::CallInst& call) {
    const llvm::Function* callee = call.getCalledFunction();
    if (callee == nullptr || callee->isDeclaration()) {
      return Unsupported(function_, "inline void call", call);
    }
    if (callee->arg_size() != call.arg_size() || callee->size() != 1) {
      return Unsupported(function_, "inline void function shape", call);
    }

    absl::flat_hash_map<const llvm::Value*, StoredInlineValue> values;
    absl::flat_hash_map<const llvm::Value*, StoredInlineValue> memory;
    std::optional<std::string> output_store;

    const llvm::BasicBlock& block = callee->getEntryBlock();
    for (const llvm::Instruction& instruction : block) {
      if (llvm::isa<llvm::ReturnInst>(&instruction)) break;

      if (auto* alloca = llvm::dyn_cast<llvm::AllocaInst>(&instruction)) {
        TF_ASSIGN_OR_RETURN(std::string zero,
                            ZeroInitializer(alloca->getAllocatedType()));
        memory.insert_or_assign(
            alloca, StoredInlineValue{zero, alloca->getAllocatedType()});
        continue;
      }

      if (auto* store = llvm::dyn_cast<llvm::StoreInst>(&instruction)) {
        TF_ASSIGN_OR_RETURN(
            std::string value,
            InlineValueExpr(store->getValueOperand(), call, values, memory));
        const llvm::Value* root = InlinePointerRoot(store->getPointerOperand());
        if (root == nullptr) {
          return Unsupported(function_, "inline store pointer", *store);
        }
        StoredInlineValue stored{value, store->getValueOperand()->getType()};
        if (llvm::isa<llvm::AllocaInst>(root)) {
          memory.insert_or_assign(root, std::move(stored));
          continue;
        }
        if (auto* argument = llvm::dyn_cast<llvm::Argument>(root)) {
          TF_ASSIGN_OR_RETURN(PtrExpr pointer,
                              PointerExprFor(call.getArgOperand(
                                  argument->getArgNo())));
          TF_ASSIGN_OR_RETURN(
              std::string converted,
              ConvertStoredValue(stored.expression, stored.type,
                                 pointer.element_type));
          output_store =
              absl::StrCat(pointer.base, "[", pointer.index, "] = ",
                           converted, ";");
          continue;
        }
        return Unsupported(function_, "inline store destination", *store);
      }

      if (auto* load = llvm::dyn_cast<llvm::LoadInst>(&instruction)) {
        TF_ASSIGN_OR_RETURN(std::string expression,
                            InlineLoadExpr(*load, call, values, memory));
        values.insert_or_assign(
            load, StoredInlineValue{std::move(expression), load->getType()});
        continue;
      }

      if (instruction.getType()->isVoidTy()) {
        return Unsupported(function_, "inline void instruction", instruction);
      }

      TF_ASSIGN_OR_RETURN(
          std::string expression,
          InlineInstructionExpr(instruction, call, values, memory));
      values.insert_or_assign(
          &instruction,
          StoredInlineValue{std::move(expression), instruction.getType()});
    }

    if (!output_store.has_value()) {
      return Unsupported(function_, "inline void function without output store",
                         call);
    }
    return *output_store;
  }

  const llvm::Value* InlinePointerRoot(const llvm::Value* value) const {
    while (value != nullptr) {
      if (llvm::isa<llvm::Argument>(value) ||
          llvm::isa<llvm::AllocaInst>(value) ||
          llvm::isa<llvm::GlobalVariable>(value)) {
        return value;
      }
      if (auto* op = llvm::dyn_cast<llvm::Operator>(value)) {
        switch (op->getOpcode()) {
          case llvm::Instruction::BitCast:
          case llvm::Instruction::AddrSpaceCast:
            value = op->getOperand(0);
            continue;
          default:
            return value;
        }
      }
      return value;
    }
    return nullptr;
  }

  absl::StatusOr<std::string> InlineLoadExpr(
      const llvm::LoadInst& load, const llvm::CallInst& call,
      const absl::flat_hash_map<const llvm::Value*, StoredInlineValue>& values,
      const absl::flat_hash_map<const llvm::Value*, StoredInlineValue>& memory) {
    const llvm::Value* root = InlinePointerRoot(load.getPointerOperand());
    if (root == nullptr) {
      return Unsupported(function_, "inline load pointer", load);
    }

    if (llvm::isa<llvm::AllocaInst>(root)) {
      auto it = memory.find(root);
      if (it == memory.end()) {
        return Unsupported(function_, "inline uninitialized stack load", load);
      }
      return ConvertStoredValue(it->second.expression, it->second.type,
                                load.getType());
    }

    if (auto* argument = llvm::dyn_cast<llvm::Argument>(root)) {
      TF_ASSIGN_OR_RETURN(PtrExpr pointer,
                          PointerExprFor(call.getArgOperand(
                              argument->getArgNo())));
      return LoadFromPointer(pointer, load.getType());
    }

    if (auto* global = llvm::dyn_cast<llvm::GlobalVariable>(root)) {
      return InlineConstantGlobalLoadExpr(*global, load.getType());
    }

    TF_ASSIGN_OR_RETURN(PtrExpr pointer,
                        InlinePointerExprFor(load.getPointerOperand(), call));
    return LoadFromPointer(pointer, load.getType());
  }

  absl::StatusOr<PtrExpr> InlinePointerExprFor(
      const llvm::Value* value, const llvm::CallInst& call) {
    if (auto* argument = llvm::dyn_cast<llvm::Argument>(value)) {
      return PointerExprFor(call.getArgOperand(argument->getArgNo()));
    }

    if (llvm::isa<llvm::GlobalVariable>(value) ||
        llvm::isa<llvm::AllocaInst>(value)) {
      return PointerExprFor(value);
    }

    if (auto* op = llvm::dyn_cast<llvm::Operator>(value)) {
      switch (op->getOpcode()) {
        case llvm::Instruction::BitCast:
        case llvm::Instruction::AddrSpaceCast:
          return InlinePointerExprFor(op->getOperand(0), call);
        case llvm::Instruction::GetElementPtr:
          return InlineGepExpr(*llvm::cast<llvm::GEPOperator>(op), call);
        default:
          break;
      }
    }

    return Unsupported(function_, "inline pointer expression", *value);
  }

  absl::StatusOr<PtrExpr> InlineGepExpr(const llvm::GEPOperator& gep,
                                        const llvm::CallInst& call) {
    TF_ASSIGN_OR_RETURN(PtrExpr base,
                        InlinePointerExprFor(gep.getPointerOperand(), call));
    llvm::Type* current_type = gep.getSourceElementType();
    std::string offset = "0";
    bool first_index = true;

    for (const llvm::Use& use : gep.indices()) {
      const llvm::Value* index = use.get();

      if (!IsZeroConstant(index)) {
        if (auto* struct_type = llvm::dyn_cast<llvm::StructType>(current_type);
            struct_type != nullptr && !first_index) {
          auto* constant_index = llvm::dyn_cast<llvm::ConstantInt>(index);
          if (constant_index == nullptr) {
            return absl::UnimplementedError(
                "Metal MSL emission requires constant struct GEP indices.");
          }
          uint64_t field = constant_index->getZExtValue();
          int64_t field_offset = 0;
          for (uint64_t i = 0; i < field; ++i) {
            TF_ASSIGN_OR_RETURN(
                int64_t count,
                FlattenedElementCount(data_layout_,
                                      struct_type->getElementType(i),
                                      base.element_type));
            field_offset += count;
          }
          offset = AddIndex(offset, absl::StrCat(field_offset));
        } else {
          TF_ASSIGN_OR_RETURN(llvm::Type* scaled_type,
                              TypeScaledByGepIndex(current_type, first_index));
          TF_ASSIGN_OR_RETURN(int64_t scale,
                              ScaleForGepIndex(scaled_type, &base));
          TF_ASSIGN_OR_RETURN(std::string index_expr,
                              InlineSimpleValue(index, call));
          offset = AddIndex(offset, ScaleIndex(index_expr, scale));
        }
      }

      TF_ASSIGN_OR_RETURN(current_type,
                          TypeAfterGepIndex(current_type, index, first_index));
      first_index = false;
    }

    return PtrExpr{base.base, AddIndex(base.index, offset), base.element_type,
                   base.address_space};
  }

  absl::StatusOr<std::string> InlineInstructionExpr(
      const llvm::Instruction& instruction, const llvm::CallInst& call,
      const absl::flat_hash_map<const llvm::Value*, StoredInlineValue>& values,
      const absl::flat_hash_map<const llvm::Value*, StoredInlineValue>& memory) {
    if (auto* binary = llvm::dyn_cast<llvm::BinaryOperator>(&instruction)) {
      TF_ASSIGN_OR_RETURN(
          std::string lhs,
          InlineValueExpr(binary->getOperand(0), call, values, memory));
      TF_ASSIGN_OR_RETURN(
          std::string rhs,
          InlineValueExpr(binary->getOperand(1), call, values, memory));
      return BinaryExprWithOperands(*binary, lhs, rhs);
    }
    if (instruction.getOpcode() == llvm::Instruction::FNeg) {
      TF_ASSIGN_OR_RETURN(
          std::string operand,
          InlineValueExpr(instruction.getOperand(0), call, values, memory));
      return absl::StrCat("(-", operand, ")");
    }
    if (auto* icmp = llvm::dyn_cast<llvm::ICmpInst>(&instruction)) {
      bool unsigned_compare = icmp->isUnsigned();
      TF_ASSIGN_OR_RETURN(
          std::string lhs,
          InlineValueExpr(icmp->getOperand(0), call, values, memory));
      TF_ASSIGN_OR_RETURN(
          std::string rhs,
          InlineValueExpr(icmp->getOperand(1), call, values, memory));
      TF_ASSIGN_OR_RETURN(lhs,
                          CastForUnsignedCompare(icmp->getOperand(0)->getType(),
                                                 lhs, unsigned_compare));
      TF_ASSIGN_OR_RETURN(rhs,
                          CastForUnsignedCompare(icmp->getOperand(1)->getType(),
                                                 rhs, unsigned_compare));
      return ICmpExprWithOperands(*icmp, lhs, rhs);
    }
    if (auto* fcmp = llvm::dyn_cast<llvm::FCmpInst>(&instruction)) {
      TF_ASSIGN_OR_RETURN(
          std::string lhs,
          InlineValueExpr(fcmp->getOperand(0), call, values, memory));
      TF_ASSIGN_OR_RETURN(
          std::string rhs,
          InlineValueExpr(fcmp->getOperand(1), call, values, memory));
      return FCmpExprWithOperands(*fcmp, lhs, rhs);
    }
    if (auto* select = llvm::dyn_cast<llvm::SelectInst>(&instruction)) {
      TF_ASSIGN_OR_RETURN(
          std::string condition,
          InlineValueExpr(select->getCondition(), call, values, memory));
      TF_ASSIGN_OR_RETURN(
          std::string true_value,
          InlineValueExpr(select->getTrueValue(), call, values, memory));
      TF_ASSIGN_OR_RETURN(
          std::string false_value,
          InlineValueExpr(select->getFalseValue(), call, values, memory));
      return absl::StrCat("(", condition, " ? ", true_value, " : ",
                          false_value, ")");
    }
    if (auto* insert = llvm::dyn_cast<llvm::InsertValueInst>(&instruction)) {
      return InlineComplexInsertValueExpr(*insert, call, values, memory);
    }
    if (auto* extract = llvm::dyn_cast<llvm::ExtractValueInst>(&instruction)) {
      return InlineExtractValueExpr(*extract, call, values, memory);
    }
    if (auto* cast = llvm::dyn_cast<llvm::CastInst>(&instruction)) {
      TF_ASSIGN_OR_RETURN(
          std::string operand,
          InlineValueExpr(cast->getOperand(0), call, values, memory));
      return CastExprWithOperand(*cast, operand);
    }
    return Unsupported(function_, "inline instruction", instruction);
  }

  absl::StatusOr<std::string> InlineComplexInsertValueExpr(
      const llvm::InsertValueInst& insert, const llvm::CallInst& call,
      const absl::flat_hash_map<const llvm::Value*, StoredInlineValue>& values,
      const absl::flat_hash_map<const llvm::Value*, StoredInlineValue>& memory) {
    if (!IsComplexFloatStruct(insert.getType()) || insert.getNumIndices() != 1) {
      return Unsupported(function_, "inline aggregate insertion", insert);
    }

    unsigned field = *insert.idx_begin();
    if (field > 1) {
      return absl::InvalidArgumentError(
          absl::StrCat("complex field ", field, " is out of range."));
    }

    std::string components[2] = {"0.0f", "0.0f"};
    const llvm::Value* aggregate = insert.getAggregateOperand();
    if (!llvm::isa<llvm::UndefValue>(aggregate) &&
        !llvm::isa<llvm::PoisonValue>(aggregate)) {
      TF_ASSIGN_OR_RETURN(
          components[0],
          InlineComplexComponentExpr(aggregate, call, values, memory, 0));
      TF_ASSIGN_OR_RETURN(
          components[1],
          InlineComplexComponentExpr(aggregate, call, values, memory, 1));
    }
    TF_ASSIGN_OR_RETURN(
        std::string inserted,
        InlineValueExpr(insert.getInsertedValueOperand(), call, values,
                        memory));
    components[field] = std::move(inserted);
    return absl::StrCat("float2(", components[0], ", ", components[1], ")");
  }

  absl::StatusOr<std::string> InlineComplexComponentExpr(
      const llvm::Value* value, const llvm::CallInst& call,
      const absl::flat_hash_map<const llvm::Value*, StoredInlineValue>& values,
      const absl::flat_hash_map<const llvm::Value*, StoredInlineValue>& memory,
      unsigned field) {
    if (field > 1) {
      return absl::InvalidArgumentError(
          absl::StrCat("complex field ", field, " is out of range."));
    }
    if (llvm::isa<llvm::UndefValue>(value) ||
        llvm::isa<llvm::PoisonValue>(value)) {
      return "0.0f";
    }
    if (auto* insert = llvm::dyn_cast<llvm::InsertValueInst>(value)) {
      if (insert->getNumIndices() != 1) {
        return Unsupported(function_, "inline aggregate insertion", *insert);
      }
      if (*insert->idx_begin() == field) {
        return InlineValueExpr(insert->getInsertedValueOperand(), call, values,
                               memory);
      }
      return InlineComplexComponentExpr(insert->getAggregateOperand(), call,
                                        values, memory, field);
    }
    TF_ASSIGN_OR_RETURN(std::string expression,
                        InlineValueExpr(value, call, values, memory));
    return absl::StrCat(expression, field == 0 ? ".x" : ".y");
  }

  absl::StatusOr<std::string> InlineExtractValueExpr(
      const llvm::ExtractValueInst& extract, const llvm::CallInst& call,
      const absl::flat_hash_map<const llvm::Value*, StoredInlineValue>& values,
      const absl::flat_hash_map<const llvm::Value*, StoredInlineValue>& memory) {
    if (extract.getNumIndices() != 1) {
      return Unsupported(function_, "inline nested aggregate extraction",
                         extract);
    }

    const llvm::Value* aggregate = extract.getAggregateOperand();
    unsigned field = *extract.idx_begin();
    if (IsComplexFloatStruct(aggregate->getType())) {
      TF_ASSIGN_OR_RETURN(std::string value,
                          InlineValueExpr(aggregate, call, values, memory));
      if (field == 0) return absl::StrCat(value, ".x");
      if (field == 1) return absl::StrCat(value, ".y");
      return absl::InvalidArgumentError(
          absl::StrCat("complex field ", field, " is out of range."));
    }
    return Unsupported(function_, "inline aggregate extraction", extract);
  }

  absl::StatusOr<std::string> InlineValueExpr(
      const llvm::Value* value, const llvm::CallInst& call,
      const absl::flat_hash_map<const llvm::Value*, StoredInlineValue>& values,
      const absl::flat_hash_map<const llvm::Value*, StoredInlineValue>& memory) {
    if (auto* argument = llvm::dyn_cast<llvm::Argument>(value)) {
      return Expr(call.getArgOperand(argument->getArgNo()));
    }
    if (llvm::isa<llvm::Constant>(value)) return Expr(value);
    if (auto* instruction = llvm::dyn_cast<llvm::Instruction>(value)) {
      auto it = values.find(instruction);
      if (it != values.end()) return it->second.expression;
    }
    return Unsupported(function_, "inline value expression", *value);
  }

  absl::StatusOr<std::string> ConvertStoredValue(absl::string_view expression,
                                                 llvm::Type* from_type,
                                                 llvm::Type* to_type) {
    if (from_type == to_type) return std::string(expression);
    TF_ASSIGN_OR_RETURN(std::string to_msl_type, MslType(to_type));

    std::optional<int64_t> from_size = FixedTypeSize(data_layout_, from_type);
    std::optional<int64_t> to_size = FixedTypeSize(data_layout_, to_type);
    if (from_size.has_value() && to_size.has_value() &&
        *from_size == *to_size) {
      return absl::StrCat("as_type<", to_msl_type, ">(", expression, ")");
    }
    return absl::StrCat("static_cast<", to_msl_type, ">(", expression, ")");
  }

  absl::StatusOr<std::string> LoadFromPointer(const PtrExpr& pointer,
                                              llvm::Type* load_type) {
    std::string expression =
        absl::StrCat(pointer.base, "[", pointer.index, "]");
    if (pointer.element_type == load_type) return expression;
    TF_ASSIGN_OR_RETURN(std::string load_msl_type, MslType(load_type));
    return absl::StrCat("(*reinterpret_cast<", pointer.address_space, " ",
                        load_msl_type, "*>(&", expression, "))");
  }

  absl::StatusOr<std::string> InlineConstantGlobalLoadExpr(
      const llvm::GlobalVariable& global, llvm::Type* load_type) {
    if (!global.isConstant() || !global.hasInitializer()) {
      return Unsupported(function_, "inline global load", global);
    }

    const llvm::Constant* initializer = global.getInitializer();
    if (llvm::isa<llvm::ConstantAggregateZero>(initializer)) {
      return ZeroInitializer(load_type);
    }

    auto* data = llvm::dyn_cast<llvm::ConstantDataSequential>(initializer);
    if (data == nullptr) {
      return Unsupported(function_, "inline global initializer", global);
    }

    std::optional<int64_t> load_size = FixedTypeSize(data_layout_, load_type);
    if (!load_size.has_value() || *load_size <= 0 ||
        data->getRawDataValues().size() < *load_size) {
      return Unsupported(function_, "inline global load type", global);
    }

    uint64_t bits = 0;
    llvm::StringRef raw = data->getRawDataValues();
    for (int64_t i = 0; i < *load_size; ++i) {
      bits |= static_cast<uint64_t>(static_cast<unsigned char>(raw[i]))
              << (i * 8);
    }

    if (load_type->isFloatTy()) {
      if (bits == 0) return "float(0)";
      return absl::StrCat("as_type<float>(", static_cast<uint32_t>(bits),
                          "u)");
    }
    if (load_type->isBFloatTy()) {
      return absl::StrCat("static_cast<ushort>(", static_cast<uint16_t>(bits),
                          "u)");
    }
    if (load_type->isHalfTy()) {
      return absl::StrCat("as_type<half>(static_cast<ushort>(",
                          static_cast<uint16_t>(bits), "u))");
    }
    if (load_type->isIntegerTy()) {
      return IntegerLiteral(llvm::APInt(load_type->getIntegerBitWidth(), bits),
                            /*is_signed=*/true);
    }

    return Unsupported(function_, "inline global load type", global);
  }

  absl::StatusOr<PtrExpr> PointerExprFor(const llvm::Value* value) {
    if (auto* argument = llvm::dyn_cast<llvm::Argument>(value)) {
      return PtrExpr{Name(argument), "0", PointerElementType(argument),
                     "device"};
    }

    if (auto* global = llvm::dyn_cast<llvm::GlobalVariable>(value)) {
      if (global->getAddressSpace() != 3) {
        return Unsupported(function_, "global pointer expression", *global);
      }
      TF_ASSIGN_OR_RETURN(ArrayInfo array,
                          FlattenedArrayInfo(global->getValueType()));
      return PtrExpr{Name(global), "0", array.element_type, "threadgroup"};
    }

    if (auto* alloca = llvm::dyn_cast<llvm::AllocaInst>(value)) {
      return PtrExpr{Name(alloca), "0", alloca->getAllocatedType(), "thread"};
    }

    if (auto* op = llvm::dyn_cast<llvm::Operator>(value)) {
      switch (op->getOpcode()) {
        case llvm::Instruction::BitCast:
        case llvm::Instruction::AddrSpaceCast:
          return PointerExprFor(op->getOperand(0));
        case llvm::Instruction::GetElementPtr:
          return GepExpr(*llvm::cast<llvm::GEPOperator>(op));
        default:
          break;
      }
    }

    return Unsupported(function_, "pointer expression", *value);
  }

  absl::StatusOr<PtrExpr> GepExpr(const llvm::GEPOperator& gep) {
    TF_ASSIGN_OR_RETURN(PtrExpr base, PointerExprFor(gep.getPointerOperand()));
    llvm::Type* current_type = gep.getSourceElementType();
    std::string offset = "0";
    bool first_index = true;

    for (const llvm::Use& use : gep.indices()) {
      const llvm::Value* index = use.get();

      if (!IsZeroConstant(index)) {
        if (auto* struct_type = llvm::dyn_cast<llvm::StructType>(current_type);
            struct_type != nullptr && !first_index) {
          auto* constant_index = llvm::dyn_cast<llvm::ConstantInt>(index);
          if (constant_index == nullptr) {
            return absl::UnimplementedError(
                "Metal MSL emission requires constant struct GEP indices.");
          }
          uint64_t field = constant_index->getZExtValue();
          int64_t field_offset = 0;
          for (uint64_t i = 0; i < field; ++i) {
            TF_ASSIGN_OR_RETURN(
                int64_t count,
                FlattenedElementCount(data_layout_,
                                      struct_type->getElementType(i),
                                      base.element_type));
            field_offset += count;
          }
          offset = AddIndex(offset, absl::StrCat(field_offset));
        } else {
          TF_ASSIGN_OR_RETURN(llvm::Type* scaled_type,
                              TypeScaledByGepIndex(current_type, first_index));
          TF_ASSIGN_OR_RETURN(int64_t scale,
                              ScaleForGepIndex(scaled_type, &base));
          TF_ASSIGN_OR_RETURN(std::string index_expr, Expr(index));
          offset = AddIndex(offset, ScaleIndex(index_expr, scale));
        }
      }

      TF_ASSIGN_OR_RETURN(current_type,
                          TypeAfterGepIndex(current_type, index, first_index));
      first_index = false;
    }

    return PtrExpr{base.base, AddIndex(base.index, offset), base.element_type,
                   base.address_space};
  }

  const llvm::Function& function_;
  const llvm::DataLayout& data_layout_;
  absl::flat_hash_map<const llvm::Value*, llvm::Type*> pointer_element_types_;
  absl::flat_hash_map<const llvm::Value*, std::string> names_;
  absl::flat_hash_map<const llvm::Value*, std::string> cmpxchg_old_names_;
  absl::flat_hash_map<const llvm::Value*, std::string> cmpxchg_success_names_;
};

}  // namespace

absl::StatusOr<std::string> EmitMslFromLlvmModule(const llvm::Module& module) {
  std::vector<std::string> kernels;
  for (const llvm::Function& function : module.functions()) {
    if (!IsKernelFunction(function)) continue;
    FunctionEmitter emitter(function);
    TF_ASSIGN_OR_RETURN(std::string kernel, emitter.Emit());
    kernels.push_back(std::move(kernel));
  }

  return absl::StrCat("#include <metal_stdlib>\n",
                      "#include <metal_math>\n",
                      "using namespace metal;\n\n",
                      WideVectorStructDefinitions(),
                      "\n",
                      "inline ushort xla_metal_f32_to_bf16(float value) {\n"
                      "  uint bits = as_type<uint>(value);\n"
                      "  uint rounding_bias = 0x7fffu + ((bits >> 16) & 1u);\n"
                      "  return static_cast<ushort>((bits + rounding_bias) >> "
                      "16);\n"
                      "}\n\n"
                      "inline float xla_metal_bf16_to_f32(ushort value) {\n"
                      "  return as_type<float>(static_cast<uint>(value) << "
                      "16);\n"
                      "}\n\n",
                      absl::StrJoin(kernels, "\n"));
}

}  // namespace gpu
}  // namespace xla
