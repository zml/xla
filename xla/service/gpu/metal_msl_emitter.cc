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

bool IsPointerType(const llvm::Value& value) {
  return value.getType()->isPointerTy();
}

bool IsAggregateType(llvm::Type* type) {
  return type->isStructTy() || type->isArrayTy();
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
  if (type->isHalfTy()) return "half";
  if (type->isFloatTy()) return "float";
  if (type->isDoubleTy()) return "double";
  return absl::UnimplementedError(
      absl::StrCat("Unsupported LLVM scalar type for MSL: ",
                   PrintLlvmType(*type)));
}

absl::StatusOr<std::string> MslType(llvm::Type* type,
                                    bool unsigned_integer = false) {
  if (auto* vector_type = llvm::dyn_cast<llvm::FixedVectorType>(type)) {
    int elements = vector_type->getNumElements();
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
  return MslScalarType(type, unsigned_integer);
}

absl::StatusOr<std::string> ZeroInitializer(llvm::Type* type) {
  if (type->isIntegerTy(1)) return "false";
  if (type->isIntegerTy()) return "0";
  if (type->isHalfTy()) return "half(0.0)";
  if (type->isFloatTy()) return "0.0f";
  if (type->isDoubleTy()) return "0.0";
  if (auto* vector_type = llvm::dyn_cast<llvm::FixedVectorType>(type)) {
    TF_ASSIGN_OR_RETURN(std::string msl_type, MslType(type));
    TF_ASSIGN_OR_RETURN(std::string zero,
                        ZeroInitializer(vector_type->getElementType()));
    std::vector<std::string> values(vector_type->getNumElements(), zero);
    return absl::StrCat(msl_type, "(", absl::StrJoin(values, ", "), ")");
  }
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

    std::string output;
    RETURN_IF_ERROR(EmitSignature(&output));
    absl::StrAppend(&output, " {\n");
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
        }
      }
    }
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
    for (const llvm::BasicBlock& block : function_) {
      for (const llvm::Instruction& instruction : block) {
        if (instruction.getType()->isVoidTy()) continue;
        if (IsPointerType(instruction)) continue;
        if (IsAggregateType(instruction.getType())) continue;
        names_[&instruction] = absl::StrCat("v", value_index++);
      }
    }
  }

  std::string Name(const llvm::Value* value) const {
    auto it = names_.find(value);
    if (it != names_.end()) return it->second;
    return MslIdentifier(value->getName().str(), "unnamed");
  }

  absl::Status EmitSignature(std::string* output) {
    std::string kernel_name =
        MslIdentifier(function_.getName().str(), "xla_metal_kernel");
    if (kernel_name != function_.getName().str()) {
      return absl::UnimplementedError(absl::StrCat(
          "Metal kernel name '", function_.getName().str(),
          "' is not a valid MSL identifier after XLA sanitization."));
    }

    absl::StrAppend(output, "kernel void ", kernel_name, "(");
    std::vector<std::string> params;
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
    params.push_back("uint3 metal_tid [[thread_position_in_threadgroup]]");
    params.push_back("uint3 metal_bid [[threadgroup_position_in_grid]]");
    params.push_back("uint3 metal_ntid [[threads_per_threadgroup]]");
    params.push_back("uint3 metal_nbid [[threadgroups_per_grid]]");
    params.push_back("uint3 metal_gid [[thread_position_in_grid]]");
    absl::StrAppend(output, absl::StrJoin(params, ", "), ")");
    return absl::OkStatus();
  }

  llvm::Type* PointerElementType(const llvm::Value* pointer) const {
    const llvm::Value* root = PointerRoot(pointer);
    auto it = pointer_element_types_.find(root);
    if (it != pointer_element_types_.end()) return it->second;
    return llvm::Type::getInt8Ty(function_.getContext());
  }

  absl::Status EmitDeclarations(std::string* output) {
    for (const llvm::BasicBlock& block : function_) {
      for (const llvm::Instruction& instruction : block) {
        if (instruction.getType()->isVoidTy()) continue;
        if (IsPointerType(instruction)) continue;
        if (IsAggregateType(instruction.getType())) continue;
        TF_ASSIGN_OR_RETURN(std::string type, MslType(instruction.getType()));
        TF_ASSIGN_OR_RETURN(std::string zero,
                            ZeroInitializer(instruction.getType()));
        absl::StrAppend(output, "  ", type, " ", Name(&instruction), " = ",
                        zero, ";\n");
      }
    }
    return absl::OkStatus();
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

      return Unsupported(function_, "control flow shape", *terminator);
    }

    return absl::OkStatus();
  }

  bool CanEmitStructuredBlockToStop(const llvm::BasicBlock* block,
                                    const llvm::BasicBlock* stop_block) const {
    absl::flat_hash_set<const llvm::BasicBlock*> visited;
    return CanEmitStructuredBlockToStop(block, stop_block, &visited);
  }

  bool CanEmitStructuredBlockToStop(
      const llvm::BasicBlock* block, const llvm::BasicBlock* stop_block,
      absl::flat_hash_set<const llvm::BasicBlock*>* visited) const {
    if (block == stop_block) return true;
    if (block == nullptr || !visited->insert(block).second) return false;

    const llvm::Instruction* terminator = block->getTerminator();
    if (llvm::isa<llvm::ReturnInst>(terminator)) return false;

    if (const auto* branch = llvm::dyn_cast<llvm::UncondBrInst>(terminator)) {
      return CanEmitStructuredBlockToStop(branch->getSuccessor(0), stop_block,
                                          visited);
    }

    const auto* branch = llvm::dyn_cast<llvm::CondBrInst>(terminator);
    if (branch == nullptr) return false;

    const llvm::BasicBlock* true_block = branch->getSuccessor(0);
    const llvm::BasicBlock* false_block = branch->getSuccessor(1);
    const llvm::BasicBlock* true_successor =
        UnconditionalBranchSuccessor(*true_block);
    const llvm::BasicBlock* false_successor =
        UnconditionalBranchSuccessor(*false_block);

    if (true_successor != nullptr && true_successor == false_successor) {
      absl::flat_hash_set<const llvm::BasicBlock*> true_visited = *visited;
      absl::flat_hash_set<const llvm::BasicBlock*> false_visited = *visited;
      absl::flat_hash_set<const llvm::BasicBlock*> merge_visited = *visited;
      return CanEmitStructuredBlockToStop(true_block, true_successor,
                                          &true_visited) &&
             CanEmitStructuredBlockToStop(false_block, false_successor,
                                          &false_visited) &&
             CanEmitStructuredBlockToStop(true_successor, stop_block,
                                          &merge_visited);
    }

    absl::flat_hash_set<const llvm::BasicBlock*> true_visited = *visited;
    absl::flat_hash_set<const llvm::BasicBlock*> false_visited = *visited;
    if (CanEmitStructuredBlockToStop(true_block, false_block, &true_visited) &&
        CanEmitStructuredBlockToStop(false_block, stop_block, &false_visited)) {
      return true;
    }

    true_visited = *visited;
    false_visited = *visited;
    return CanEmitStructuredBlockToStop(false_block, true_block,
                                        &false_visited) &&
           CanEmitStructuredBlockToStop(true_block, stop_block, &true_visited);
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
        absl::StrAppend(output, Indent(indent), "*reinterpret_cast<device ",
                        value_msl_type, "*>(&", pointer.base, "[",
                        pointer.index, "]) = ", value, ";\n");
      }
      return absl::OkStatus();
    }

    if (auto* call = llvm::dyn_cast<llvm::CallInst>(&instruction)) {
      if (call->getType()->isVoidTy()) {
        TF_ASSIGN_OR_RETURN(std::string statement, VoidCallStatement(*call));
        absl::StrAppend(output, Indent(indent), statement, "\n");
        return absl::OkStatus();
      }
    }

    if (IsAggregateType(instruction.getType())) return absl::OkStatus();

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
        if (value->getType()->isHalfTy()) {
          return absl::StrCat("half(", sign, literal, ")");
        }
        if (value->getType()->isFloatTy()) {
          return absl::StrCat("float(", sign, literal, ")");
        }
        return absl::StrCat("double(", sign, literal, ")");
      }
      double as_double = as_apfloat.convertToDouble();
      if (value->getType()->isHalfTy()) {
        return absl::StrFormat("half(%.9g)", as_double);
      }
      if (value->getType()->isFloatTy()) {
        return absl::StrFormat("float(%.9g)", as_double);
      }
      return absl::StrFormat("%.17g", as_double);
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
      if (load->getType() != pointer.element_type) {
        TF_ASSIGN_OR_RETURN(std::string load_msl_type,
                            MslType(load->getType()));
        return absl::StrCat("(*reinterpret_cast<device ", load_msl_type,
                            "*>(&", pointer.base, "[", pointer.index, "]))");
      }
      return absl::StrCat(pointer.base, "[", pointer.index, "]");
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
    if (auto* extract = llvm::dyn_cast<llvm::ExtractValueInst>(&instruction)) {
      return ExtractValueExpr(*extract);
    }
    if (auto* extract =
            llvm::dyn_cast<llvm::ExtractElementInst>(&instruction)) {
      TF_ASSIGN_OR_RETURN(std::string vector, Expr(extract->getVectorOperand()));
      TF_ASSIGN_OR_RETURN(std::string index, Expr(extract->getIndexOperand()));
      return absl::StrCat(vector, "[", index, "]");
    }
    return Unsupported(function_, "instruction", instruction);
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
    if (auto* call = llvm::dyn_cast<llvm::CallInst>(aggregate)) {
      return InlineSimpleAggregateElement(*call, field);
    }
    return AggregateElementValue(aggregate, /*outer_call=*/nullptr, field);
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
      case llvm::Instruction::LShr:
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

  absl::StatusOr<std::string> FCmpExpr(const llvm::FCmpInst& fcmp) {
    TF_ASSIGN_OR_RETURN(std::string lhs, Expr(fcmp.getOperand(0)));
    TF_ASSIGN_OR_RETURN(std::string rhs, Expr(fcmp.getOperand(1)));
    return FCmpExprWithOperands(fcmp, lhs, rhs);
  }

  absl::StatusOr<std::string> FCmpExprWithOperands(
      const llvm::FCmpInst& fcmp, absl::string_view lhs,
      absl::string_view rhs) {
    switch (fcmp.getPredicate()) {
      case llvm::CmpInst::FCMP_OEQ:
      case llvm::CmpInst::FCMP_UEQ:
        return absl::StrCat("(", lhs, " == ", rhs, ")");
      case llvm::CmpInst::FCMP_ONE:
      case llvm::CmpInst::FCMP_UNE:
        return absl::StrCat("(", lhs, " != ", rhs, ")");
      case llvm::CmpInst::FCMP_OGT:
      case llvm::CmpInst::FCMP_UGT:
        return absl::StrCat("(", lhs, " > ", rhs, ")");
      case llvm::CmpInst::FCMP_OGE:
      case llvm::CmpInst::FCMP_UGE:
        return absl::StrCat("(", lhs, " >= ", rhs, ")");
      case llvm::CmpInst::FCMP_OLT:
      case llvm::CmpInst::FCMP_ULT:
        return absl::StrCat("(", lhs, " < ", rhs, ")");
      case llvm::CmpInst::FCMP_OLE:
      case llvm::CmpInst::FCMP_ULE:
        return absl::StrCat("(", lhs, " <= ", rhs, ")");
      case llvm::CmpInst::FCMP_FALSE:
        return "false";
      case llvm::CmpInst::FCMP_TRUE:
        return "true";
      default:
        return Unsupported(function_, "floating-point comparison", fcmp);
    }
  }

  absl::StatusOr<std::string> CastExpr(const llvm::CastInst& cast) {
    TF_ASSIGN_OR_RETURN(std::string operand, Expr(cast.getOperand(0)));
    TF_ASSIGN_OR_RETURN(std::string type, MslType(cast.getType()));
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
    if (name.starts_with("llvm.maximum.")) return BinaryMathCall("max", call);
    if (name.starts_with("llvm.minimum.")) return BinaryMathCall("min", call);
    if (name.starts_with("llvm.smax.")) return BinaryMathCall("max", call);
    if (name.starts_with("llvm.smin.")) return BinaryMathCall("min", call);

    if (name == "__nv_fabsf") return UnaryMathCall("fabs", call);
    if (name == "__nv_sqrtf") return UnaryMathCall("sqrt", call);
    if (name == "__nv_sinf") return UnaryMathCall("sin", call);
    if (name == "__nv_cosf") return UnaryMathCall("cos", call);
    if (name == "__nv_expf") return UnaryMathCall("exp", call);
    if (name == "__nv_logf") return UnaryMathCall("log", call);
    if (name == "__nv_fmaf") return TernaryMathCall("fma", call);

    return Unsupported(function_, "call", call);
  }

  absl::StatusOr<std::string> InlineSimpleFunctionCall(
      const llvm::CallInst& call) {
    const llvm::Function* callee = call.getCalledFunction();
    if (callee == nullptr || callee->isDeclaration()) {
      return Unsupported(function_, "inline call", call);
    }
    if (callee->arg_size() != call.arg_size() || callee->size() != 1) {
      return Unsupported(function_, "inline function shape", call);
    }
    const llvm::BasicBlock& block = callee->getEntryBlock();
    const auto* ret = llvm::dyn_cast<llvm::ReturnInst>(block.getTerminator());
    if (ret == nullptr || ret->getReturnValue() == nullptr) {
      return Unsupported(function_, "inline function return", call);
    }
    return InlineSimpleValue(ret->getReturnValue(), call);
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
    if (auto* cast = llvm::dyn_cast<llvm::CastInst>(value)) {
      TF_ASSIGN_OR_RETURN(std::string operand,
                          InlineSimpleValue(cast->getOperand(0), call));
      TF_ASSIGN_OR_RETURN(std::string type, MslType(cast->getType()));
      if (cast->getOpcode() == llvm::Instruction::BitCast) {
        return absl::StrCat("as_type<", type, ">(", operand, ")");
      }
      return absl::StrCat("static_cast<", type, ">(", operand, ")");
    }
    if (auto* nested_call = llvm::dyn_cast<llvm::CallInst>(value)) {
      return InlineSimpleCall(*nested_call, call);
    }
    return Unsupported(function_, "inline value", *value);
  }

  absl::StatusOr<std::string> InlineSimpleCall(
      const llvm::CallInst& nested_call, const llvm::CallInst& outer_call) {
    const llvm::Function* callee = nested_call.getCalledFunction();
    if (callee == nullptr) return Unsupported(function_, "inline call", nested_call);
    llvm::StringRef name = callee->getName();
    if (name.starts_with("llvm.maximum.") || name.starts_with("llvm.smax.")) {
      return InlineSimpleBinaryCall("max", nested_call, outer_call);
    }
    if (name.starts_with("llvm.minimum.") || name.starts_with("llvm.smin.")) {
      return InlineSimpleBinaryCall("min", nested_call, outer_call);
    }
    return Unsupported(function_, "inline call", nested_call);
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

  absl::StatusOr<std::string> UnaryMathCall(absl::string_view msl_name,
                                            const llvm::CallInst& call) {
    if (call.arg_size() != 1) return Unsupported(function_, "math call", call);
    TF_ASSIGN_OR_RETURN(std::string operand, Expr(call.getArgOperand(0)));
    return absl::StrCat(msl_name, "(", operand, ")");
  }

  absl::StatusOr<std::string> BinaryMathCall(absl::string_view msl_name,
                                             const llvm::CallInst& call) {
    if (call.arg_size() != 2) return Unsupported(function_, "math call", call);
    TF_ASSIGN_OR_RETURN(std::string lhs, Expr(call.getArgOperand(0)));
    TF_ASSIGN_OR_RETURN(std::string rhs, Expr(call.getArgOperand(1)));
    return absl::StrCat(msl_name, "(", lhs, ", ", rhs, ")");
  }

  absl::StatusOr<std::string> TernaryMathCall(absl::string_view msl_name,
                                              const llvm::CallInst& call) {
    if (call.arg_size() != 3) return Unsupported(function_, "math call", call);
    TF_ASSIGN_OR_RETURN(std::string arg0, Expr(call.getArgOperand(0)));
    TF_ASSIGN_OR_RETURN(std::string arg1, Expr(call.getArgOperand(1)));
    TF_ASSIGN_OR_RETURN(std::string arg2, Expr(call.getArgOperand(2)));
    return absl::StrCat(msl_name, "(", arg0, ", ", arg1, ", ", arg2, ")");
  }

  absl::StatusOr<std::string> VoidCallStatement(const llvm::CallInst& call) {
    const llvm::Function* callee = call.getCalledFunction();
    if (callee == nullptr) return Unsupported(function_, "indirect call", call);
    llvm::StringRef name = callee->getName();
    if (name == "llvm.nvvm.barrier.cta.sync.aligned.all" ||
        name == "llvm.cuda.syncthreads") {
      return "threadgroup_barrier(mem_flags::mem_threadgroup);";
    }
    return Unsupported(function_, "void call", call);
  }

  absl::StatusOr<PtrExpr> PointerExprFor(const llvm::Value* value) {
    if (auto* argument = llvm::dyn_cast<llvm::Argument>(value)) {
      return PtrExpr{Name(argument), "0", PointerElementType(argument)};
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
          TF_ASSIGN_OR_RETURN(
              int64_t scale,
              FlattenedElementCount(data_layout_, scaled_type,
                                    base.element_type));
          TF_ASSIGN_OR_RETURN(std::string index_expr, Expr(index));
          offset = AddIndex(offset, ScaleIndex(index_expr, scale));
        }
      }

      TF_ASSIGN_OR_RETURN(current_type,
                          TypeAfterGepIndex(current_type, index, first_index));
      first_index = false;
    }

    return PtrExpr{base.base, AddIndex(base.index, offset), base.element_type};
  }

  const llvm::Function& function_;
  const llvm::DataLayout& data_layout_;
  absl::flat_hash_map<const llvm::Value*, llvm::Type*> pointer_element_types_;
  absl::flat_hash_map<const llvm::Value*, std::string> names_;
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
                      absl::StrJoin(kernels, "\n"));
}

}  // namespace gpu
}  // namespace xla
