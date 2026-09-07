/* Copyright 2025 The OpenXLA Authors.

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

#ifndef XLA_HLO_ANALYSIS_SYMBOLIC_EXPR_H_
#define XLA_HLO_ANALYSIS_SYMBOLIC_EXPR_H_

#include <cstdint>
#include <functional>
#include <optional>
#include <ostream>
#include <string>
#include <vector>

#include "absl/types/span.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/LLVM.h"

namespace xla {

class SymbolicExprStorage;

typedef int64_t VariableID;

enum class SymbolicExprType {
  kAdd,
  kMul,
  kMod,
  kFloorDiv,
  kCeilDiv,
  kMax,
  kMin,
  kVariable,
  kConstant,  // Constant should be the last type for the comparator.
  // TODO: b/459357586 - Add kIn operator.
  // kIn,  // 'var in [a, b]' .
};

class SymbolicExpr {
 public:
  using ImplType = SymbolicExprStorage;
  /* implicit */ SymbolicExpr(const ImplType* impl = nullptr) : impl_(impl) {}

  explicit operator bool() const { return impl_ != nullptr; }
  bool operator!() const { return impl_ == nullptr; }
  bool operator==(SymbolicExpr other) const { return impl_ == other.impl_; }
  bool operator!=(SymbolicExpr other) const { return !(*this == other); }
  bool operator<(const SymbolicExpr& other) const;

  mlir::MLIRContext* GetContext() const;
  SymbolicExprType GetType() const;
  bool IsBinaryOp() const;
  SymbolicExpr GetLHS() const;
  SymbolicExpr GetRHS() const;
  int64_t GetValue() const;
  // If num_dims is provided, then the first num_dims variables are dimensions,
  // and the rest are symbols. If variable names are provided, then they are
  // used instead of numbers.
  std::string ToString(std::optional<int64_t> num_dims = std::nullopt) const;
  std::string ToString(absl::Span<const std::string> var_names) const;
  std::string ToString(absl::Span<const std::string> dim_names,
                       absl::Span<const std::string> sym_names) const;
  int64_t Evaluate(absl::Span<const int64_t> variable_values) const;

  // Safely evaluates the given expression, returning nullopt if the result is
  // undefined (due to undefined behavior, e.g. division by zero or overflow).
  SymbolicExpr ReplaceVariables(
      absl::Span<const SymbolicExpr> substitutions) const;
  // TODO: b/459357586 - These methods are needed for IndexingMap, but
  // dimensions and symbols are SymbolicMap specific. We should remove them once
  // we have a better way to integrate SymbolicExpr with IndexingMap. It is
  // assuming that dimensions are the first (0...num_dims-1) variables and
  // symbols are the rest.
  SymbolicExpr ReplaceDims(absl::Span<const SymbolicExpr> replacements) const;
  SymbolicExpr ReplaceDims(absl::Span<const SymbolicExpr> replacements,
                           int64_t current_num_dims, int64_t new_num_dims,
                           int64_t num_symbols) const;
  SymbolicExpr ReplaceSymbols(absl::Span<const SymbolicExpr> replacements,
                              int64_t num_dims) const;
  SymbolicExpr ReplaceDimsAndSymbols(
      absl::Span<const SymbolicExpr> dim_replacements,
      absl::Span<const SymbolicExpr> symbol_replacements) const;

  SymbolicExpr Canonicalize() const;

  /// Sparse replace method. Replace `expr` by `replacement` and return the
  /// modified expression tree.
  SymbolicExpr Replace(SymbolicExpr expr, SymbolicExpr replacement) const;

  /// Sparse replace method. If `*this` appears in `map` replaces it by
  /// `map[*this]` and return the modified expression tree. Otherwise traverse
  /// `*this` and apply replace with `map` on its subexpressions.
  SymbolicExpr Replace(
      const llvm::DenseMap<SymbolicExpr, SymbolicExpr>& replacements) const;

  void GetUsedVariables(llvm::DenseSet<VariableID>& used_vars) const;

  // Returns true if this expression depends on the given variable.
  bool IsFunctionOfVariable(VariableID var_id) const;

  // Traverses the expression tree and calls the callback for each
  // subexpression in postorder.
  void Walk(const std::function<void(SymbolicExpr)>& callback) const;

  // Return true if the expression is a multiple of `factor`.
  bool IsMultipleOf(int64_t factor) const;

  SymbolicExpr operator+(int64_t v) const;
  SymbolicExpr operator+(SymbolicExpr other) const;
  SymbolicExpr operator-() const;
  SymbolicExpr operator-(int64_t v) const;
  SymbolicExpr operator-(SymbolicExpr other) const;
  SymbolicExpr operator*(int64_t v) const;
  SymbolicExpr operator*(SymbolicExpr other) const;
  SymbolicExpr operator/(int64_t v) const { return this->floorDiv(v); }
  SymbolicExpr operator/(SymbolicExpr other) const {
    return this->floorDiv(other);
  }
  SymbolicExpr operator%(int64_t v) const;
  SymbolicExpr operator%(SymbolicExpr other) const;
  SymbolicExpr floorDiv(int64_t v) const;
  SymbolicExpr floorDiv(SymbolicExpr other) const;
  SymbolicExpr ceilDiv(int64_t v) const;
  SymbolicExpr ceilDiv(SymbolicExpr other) const;
  SymbolicExpr min(int64_t v) const;
  SymbolicExpr min(SymbolicExpr other) const;
  SymbolicExpr max(int64_t v) const;
  SymbolicExpr max(SymbolicExpr other) const;

  const ImplType* GetImpl() const { return impl_; }

  template <typename Sink>
  friend void AbslStringify(Sink& sink, const SymbolicExpr expr) {
    sink.Append(expr.ToString());
  }

  friend llvm::raw_ostream& operator<<(llvm::raw_ostream& os,
                                       const SymbolicExpr expr) {
    os << expr.ToString();
    return os;
  }

  friend std::ostream& operator<<(std::ostream& os, const SymbolicExpr expr) {
    os << expr.ToString();
    return os;
  }

 private:
  static int64_t EvaluateImpl(const ImplType* impl,
                              absl::Span<const int64_t> variable_values);

  const ImplType* impl_ = nullptr;
};

SymbolicExpr operator+(int64_t lhs, SymbolicExpr rhs);
SymbolicExpr operator*(int64_t lhs, SymbolicExpr rhs);

// Evaluates a batch of expressions with the same variable values. Terminal
// expressions are handled directly to avoid entering the recursive evaluator
// once per result.
void EvaluateSymbolicExprs(absl::Span<const SymbolicExpr> expressions,
                           absl::Span<const int64_t> variable_values,
                           absl::Span<int64_t> results);

// An immutable evaluation program for a batch of expressions sharing variable
// values. Preparation records each distinct subexpression once in dependency
// order. Evaluation neither allocates storage nor accesses the MLIR context, so
// the program can outlive the expressions and can be shared between threads
// with separate scratch and result buffers.
//
// This has the same arithmetic preconditions as SymbolicExpr::Evaluate; it is
// not a substitute for SafeEvaluateSymbolicExpr. In particular, callers must
// avoid integer overflow and division by zero.
class SymbolicExprProgram {
 public:
  explicit SymbolicExprProgram(absl::Span<const SymbolicExpr> expressions);

  int64_t scratch_size() const { return instructions_.size(); }
  int64_t result_count() const { return result_indices_.size(); }

  // Scratch must contain at least scratch_size() elements, results must contain
  // exactly result_count() elements, and the three spans must not overlap.
  // Every scratch value is overwritten, so the same storage can be reused for
  // the next set of variable values without clearing it.
  void Evaluate(absl::Span<const int64_t> variable_values,
                absl::Span<int64_t> scratch, absl::Span<int64_t> results) const;

 private:
  struct Instruction {
    SymbolicExprType type;
    // Constants and variables use lhs_or_value for their value or variable ID.
    // Binary operations use both fields as indices into scratch.
    int64_t lhs_or_value;
    int64_t rhs = 0;
  };

  std::vector<Instruction> instructions_;
  std::vector<int64_t> result_indices_;
};

inline ::llvm::hash_code hash_value(SymbolicExpr expr) {
  return ::llvm::hash_value(expr.GetImpl());
}

template <typename H>
H AbslHashValue(H h, const SymbolicExpr& expr) {
  return H::combine(std::move(h), hash_value(expr));
}

// This method should be called once permlir::MLIRContext to register the
// SymbolicExprStorage type with themlir::MLIRContext's uniquifier. It should be
// called before any SymbolicExprs are created.
void RegisterSymbolicExprStorage(mlir::MLIRContext* mlir_context);

// Helpers to create SymbolicExprs.
SymbolicExpr CreateSymbolicConstant(int64_t value,
                                    mlir::MLIRContext* mlir_context);
SymbolicExpr CreateSymbolicVariable(int64_t var_id,
                                    mlir::MLIRContext* mlir_context);
SymbolicExpr CreateSymbolicBinaryOp(SymbolicExprType type, SymbolicExpr lhs,
                                    SymbolicExpr rhs,
                                    mlir::MLIRContext* mlir_context);
llvm::SmallVector<SymbolicExpr> CreateSymbolicConstantExprs(
    llvm::ArrayRef<int64_t> constants, mlir::MLIRContext* mlir_context);

// TODO: b/459357586 - This method is needed for IndexingMap, but
// dimensions and symbols are SymbolicMap specific. We should refactor it and
// include it as a method inside the class.
std::optional<int64_t> SafeEvaluateSymbolicExpr(SymbolicExpr expr,
                                                absl::Span<int64_t const> dims,
                                                absl::Span<int64_t const> syms);

}  // namespace xla

namespace llvm {

// SymbolicExpr hash just like pointers
template <>
struct DenseMapInfo<xla::SymbolicExpr> {
  static unsigned getHashValue(xla::SymbolicExpr val) {
    return hash_value(val);
  }
  static bool isEqual(xla::SymbolicExpr LHS, xla::SymbolicExpr RHS) {
    return LHS == RHS;
  }
};

}  // namespace llvm

#endif  // XLA_HLO_ANALYSIS_SYMBOLIC_EXPR_H_
