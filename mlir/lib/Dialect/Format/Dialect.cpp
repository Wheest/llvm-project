//===- Dialect.cpp - mlir-format IR Dialect registration in MLIR--------- === //
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the dialect for the mlir-format: custom type
// parsing and operation verification.
//
//===----------------------------------------------------------------------===//
#include "mlir/Dialect/Format/Format.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include <algorithm>
#include <string>

using namespace mlir;
using namespace mlir::mlirformat;

//===----------------------------------------------------------------------===//
// FormatDialect
//===----------------------------------------------------------------------===//

/// Dialect initialization, the instance will be owned by the context. This is
/// the point of registration of types and operations for the dialect.
void FormatDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/Format/FmtOps.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
//  Operations
//===----------------------------------------------------------------------===//

/// A generalized printer for comment operations.
// static void printCommentOp(mlir::OpAsmPrinter &printer, mlir::Operation *op)
// {
//   printer << " " << op->getOperands();
//   printer.printOptionalAttrDict(op->getAttrs());
//   printer << " :hey trying to print mlir-format comment";
// }

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "mlir/Dialect/Format/FmtOps.cpp.inc"
