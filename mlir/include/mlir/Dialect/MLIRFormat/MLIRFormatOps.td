//===- Ops.td - mlir-format dialect operation definitions  -*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines the operations of the mlir-format dialect.
//
//===----------------------------------------------------------------------===//
#ifndef FORMAT_OPS
#define FORMAT_OPS

include "mlir/Dialect/MLIRFormat/MLIRFormatBase.td"
include "mlir/Pass/PassBase.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

// Base class for mlirformat dialect operations. This operation inherits from the base
// `Op` class in OpBase.td, and provides:
//   * The parent dialect of the operation.
//   * The mnemonic for the operation, or the name without the dialect prefix.
//   * A list of traits for the operation.
// class Format_Op<string mnemonic, list<Trait> traits = []> :
//     Op<MLIRFormat_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// mlir-format Operations
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// LineCommentOp
//===----------------------------------------------------------------------===//

// def LineCommentOp : Format_Op<"line_comment"> {
//   let summary = "line comment";
//   let description = [{
//     The "line_comment" builtin operation just stores a line comment, and produces
//     no results.
//   }];

//   // The operation takes an input tensor to print.
//   // let arguments = (ins FlatSymbolRefAttr:$comment);

//   let assemblyFormat = "`hey_we_have_out_comment` attr-dict";
// }
#endif // FORMAT_OPS
