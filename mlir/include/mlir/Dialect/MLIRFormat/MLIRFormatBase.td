//===- MLIRFormatBase.td - Base def for mlir-format dialect *- tablegen -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef FORMAT_MLIR_BASE
#define FORMAT_MLIR_BASE

include "mlir/IR/OpBase.td"

def MLIRFormat_Dialect : Dialect {
  let name = "mlirformat";
  let cppNamespace = "::mlir::mlirformat";
  let description = [{
    The mlir-format dialect is intended represent constructs used
    in the mlir-format tool, e.g., comments.
  }];
}
#endif // FORMAT_MLIR_BASE
