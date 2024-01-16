//===- MlirOptUtil.h - MLIR Optimizer Driver util ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// These are utilities used by the main mlir-opt binary
//
//===----------------------------------------------------------------------===//
#ifndef MLIR_TOOLS_MLIROPT_MLIROPTUTIL_H
#define MLIR_TOOLS_MLIROPT_MLIROPTUTIL_H

#include "mlir/Support/LogicalResult.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "llvm/ADT/StringRef.h"

mlir::LogicalResult loadIRDLDialects(llvm::StringRef irdlFile,
                                     mlir::MLIRContext &ctx);
mlir::LogicalResult doVerifyRoundTrip(mlir::Operation *op,
                                      const mlir::MlirOptMainConfig &config);

#endif // MLIR_TOOLS_MLIROPT_MLIROPTUTIL_H
