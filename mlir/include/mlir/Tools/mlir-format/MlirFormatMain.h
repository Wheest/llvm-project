//===- MlirFormatMain.h - MLIR Format Driver main --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Main entry function for mlir-format for when built as standalone binary.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TOOLS_MLIRFORMAT_MLIRFORMATMAIN_H
#define MLIR_TOOLS_MLIRFORMAT_MLIRFORMATMAIN_H

#include "mlir/Debug/CLOptionsSetup.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "llvm/ADT/StringRef.h"

#include <cstdlib>
#include <functional>
#include <memory>

namespace llvm {
class raw_ostream;
class MemoryBuffer;
} // namespace llvm

namespace mlir {
/// Perform the core processing behind `mlir-format`.
/// - outputStream is the stream where the resulting IR is printed.
/// - buffer is the in-memory file to parser and process.
/// - registry should contain all the dialects that can be parsed in the source.
/// - config contains the configuration options for the tool.
LogicalResult MlirFormatMain(llvm::raw_ostream &outputStream,
                             std::unique_ptr<llvm::MemoryBuffer> buffer,
                             DialectRegistry &registry,
                             const MlirOptMainConfig &config,
                             bool removeModule);

/// Implementation for tools like `mlir-format`.
/// - toolName is used for the header displayed by `--help`.
/// - registry should contain all the dialects that can be parsed in the source.
LogicalResult MlirFormatMain(int argc, char **argv, llvm::StringRef toolName,
                             DialectRegistry &registry);

/// Implementation for tools like `mlir-format`.
/// This function can be used with registrationAndParseCLIOptions so that
/// CLI options can be accessed before running MlirFormatMain.
/// - inputFilename is the name of the input mlir file.
/// - outputFilename is the name of the output file.
/// - registry should contain all the dialects that can be parsed in the source.
LogicalResult MlirFormatMain(int argc, char **argv,
                             llvm::StringRef inputFilename,
                             llvm::StringRef outputFilename,
                             DialectRegistry &registry);

} // namespace mlir

#endif // MLIR_TOOLS_MLIRFORMAT_MLIRFORMATMAIN_H
