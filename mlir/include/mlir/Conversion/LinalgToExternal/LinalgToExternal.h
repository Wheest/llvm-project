//===- LinalgToExternal.h - Convert Linalg to AXI4MLIR calls ----*- C++ -*-===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_LINALGTOEXTERNAL_H_
#define MLIR_CONVERSION_LINALGTOEXTERNAL_H_

#include "mlir/IR/PatternMatch.h"
namespace mlir {
class MLIRContext;
class RewritePatternSet;
class ModuleOp;
template <typename T>
class OperationPass;

struct LinalgToExternalOptions {
  /// Tile Size information
  unsigned tileSize = 1;

  /// DMA Information
  unsigned dmaAddress = 0;
  unsigned dmaInputAddress = 0;
  unsigned dmaInputBufferSize = 100000;
  unsigned dmaOutputAddress = 100000;
  unsigned dmaOutputBufferSize = 100000;

  /// Flow information
  bool flowCpuAcc = false;
};

/// Collect a set of patterns to convert from the Linagl dialect to External
/// calls
void populateLinalgToExternalConversionPatterns(
    RewritePatternSet &patterns,
    const LinalgToExternalOptions &options = LinalgToExternalOptions());

} // namespace mlir

#endif // MLIR_CONVERSION_LINALGTOEXTERNAL_H_
