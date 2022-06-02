//===- LinalgToExternal.cpp - Convert Linalg to external C++ calls --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements lowering of Linalg to external C++ calls
//
//===----------------------------------------------------------------------===//

// #include <type_traits>

#include "mlir/Conversion/LinalgToExternal/LinalgToExternal.h"
#include "mlir/Conversion/LinalgToExternal/LinalgToExternalPass.h"


#include "../PassDetail.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/IR/FunctionInterfaces.h"

using namespace mlir;
using namespace mlir::linalg;

//===----------------------------------------------------------------------===//
// External Runtime C API declaration.
//===----------------------------------------------------------------------===//
static constexpr const char *compute = "compute";

// Marker used as attribute name in generated Linalg rewriting transformations.
const StringLiteral kLinalgTransformMarker = "__internal_linalg_transform__";

static void addExternalRuntimeApiDeclarations(ModuleOp module) {

  auto builder =
      ImplicitLocOpBuilder::atBlockEnd(module.getLoc(), module.getBody());

  MLIRContext *ctx = module.getContext();

  // Types
  // TODO, for now hardcoded to floats
  Type myType = builder.getF32Type();
  Type intTy = builder.getI64Type();
  Type indexTy = builder.getIndexType();
  Type unrankedType = UnrankedMemRefType::get(myType, 0);

  // auto addFuncDecl = [&](StringRef name, FunctionType type) {
  //   if (module.lookupSymbol<FuncOp>(name))
  //     return;
  //   builder.create<FuncOp>(name, type).setPrivate();
  //   assert(isa<FunctionOpInterface>(SymbolTable::lookupSymbolIn(module, name)));
  // };

    // addFuncDecl(kDmaInit,
    //           FunctionType::get(
    //               ctx, {indexTy, indexTy, indexTy, indexTy, indexTy}, {}));
}


std::unique_ptr<OperationPass<ModuleOp>>
mlir::createConvertLinalgToExternalPass() {
  return std::make_unique<ConvertLinalgToExternalPass>();
}

std::unique_ptr<OperationPass<ModuleOp>>
mlir::createConvertLinalgToExternalPass(
    const LinalgToExternalOptions &options) {
  return std::make_unique<ConvertLinalgToExternalPass>(options);
}
