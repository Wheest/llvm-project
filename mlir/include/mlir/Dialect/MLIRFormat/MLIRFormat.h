#ifndef MLIR_DIALECT_FORMAT_H_
#define MLIR_DIALECT_FORMAT_H_

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/VectorInterfaces.h"

//===----------------------------------------------------------------------===//
// mlir-format Dialect
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/MLIRFormat/MLIRFormatOpsDialect.h.inc"

//===----------------------------------------------------------------------===//
// mlir-format Dialect Operations
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "mlir/Dialect/MLIRFormat/MLIRFormatOps.h.inc"

#endif // MLIR_DIALECT_FORMAT_H_
