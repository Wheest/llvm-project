#include "mlir/Dialect/MLIRFormat/MLIRFormat.h"

using namespace mlir;
using namespace mlir::mlirformat;

#include "mlir/Dialect/MLIRFormat/MLIRFormatOpsDialect.cpp.inc"

void mlir::mlirformat::MLIRFormatDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/MLIRFormat/MLIRFormatOps.cpp.inc"
#undef GET_OP_LIST
      >();
}

#define GET_OP_CLASSES
#include "mlir/Dialect/MLIRFormat/MLIRFormatOps.cpp.inc"
