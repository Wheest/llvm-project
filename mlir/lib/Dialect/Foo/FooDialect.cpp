#include "mlir/Dialect/Foo/Foo.h"

using namespace mlir;
using namespace mlir::foo;

#include "mlir/Dialect/Foo/FooOpsDialect.cpp.inc"

void mlir::foo::FooDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/Foo/FooOps.cpp.inc"
      >();
}

#define GET_OP_CLASSES
#include "mlir/Dialect/Foo/FooOps.cpp.inc"
