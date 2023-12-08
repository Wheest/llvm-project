add_mlir_dialect_library(MLIRFormatDialect
  Dialect.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/inlude/mlir/Dialect/Format

  DEPENDS
  MLIRMemRefOpsIncGen
  # MLIRFormatOpsIncGen
  MLIRFmtOpsIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRArithUtils
  MLIRControlFlowInterfaces
  MLIRDialect
  MLIRDialectUtils
  MLIRInferTypeOpInterface
  MLIRIR
  MLIRMemorySlotInterfaces
  MLIRShapedOpInterfaces
  MLIRSideEffectInterfaces
  MLIRValueBoundsOpInterface
  MLIRViewLikeInterface
)
