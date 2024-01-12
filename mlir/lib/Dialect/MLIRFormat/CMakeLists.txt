add_mlir_dialect_library(MLIRFormatDialect
  MLIRFormatDialect.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/MLIRFormatDialect/MLIRFormat

  DEPENDS
  MLIRMLIRFormatOpsIncGen

  LINK_LIBS PUBLIC
  MLIRDialect
  MLIRIR
  MLIRUBDialect
)
