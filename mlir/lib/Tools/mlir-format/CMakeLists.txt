add_mlir_library(MLIRFormatLib
  MlirFormatMain.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Tools/mlir-format

  LINK_LIBS PUBLIC
  MLIRBytecodeWriter
  MLIRDebug
  MLIRObservers
  MLIROptLib
  MLIRPass
  MLIRParser
  MLIRPluginsLib
  MLIRSupport
  MLIRIRDL
  )
