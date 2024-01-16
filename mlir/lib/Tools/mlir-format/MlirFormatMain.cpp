//===- MlirFormatMain.cpp - MLIR Format Driver---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is a utility that formats MLIR files and prints the result back out.
//
//===----------------------------------------------------------------------===//

#include "mlir/Tools/mlir-format/MlirFormatMain.h"
#include "mlir/Bytecode/BytecodeWriter.h"
#include "mlir/Debug/CLOptionsSetup.h"
#include "mlir/Debug/Counter.h"
#include "mlir/Debug/DebuggerExecutionContextHook.h"
#include "mlir/Debug/ExecutionContext.h"
#include "mlir/Debug/Observers/ActionLogging.h"
#include "mlir/Dialect/IRDL/IR/IRDL.h"
#include "mlir/Dialect/IRDL/IRDLLoading.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/Timing.h"
#include "mlir/Support/ToolUtilities.h"
#include "mlir/Tools/ParseUtilities.h"
#include "mlir/Tools/Plugins/DialectPlugin.h"
#include "mlir/Tools/Plugins/PassPlugin.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Tools/mlir-opt/MlirOptUtil.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/Regex.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/StringSaver.h"
#include "llvm/Support/ThreadPool.h"
#include "llvm/Support/ToolOutputFile.h"

#include <fstream>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/raw_ostream.h>
#include <regex>
#include <string>

using namespace mlir;
using namespace llvm;

std::string process_comment(std::string line, const MlirOptMainConfig &config,
                            MLIRContext *context) {
  std::error_code EC;
  std::unique_ptr<llvm::MemoryBuffer> src_buffer =
      llvm::MemoryBuffer::getMemBuffer(llvm::StringRef(line), "comment:" + line,
                                       false);

  // Tell sourceMgr about this buffer, which is what the parser will pick
  // up.
  auto sourceMgr = std::make_shared<SourceMgr>();
  sourceMgr->AddNewSourceBuffer(std::move(src_buffer), SMLoc());

  PassReproducerOptions reproOptions;
  FallbackAsmResourceMap fallbackResourceMap;
  ParserConfig parseConfig(context, /*verifyAfterParse=*/true,
                           &fallbackResourceMap);
  OwningOpRef<Operation *> op = parseSourceFileForTool(
      sourceMgr, parseConfig, !config.shouldUseExplicitModule());

  AsmState asmState(op.get(), OpPrintingFlags(), /*locationMap=*/nullptr,
                    &fallbackResourceMap);

  llvm::raw_fd_ostream fileStream2("/tmp/output_post.mlir", EC,
                                   llvm::sys::fs::OF_Text);

  if (EC) {
    llvm::errs() << "Error: " << EC.message() << "\n";
    // Handle error appropriately.
  }
  auto my_op = op.get();
  std::string comment_str;
  // Fetch the comment op, and extract its attrs
  // expect only on op in this region
  for (Region &region : my_op->getRegions()) {
    for (Block &block : region.getBlocks()) {
      for (Operation &op : block.getOperations()) {
        Attribute comment_attr = op.getAttr("str");
        std::ostringstream oss;
        std::string str;
        llvm::raw_string_ostream rso(comment_str);
        comment_attr.print(rso);

        // The flush() method is called to make sure all changes are committed
        // to the string
        rso.flush();

        // Remove the first and last characters (the quotes)
        if (!comment_str.empty()) {
          comment_str.erase(0, 1);
          comment_str.erase(comment_str.size() - 1, 1);
        }
      }
    }
  }
  return comment_str;
}

void mlir_format_process(std::string &fileStr, const MlirOptMainConfig &config,
                         MLIRContext *context) {
  // performs post-processing on the printed MLIR IR

  // replace: `"mlirformat.comment"() {str = "foo"} : () -> ()`
  // with: `// foo`, keeping the indentation
  const std::string searchKeyword = "\"mlirformat.comment\"";
  std::vector<std::string> lines; // modified IR strings
  std::istringstream iss(fileStr);
  std::string line;
  while (std::getline(iss, line)) {
    if (line.find(searchKeyword) != std::string::npos) {
      // Extract the leading whitespace (indentation)
      std::string leadingWhitespace =
          line.substr(0, line.find_first_not_of("\t\v\f "));
      line = leadingWhitespace + "//" + process_comment(line, config, context);
    }
    lines.push_back(line);
  }
  llvm::outs() << "searched for line comments!\n";

  // Remove the inserted module wrapping
  bool removeModule = true; // TODO make this a condition check if the original
                            // file is wrapped by a module
  if (removeModule && !lines.empty() && lines.front() == "module {") {
    lines.erase(lines.begin()); // Remove first element
    if (!lines.empty()) {
      lines.pop_back(); // Remove last element
    }
    // Remove one level of indentation introduced by the module wrapping
    if (lines.size() > 1) {
      const std::string &first_line = lines[0];
      std::size_t indent_length = first_line.find_first_not_of(
          " \t"); // Assume tabs or spaces for whitespace

      if (indent_length != std::string::npos && indent_length > 0) {
        for (std::size_t i = 0; i < lines.size(); ++i) {
          std::string &line = lines[i];
          if (line.substr(0, indent_length).find_first_not_of(" \t") ==
              std::string::npos) {
            line.erase(0, indent_length);
          }
        }
      }
    }
  }

  // Write the modified lines back to the file
  const std::string outputFilePath = "/tmp/output_mod.mlir";
  std::error_code EC;
  llvm::raw_fd_ostream outputFile(outputFilePath, EC, llvm::sys::fs::OF_Text);
  if (EC) {
    llvm::errs() << "Error opening file for writing: " << outputFilePath
                 << "\n";
    return;
  }
  for (const auto &modLine : lines) {
    outputFile << modLine << "\n";
  }
  llvm::outs() << "ran find_comments\n";
}

/// Perform the actions on the input file indicated by the command line flags
/// within the specified context.
///
/// This typically parses the main source file, runs zero or more optimization
/// passes, then prints the output.
///
static LogicalResult
performActions(raw_ostream &os,
               const std::shared_ptr<llvm::SourceMgr> &sourceMgr,
               MLIRContext *context, const MlirOptMainConfig &config) {
  llvm::outs() << "Mlir-opt-main performing actions\n";
  DefaultTimingManager tm;
  applyDefaultTimingManagerCLOptions(tm);
  TimingScope timing = tm.getRootScope();

  // Disable multi-threading when parsing the input file. This removes the
  // unnecessary/costly context synchronization when parsing.
  bool wasThreadingEnabled = context->isMultithreadingEnabled();
  context->disableMultithreading();

  // Prepare the parser config, and attach any useful/necessary resource
  // handlers. Unhandled external resources are treated as passthrough, i.e.
  // they are not processed and will be emitted directly to the output
  // untouched.
  PassReproducerOptions reproOptions;
  FallbackAsmResourceMap fallbackResourceMap;
  ParserConfig parseConfig(context, /*verifyAfterParse=*/true,
                           &fallbackResourceMap);
  if (config.shouldRunReproducer())
    reproOptions.attachResourceParser(parseConfig);

  // Parse the input file and reset the context threading state.
  TimingScope parserTiming = timing.nest("Parser");
  OwningOpRef<Operation *> op = parseSourceFileForTool(
      sourceMgr, parseConfig, !config.shouldUseExplicitModule());
  parserTiming.stop();
  if (!op)
    return failure();
  llvm::outs() << "Mlir-opt-main parser complete?\n";
  // Perform round-trip verification if requested
  if (config.shouldVerifyRoundtrip() &&
      failed(doVerifyRoundTrip(op.get(), config)))
    return failure();

  context->enableMultithreading(wasThreadingEnabled);

  // Prepare the pass manager, applying command-line and reproducer options.
  PassManager pm(op.get()->getName(), PassManager::Nesting::Implicit);
  pm.enableVerifier(config.shouldVerifyPasses());
  if (failed(applyPassManagerCLOptions(pm)))
    return failure();
  pm.enableTiming(timing);
  if (config.shouldRunReproducer() && failed(reproOptions.apply(pm)))
    return failure();
  if (failed(config.setupPassPipeline(pm)))
    return failure();

  // Run the pipeline.
  if (failed(pm.run(*op)))
    return failure();

  // Print the output.
  llvm::outs() << "Mlir-opt-main printing output\n";
  TimingScope outputTiming = timing.nest("Output");
  if (config.shouldEmitBytecode()) {
    BytecodeWriterConfig writerConfig(fallbackResourceMap);
    if (auto v = config.bytecodeVersionToEmit())
      writerConfig.setDesiredBytecodeVersion(*v);
    if (config.shouldElideResourceDataFromBytecode())
      writerConfig.setElideResourceDataFlag();
    return writeBytecodeToFile(op.get(), os, writerConfig);
  }

  if (config.bytecodeVersionToEmit().has_value())
    return emitError(UnknownLoc::get(pm.getContext()))
           << "bytecode version while not emitting bytecode";
  AsmState asmState(op.get(), OpPrintingFlags(), /*locationMap=*/nullptr,
                    &fallbackResourceMap);
  llvm::outs() << "Mlir-opt-main about to print asmstate?\n";
  op.get()->print(os, asmState);

  llvm::outs() << "Mlir-opt-main printed asmstate?\n";
  os << '\n';
  llvm::outs() << "Mlir-opt-main actions performed\n";

  // Create a raw_string_ostream that writes the IR to a std::string.
  std::string irStr;
  llvm::raw_string_ostream rso(irStr);
  op.get()->print(rso, asmState);
  rso.flush();
  mlir_format_process(irStr, config, context);

  return success();
}

/// Parses the memory buffer.  If successfully, run a series of passes against
/// it and print the result.
static LogicalResult processBuffer(raw_ostream &os,
                                   std::unique_ptr<MemoryBuffer> ownedBuffer,
                                   const MlirOptMainConfig &config,
                                   DialectRegistry &registry,
                                   llvm::ThreadPool *threadPool) {
  // Tell sourceMgr about this buffer, which is what the parser will pick up.
  auto sourceMgr = std::make_shared<SourceMgr>();
  sourceMgr->AddNewSourceBuffer(std::move(ownedBuffer), SMLoc());

  // Create a context just for the current buffer. Disable threading on creation
  // since we'll inject the thread-pool separately.
  MLIRContext context(registry, MLIRContext::Threading::DISABLED);
  if (threadPool)
    context.setThreadPool(*threadPool);

  StringRef irdlFile = config.getIrdlFile();
  if (!irdlFile.empty() && failed(loadIRDLDialects(irdlFile, context)))
    return failure();

  // Parse the input file.
  context.allowUnregisteredDialects(config.shouldAllowUnregisteredDialects());
  if (config.shouldVerifyDiagnostics())
    context.printOpOnDiagnostic(false);

  tracing::InstallDebugHandler installDebugHandler(context,
                                                   config.getDebugConfig());

  // If we are in verify diagnostics mode then we have a lot of work to do,
  // otherwise just perform the actions without worrying about it.
  if (!config.shouldVerifyDiagnostics()) {
    SourceMgrDiagnosticHandler sourceMgrHandler(*sourceMgr, &context);
    return performActions(os, sourceMgr, &context, config);
  }

  SourceMgrDiagnosticVerifierHandler sourceMgrHandler(*sourceMgr, &context);

  // Do any processing requested by command line flags.  We don't care whether
  // these actions succeed or fail, we only care what diagnostics they produce
  // and whether they match our expectations.
  (void)performActions(os, sourceMgr, &context, config);

  llvm::outs() << "Mlir-opt-main finished\n";
  // Verify the diagnostic handler to make sure that each of the diagnostics
  // matched.
  return sourceMgrHandler.verify();
}

LogicalResult mlir::MlirFormatMain(llvm::raw_ostream &outputStream,
                                   std::unique_ptr<llvm::MemoryBuffer> buffer,
                                   DialectRegistry &registry,
                                   const MlirOptMainConfig &config) {
  if (config.shouldShowDialects()) {
    llvm::outs() << "Available Dialects: ";
    interleave(registry.getDialectNames(), llvm::outs(), ",");
    llvm::outs() << "\n";
  }

  // The split-input-file mode is a very specific mode that slices the file
  // up into small pieces and checks each independently.
  // We use an explicit threadpool to avoid creating and joining/destroying
  // threads for each of the split.
  ThreadPool *threadPool = nullptr;

  // Create a temporary context for the sake of checking if
  // --mlir-disable-threading was passed on the command line.
  // We use the thread-pool this context is creating, and avoid
  // creating any thread when disabled.
  MLIRContext threadPoolCtx;
  if (threadPoolCtx.isMultithreadingEnabled())
    threadPool = &threadPoolCtx.getThreadPool();

  auto chunkFn = [&](std::unique_ptr<MemoryBuffer> chunkBuffer,
                     raw_ostream &os) {
    return processBuffer(os, std::move(chunkBuffer), config, registry,
                         threadPool);
  };
  return splitAndProcessBuffer(std::move(buffer), chunkFn, outputStream,
                               config.shouldSplitInputFile(),
                               /*insertMarkerInOutput=*/true);
}

LogicalResult mlir::MlirFormatMain(int argc, char **argv,
                                   llvm::StringRef inputFilename,
                                   llvm::StringRef outputFilename,
                                   DialectRegistry &registry) {

  InitLLVM y(argc, argv);

  MlirOptMainConfig config = MlirOptMainConfig::createFromCLOptions();

  // When reading from stdin and the input is a tty, it is often a user mistake
  // and the process "appears to be stuck". Print a message to let the user know
  // about it!
  if (inputFilename == "-" &&
      sys::Process::FileDescriptorIsDisplayed(fileno(stdin)))
    llvm::errs() << "(processing input from stdin now, hit ctrl-c/ctrl-d to "
                    "interrupt)\n";

  // Set up the input file.
  std::string errorMessage;
  auto file = openInputFile(inputFilename, &errorMessage);
  if (!file) {
    llvm::errs() << errorMessage << "\n";
    return failure();
  }

  auto output = openOutputFile(outputFilename, &errorMessage);
  if (!output) {
    llvm::errs() << errorMessage << "\n";
    return failure();
  }
  if (failed(MlirFormatMain(output->os(), std::move(file), registry, config)))
    return failure();

  // Keep the output file if the invocation of MlirFormatMain was successful.
  output->keep();
  return success();
}

LogicalResult mlir::MlirFormatMain(int argc, char **argv,
                                   llvm::StringRef toolName,
                                   DialectRegistry &registry) {
  // Register and parse command line options.
  std::string inputFilename, outputFilename;
  std::tie(inputFilename, outputFilename) =
      registerAndParseCLIOptions(argc, argv, toolName, registry);

  return MlirFormatMain(argc, argv, inputFilename, outputFilename, registry);
}
