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

namespace {
class BytecodeVersionParser : public cl::parser<std::optional<int64_t>> {
public:
  BytecodeVersionParser(cl::Option &O)
      : cl::parser<std::optional<int64_t>>(O) {}

  bool parse(cl::Option &O, StringRef /*argName*/, StringRef arg,
             std::optional<int64_t> &v) {
    long long w;
    if (getAsSignedInteger(arg, 10, w))
      return O.error("Invalid argument '" + arg +
                     "', only integer is supported.");
    v = w;
    return false;
  }
};

/// This class is intended to manage the handling of command line options for
/// creating a *-opt config. This is a singleton.
struct MlirFormatMainConfigCLOptions : public MlirFormatMainConfig {
  MlirFormatMainConfigCLOptions() {
    // These options are static but all uses ExternalStorage to initialize the
    // members of the parent class. This is unusual but since this class is a
    // singleton it basically attaches command line option to the singleton
    // members.

    static cl::opt<bool, /*ExternalStorage=*/true> allowUnregisteredDialects(
        "allow-unregistered-dialect",
        cl::desc("Allow operation with no registered dialects"),
        cl::location(allowUnregisteredDialectsFlag), cl::init(false));

    static cl::opt<bool, /*ExternalStorage=*/true> dumpPassPipeline(
        "dump-pass-pipeline", cl::desc("Print the pipeline that will be run"),
        cl::location(dumpPassPipelineFlag), cl::init(false));

    static cl::opt<bool, /*ExternalStorage=*/true> emitBytecode(
        "emit-bytecode", cl::desc("Emit bytecode when generating output"),
        cl::location(emitBytecodeFlag), cl::init(false));

    static cl::opt<bool, /*ExternalStorage=*/true> elideResourcesFromBytecode(
        "elide-resource-data-from-bytecode",
        cl::desc("Elide resources when generating bytecode"),
        cl::location(elideResourceDataFromBytecodeFlag), cl::init(false));

    static cl::opt<std::optional<int64_t>, /*ExternalStorage=*/true,
                   BytecodeVersionParser>
        bytecodeVersion(
            "emit-bytecode-version",
            cl::desc("Use specified bytecode when generating output"),
            cl::location(emitBytecodeVersion), cl::init(std::nullopt));

    static cl::opt<std::string, /*ExternalStorage=*/true> irdlFile(
        "irdl-file",
        cl::desc("IRDL file to register before processing the input"),
        cl::location(irdlFileFlag), cl::init(""), cl::value_desc("filename"));

    static cl::opt<bool, /*ExternalStorage=*/true> enableDebuggerHook(
        "mlir-enable-debugger-hook",
        cl::desc("Enable Debugger hook for debugging MLIR Actions"),
        cl::location(enableDebuggerActionHookFlag), cl::init(false));

    static cl::opt<bool, /*ExternalStorage=*/true> explicitModule(
        "no-implicit-module",
        cl::desc("Disable implicit addition of a top-level module op during "
                 "parsing"),
        cl::location(useExplicitModuleFlag), cl::init(false));

    static cl::opt<bool, /*ExternalStorage=*/true> runReproducer(
        "run-reproducer", cl::desc("Run the pipeline stored in the reproducer"),
        cl::location(runReproducerFlag), cl::init(false));

    static cl::opt<bool, /*ExternalStorage=*/true> showDialects(
        "show-dialects",
        cl::desc("Print the list of registered dialects and exit"),
        cl::location(showDialectsFlag), cl::init(false));

    static cl::opt<bool, /*ExternalStorage=*/true> splitInputFile(
        "split-input-file",
        cl::desc("Split the input file into pieces and process each "
                 "chunk independently"),
        cl::location(splitInputFileFlag), cl::init(false));

    static cl::opt<bool, /*ExternalStorage=*/true> verifyDiagnostics(
        "verify-diagnostics",
        cl::desc("Check that emitted diagnostics match "
                 "expected-* lines on the corresponding line"),
        cl::location(verifyDiagnosticsFlag), cl::init(false));

    static cl::opt<bool, /*ExternalStorage=*/true> verifyPasses(
        "verify-each",
        cl::desc("Run the verifier after each transformation pass"),
        cl::location(verifyPassesFlag), cl::init(true));

    static cl::opt<bool, /*ExternalStorage=*/true> verifyRoundtrip(
        "verify-roundtrip",
        cl::desc("Round-trip the IR after parsing and ensure it succeeds"),
        cl::location(verifyRoundtripFlag), cl::init(false));

    static cl::list<std::string> passPlugins(
        "load-pass-plugin", cl::desc("Load passes from plugin library"));
    /// Set the callback to load a pass plugin.
    passPlugins.setCallback([&](const std::string &pluginPath) {
      auto plugin = PassPlugin::load(pluginPath);
      if (!plugin) {
        errs() << "Failed to load passes from '" << pluginPath
               << "'. Request ignored.\n";
        return;
      }
      plugin.get().registerPassRegistryCallbacks();
    });

    static cl::list<std::string> dialectPlugins(
        "load-dialect-plugin", cl::desc("Load dialects from plugin library"));
    this->dialectPlugins = std::addressof(dialectPlugins);

    static PassPipelineCLParser passPipeline("", "Compiler passes to run", "p");
    setPassPipelineParser(passPipeline);
  }

  /// Set the callback to load a dialect plugin.
  void setDialectPluginsCallback(DialectRegistry &registry);

  /// Pointer to static dialectPlugins variable in constructor, needed by
  /// setDialectPluginsCallback(DialectRegistry&).
  cl::list<std::string> *dialectPlugins = nullptr;
};
} // namespace

ManagedStatic<MlirFormatMainConfigCLOptions> clOptionsConfig;

void MlirFormatMainConfig::registerCLOptions(DialectRegistry &registry) {
  clOptionsConfig->setDialectPluginsCallback(registry);
  tracing::DebugConfig::registerCLOptions();
}

MlirFormatMainConfig MlirFormatMainConfig::createFromCLOptions() {
  clOptionsConfig->setDebugConfig(tracing::DebugConfig::createFromCLOptions());
  return *clOptionsConfig;
}

MlirFormatMainConfig &MlirFormatMainConfig::setPassPipelineParser(
    const PassPipelineCLParser &passPipeline) {
  passPipelineCallback = [&](PassManager &pm) {
    auto errorHandler = [&](const Twine &msg) {
      emitError(UnknownLoc::get(pm.getContext())) << msg;
      return failure();
    };
    if (failed(passPipeline.addToPipeline(pm, errorHandler)))
      return failure();
    if (this->shouldDumpPassPipeline()) {

      pm.dump();
      llvm::errs() << "\n";
    }
    return success();
  };
  return *this;
}

void MlirFormatMainConfigCLOptions::setDialectPluginsCallback(
    DialectRegistry &registry) {
  dialectPlugins->setCallback([&](const std::string &pluginPath) {
    auto plugin = DialectPlugin::load(pluginPath);
    if (!plugin) {
      errs() << "Failed to load dialect plugin from '" << pluginPath
             << "'. Request ignored.\n";
      return;
    };
    plugin.get().registerDialectRegistryCallbacks(registry);
  });
}

LogicalResult loadIRDLDialects(StringRef irdlFile, MLIRContext &ctx) {
  DialectRegistry registry;
  registry.insert<irdl::IRDLDialect>();
  ctx.appendDialectRegistry(registry);

  // Set up the input file.
  std::string errorMessage;
  std::unique_ptr<MemoryBuffer> file = openInputFile(irdlFile, &errorMessage);
  if (!file) {
    emitError(UnknownLoc::get(&ctx)) << errorMessage;
    return failure();
  }

  // Give the buffer to the source manager.
  // This will be picked up by the parser.
  SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(file), SMLoc());

  SourceMgrDiagnosticHandler sourceMgrHandler(sourceMgr, &ctx);

  // Parse the input file.
  OwningOpRef<ModuleOp> module(parseSourceFile<ModuleOp>(sourceMgr, &ctx));

  // Load IRDL dialects.
  return irdl::loadDialects(module.get());
}

// Return success if the module can correctly round-trip. This intended to test
// that the custom printers/parsers are complete.
static LogicalResult doVerifyRoundTrip(Operation *op,
                                       const MlirFormatMainConfig &config,
                                       bool useBytecode) {
  // We use a new context to avoid resource handle renaming issue in the diff.
  MLIRContext roundtripContext;
  OwningOpRef<Operation *> roundtripModule;
  roundtripContext.appendDialectRegistry(
      op->getContext()->getDialectRegistry());
  if (op->getContext()->allowsUnregisteredDialects())
    roundtripContext.allowUnregisteredDialects();
  StringRef irdlFile = config.getIrdlFile();
  if (!irdlFile.empty() && failed(loadIRDLDialects(irdlFile, roundtripContext)))
    return failure();

  // Print a first time with custom format (or bytecode) and parse it back to
  // the roundtripModule.
  {
    std::string buffer;
    llvm::raw_string_ostream ostream(buffer);
    if (useBytecode) {
      if (failed(writeBytecodeToFile(op, ostream))) {
        op->emitOpError()
            << "failed to write bytecode, cannot verify round-trip.\n";
        return failure();
      }
    } else {
      op->print(ostream,
                OpPrintingFlags().printGenericOpForm(false).enableDebugInfo());
    }
    FallbackAsmResourceMap fallbackResourceMap;
    ParserConfig parseConfig(&roundtripContext, /*verifyAfterParse=*/true,
                             &fallbackResourceMap);
    roundtripModule =
        parseSourceString<Operation *>(ostream.str(), parseConfig);
    if (!roundtripModule) {
      op->emitOpError()
          << "failed to parse bytecode back, cannot verify round-trip.\n";
      return failure();
    }
  }

  // Print in the generic form for the reference module and the round-tripped
  // one and compare the outputs.
  std::string reference, roundtrip;
  {
    llvm::raw_string_ostream ostreamref(reference);
    op->print(ostreamref,
              OpPrintingFlags().printGenericOpForm().enableDebugInfo());
    llvm::raw_string_ostream ostreamrndtrip(roundtrip);
    roundtripModule.get()->print(
        ostreamrndtrip,
        OpPrintingFlags().printGenericOpForm().enableDebugInfo());
  }
  if (reference != roundtrip) {
    // TODO implement a diff.
    return op->emitOpError() << "roundTrip testing roundtripped module differs "
                                "from reference:\n<<<<<<Reference\n"
                             << reference << "\n=====\n"
                             << roundtrip << "\n>>>>>roundtripped\n";
  }
  llvm::outs() << "Mlir-opt-main round trip verified\n";
  return success();
}

static LogicalResult doVerifyRoundTrip(Operation *op,
                                       const MlirFormatMainConfig &config) {
  // Textual round-trip isn't fully robust at the moment (for example implicit
  // terminator are losing location informations).

  return doVerifyRoundTrip(op, config, /*useBytecode=*/true);
}

// void replaceLineCommentWithActualComment(std::string &buffer) {
//   // Regular expression to find the "mlirformat.line_comment" operation.
//   // This regex includes capturing groups to preserve indentation.
//     // Define a regex to match the pattern
//     std::regex lineCommentRegex(R"((\s*)"mlirformat\.line_comment"\(\) \{str
//     = "([^"]*)"\} : \(\) -> \(\))");

//     // // Replace matches using a lambda function to keep the indentation
//     // std::string replaced = std::regex_replace(code, re, [](const
//     std::smatch& m) {
//     //     return m[1].str() + "// " + m[2].str();  // m[1] is indentation,
//     m[2] is the comment text
//     // });

//   // Perform the regex replacement.
//   buffer = std::regex_replace(buffer, lineCommentRegex, "$1// $2\n");
// }

std::string process_line_comment(std::string line,
                                 const MlirFormatMainConfig &config,
                                 MLIRContext *context) {
  const std::string commentFilePath = "/tmp/line_comment.mlir";
  std::error_code EC;
  llvm::raw_fd_ostream outFile(commentFilePath, EC, llvm::sys::fs::OF_None);
  if (EC) {
    llvm::errs() << "Error opening output file: " << commentFilePath << "\n";
  }
  outFile << line << "\n";
  outFile.close();
  std::string errorMessage;
  auto src_buffer = openInputFile(commentFilePath, &errorMessage);
  if (!src_buffer) {
    llvm::errs() << errorMessage << "\n";
    // return failure();
  }

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

  llvm::outs() << "MLIR format process is running!\n";
  AsmState asmState(op.get(), OpPrintingFlags(), /*locationMap=*/nullptr,
                    &fallbackResourceMap);

  llvm::raw_fd_ostream fileStream2("/tmp/output_post.mlir", EC,
                                   llvm::sys::fs::OF_Text);

  if (EC) {
    llvm::errs() << "Error: " << EC.message() << "\n";
    // Handle error appropriately.
  }
  auto my_op = op.get();

  for (Region &region : my_op->getRegions()) {
    for (Block &block : region.getBlocks()) {
      for (Operation &op : block.getOperations()) {
        Attribute comment = op.getAttr("str");
        std::ostringstream oss;
        std::string str;
        llvm::raw_string_ostream rso(str);
        comment.print(rso);

        // The flush() method is called to make sure all changes are committed
        // to the string
        rso.flush();

        // Remove the first and last characters (the quotes)
        std::string output = str;
        if (!output.empty()) {
          output.erase(0, 1);                 // Remove the first character
          output.erase(output.size() - 1, 1); // Remove the last character
        }
        llvm::outs() << "comment without quotes: " << output
                     << "\n"; // Outputs: Foo without quotes
        return output;
      }
    }
  }
}

void find_line_comments(const MlirFormatMainConfig &config,
                        MLIRContext *context) {
  const std::string inputFilePath = "/tmp/output.mlir";

  const std::string searchKeyword = "\"mlirformat.line_comment\"";

  std::ifstream inputFile(inputFilePath);
  if (!inputFile.is_open()) {
    llvm::errs() << "Error opening input file: " << inputFilePath << "\n";
  }

  std::vector<std::string> lines;
  std::string line;
  while (std::getline(inputFile, line)) {
    if (line.find(searchKeyword) != std::string::npos) {
      // Extract the leading whitespace (indentation)
      std::string leadingWhitespace =
          line.substr(0, line.find_first_not_of("\t\v\f "));
      line = leadingWhitespace + "//" +
             process_line_comment(line, config, context);
    }
    lines.push_back(line);
  }
  llvm::outs() << "searched for line comments!\n";

  inputFile.close();

  const std::string outputFilePath = "/tmp/output_mod.mlir";
  std::error_code EC;
  llvm::raw_fd_ostream outputFile(outputFilePath, EC, llvm::sys::fs::OF_Text);
  if (EC) {
    llvm::errs() << "Error opening file for writing: " << outputFilePath
                 << "\n";
    return;
  }

  // Write the modified lines back to the file
  for (const auto &modLine : lines) {
    outputFile << modLine << "\n";
  }
}

void mlir_format_process(llvm::raw_fd_ostream &fileStream,
                         const MlirFormatMainConfig &config,
                         MLIRContext *context) {
  // performs post-processing on the printed MLIR IR
  // replaces: `"mlirformat.line_comment"() {str = "foo"} : () -> ()`
  // with: `// foo`, keeping the indentation
  // replaceLineCommentWithActualComment(fileStream);

  // Disable multi-threading when parsing the input file. This removes the
  // unnecessary/costly context synchronization when parsing.
  //
  fileStream.flush();
  find_line_comments(config, context);
  llvm::outs() << "ran find_line_comments\n";
  // context->disableMultithreading();

  // std::string file_path =
  //     "/home/pez/proj/compiler_stuff/standalone_format/single_comment.mlir";

  // std::string errorMessage;
  // auto src_buffer = openInputFile(file_path, &errorMessage);
  // if (!src_buffer) {
  //   llvm::errs() << errorMessage << "\n";
  //   // return failure();
  // }

  // // Tell sourceMgr about this buffer, which is what the parser will pick
  // // up.
  // auto sourceMgr = std::make_shared<SourceMgr>();
  // sourceMgr->AddNewSourceBuffer(std::move(src_buffer), SMLoc());

  // PassReproducerOptions reproOptions;
  // FallbackAsmResourceMap fallbackResourceMap;
  // ParserConfig parseConfig(context, /*verifyAfterParse=*/true,
  //                          &fallbackResourceMap);
  // OwningOpRef<Operation *> op = parseSourceFileForTool(
  //     sourceMgr, parseConfig, !config.shouldUseExplicitModule());

  // llvm::outs() << "MLIR format process is running!\n";
  // AsmState asmState(op.get(), OpPrintingFlags(), /*locationMap=*/nullptr,
  //                   &fallbackResourceMap);

  // std::error_code EC;
  // llvm::raw_fd_ostream fileStream2("/tmp/output_post.mlir", EC,
  //                                  llvm::sys::fs::OF_Text);

  // if (EC) {
  //   llvm::errs() << "Error: " << EC.message() << "\n";
  //   // Handle error appropriately.
  // }
  // auto my_op = op.get();
  // // my_op.bodyRegion();
  // llvm::outs() << "My_op has this many regions: " << my_op->getNumRegions()
  //              << "\n";
  // // op.get()->print(fileStream, asmState);

  // for (Region &region : my_op->getRegions()) {
  //   for (Block &block : region.getBlocks()) {
  //     for (Operation &op : block.getOperations()) {
  //       Attribute comment = op.getAttr("str");
  //       std::ostringstream oss;
  //       std::string str;
  //       llvm::raw_string_ostream rso(str);
  //       comment.print(rso);

  //       // The flush() method is called to make sure all changes are
  //       committed
  //       // to the string
  //       rso.flush();

  //       // Remove the first and last characters (the quotes)
  //       std::string output = str;
  //       if (!output.empty()) {
  //         output.erase(0, 1);                 // Remove the first character
  //         output.erase(output.size() - 1, 1); // Remove the last character
  //       }
  //       llvm::outs() << "comment without quotes: " << output
  //                    << "\n"; // Outputs: Foo without quotes
  //     }
  //   }

  // llvm::outs() << "this region has: " << block.getOperations().size()
  //              << " operations\n";
  // }

  // Clear and truncate the file to write the modified content back.
  // fileStream.seek(0);
  // fileStream.flush();
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
               MLIRContext *context, const MlirFormatMainConfig &config) {
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

  std::error_code EC;
  llvm::raw_fd_ostream fileStream("/tmp/output.mlir", EC,
                                  llvm::sys::fs::OF_Text);

  if (EC) {
    llvm::errs() << "Error: " << EC.message() << "\n";
    // Handle error appropriately.
  }
  op.get()->print(fileStream, asmState);
  llvm::outs() << "Mlir-opt-main printed asmstate?\n";
  os << '\n';
  llvm::outs() << "Mlir-opt-main actions performed\n";

  mlir_format_process(fileStream, config, context);

  return success();
}

/// Parses the memory buffer.  If successfully, run a series of passes against
/// it and print the result.
static LogicalResult processBuffer(raw_ostream &os,
                                   std::unique_ptr<MemoryBuffer> ownedBuffer,
                                   const MlirFormatMainConfig &config,
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

std::pair<std::string, std::string>
mlir::registerAndParseCLIOptions(int argc, char **argv,
                                 llvm::StringRef toolName,
                                 DialectRegistry &registry) {
  static cl::opt<std::string> inputFilename(
      cl::Positional, cl::desc("<input file>"), cl::init("-"));

  static cl::opt<std::string> outputFilename("o", cl::desc("Output filename"),
                                             cl::value_desc("filename"),
                                             cl::init("-"));
  // Register any command line options.
  MlirFormatMainConfig::registerCLOptions(registry);
  registerAsmPrinterCLOptions();
  registerMLIRContextCLOptions();
  registerPassManagerCLOptions();
  registerDefaultTimingManagerCLOptions();
  tracing::DebugCounter::registerCLOptions();

  // Build the list of dialects as a header for the --help message.
  std::string helpHeader = (toolName + "\nAvailable Dialects: ").str();
  {
    llvm::raw_string_ostream os(helpHeader);
    interleaveComma(registry.getDialectNames(), os,
                    [&](auto name) { os << name; });
  }
  // Parse pass names in main to ensure static initialization completed.
  cl::ParseCommandLineOptions(argc, argv, helpHeader);
  return std::make_pair(inputFilename.getValue(), outputFilename.getValue());
}

LogicalResult mlir::MlirFormatMain(llvm::raw_ostream &outputStream,
                                   std::unique_ptr<llvm::MemoryBuffer> buffer,
                                   DialectRegistry &registry,
                                   const MlirFormatMainConfig &config) {
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

  MlirFormatMainConfig config = MlirFormatMainConfig::createFromCLOptions();

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
