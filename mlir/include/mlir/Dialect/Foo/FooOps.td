//===- FooOps.td - Foo op definitions ----------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FOO_OPS
#define FOO_OPS

include "mlir/Dialect/Foo/FooBase.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class Foo_Op<string mnemonic, list<Trait> traits = []> :
    Op<Foo_Dialect, mnemonic, traits>;

def BarOp : Foo_Op<"bar"> {
    let summary = "bar operation";
}

#endif // FOO_OPS
