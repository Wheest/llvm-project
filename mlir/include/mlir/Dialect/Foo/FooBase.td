//===- FooBase.td - Base definitions for foo dialect --------*- tablegen -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef FOO_BASE
#define FOO_BASE
include "mlir/IR/OpBase.td"
def Foo_Dialect : Dialect {
  let name = "foo";
  let cppNamespace = "::mlir::foo";
  let description = [{
    Lorem Ipsum
  }];
}
#endif // FOO_BASE
