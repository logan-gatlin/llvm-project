//===-- WebAssemblyConstHoist.cpp - Keep large constants in locals --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// PROTOTYPE. A constant that is used many times in a function and is
/// expensive to encode (static data addresses are 5-byte relocated
/// `i32.const`s, i64/f64 constants up to 10/9 bytes) is computed once at the
/// top of the function; every use then reads it from a local (2 bytes).
/// RegStackify must agree not to rematerialize it (see wasmConstSize there).
///
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/WebAssemblyMCTargetDesc.h"
#include "WebAssembly.h"
#include "WebAssemblySubtarget.h"
#include "WebAssemblyUtilities.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/LEB128.h"
using namespace llvm;

#define DEBUG_TYPE "wasm-const-hoist"

cl::opt<bool> WasmSizeConsts(
    "wasm-size-consts", cl::Hidden, cl::init(false),
    cl::desc("WebAssembly: keep large, often-used constants in locals"));

/// Encoded size in bytes of a constant-materializing instruction, or 0 if MI
/// is not one we handle.
unsigned llvm::wasmConstSize(const MachineInstr &MI) {
  switch (MI.getOpcode()) {
  case WebAssembly::CONST_I32:
  case WebAssembly::CONST_I64: {
    const MachineOperand &MO = MI.getOperand(1);
    if (MO.isImm())
      return 1 + getSLEB128Size(MO.getImm());
    if (MO.isGlobal() || MO.isSymbol())
      return 1 + 5; // relocated: padded to 5 bytes (4 after compression)
    return 0;
  }
  case WebAssembly::CONST_F32:
    return 5;
  case WebAssembly::CONST_F64:
    return 9;
  default:
    return 0;
  }
}

/// n uses of S bytes each vs. S + local.set + n local.gets.
bool llvm::wasmConstWorthALocal(unsigned Size, unsigned Uses) {
  return Size && Uses * Size > Size + 2 + 2 * Uses;
}

namespace {
class WebAssemblyConstHoist final : public MachineFunctionPass {
  StringRef getPassName() const override { return "WebAssembly Const Hoist"; }
  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
  bool runOnMachineFunction(MachineFunction &MF) override;

public:
  static char ID;
  WebAssemblyConstHoist() : MachineFunctionPass(ID) {}
};
} // end anonymous namespace

char WebAssemblyConstHoist::ID = 0;
INITIALIZE_PASS(WebAssemblyConstHoist, DEBUG_TYPE, "Hoist large constants",
                false, false)

FunctionPass *llvm::createWebAssemblyConstHoist() {
  return new WebAssemblyConstHoist();
}

using Key = std::tuple<unsigned, int64_t, const void *, int64_t, unsigned>;

static std::optional<Key> keyOf(const MachineInstr &MI) {
  const MachineOperand &MO = MI.getOperand(1);
  if (MO.isImm())
    return Key{MI.getOpcode(), MO.getImm(), nullptr, 0, 0};
  if (MO.isFPImm())
    return Key{MI.getOpcode(),
               (int64_t)MO.getFPImm()->getValueAPF().bitcastToAPInt()
                   .getZExtValue(),
               nullptr, 0, 1};
  if (MO.isGlobal())
    return Key{MI.getOpcode(), 0, MO.getGlobal(), MO.getOffset(),
               MO.getTargetFlags() + 2};
  return std::nullopt;
}

bool WebAssemblyConstHoist::runOnMachineFunction(MachineFunction &MF) {
  if (!WasmSizeConsts || !MF.getFunction().hasOptSize())
    return false;
  MachineRegisterInfo &MRI = MF.getRegInfo();
  MapVector<Key, SmallVector<MachineInstr *, 4>> Groups;
  for (MachineBasicBlock &MBB : MF)
    for (MachineInstr &MI : MBB) {
      if (!wasmConstSize(MI))
        continue;
      Register R = MI.getOperand(0).getReg();
      if (!R.isVirtual() || !MRI.hasOneDef(R))
        continue;
      if (auto K = keyOf(MI))
        Groups[*K].push_back(&MI);
    }

  MachineBasicBlock &Entry = MF.front();
  auto InsertPt = Entry.begin();
  while (InsertPt != Entry.end() && WebAssembly::isArgument(InsertPt->getOpcode()))
    ++InsertPt;
  bool Changed = false;
  for (auto &[K, Defs] : Groups) {
    unsigned Uses = 0;
    for (MachineInstr *D : Defs)
      Uses += std::distance(MRI.use_nodbg_begin(D->getOperand(0).getReg()),
                            MRI.use_nodbg_end());
    unsigned Size = wasmConstSize(*Defs[0]);
    if (!wasmConstWorthALocal(Size, Uses))
      continue;
    MachineInstr *Keep = Defs[0];
    Register KeepReg = Keep->getOperand(0).getReg();
    Entry.splice(InsertPt, Keep->getParent(), Keep->getIterator());
    for (MachineInstr *D : ArrayRef(Defs).drop_front()) {
      MRI.replaceRegWith(D->getOperand(0).getReg(), KeepReg);
      D->eraseFromParent();
    }
    Changed = true;
  }
  return Changed;
}
