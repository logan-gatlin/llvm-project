//===-- WebAssemblyZeroLocalSets.cpp - Drop writes of zero to zero locals -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// PROTOTYPE. Wasm zero-initializes a function's non-parameter locals. PHI
/// lowering writes `const 0; local.set $x` on paths where $x has not been
/// written yet, which is a no-op. After ExplicitLocals, a forward dataflow
/// over local indices finds locals still known to be zero and removes such
/// writes.
///
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/WebAssemblyMCTargetDesc.h"
#include "WebAssembly.h"
#include "WebAssemblyMachineFunctionInfo.h"
#include "WebAssemblySubtarget.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/Support/CommandLine.h"
using namespace llvm;

#define DEBUG_TYPE "wasm-zero-local-sets"

static cl::opt<bool> EnableZeroLocalSets(
    "wasm-enable-zero-local-sets", cl::Hidden, cl::init(false),
    cl::desc("WebAssembly: drop writes of zero to locals still zero"));

namespace {
class WebAssemblyZeroLocalSets final : public MachineFunctionPass {
  StringRef getPassName() const override {
    return "WebAssembly Zero Local Sets";
  }
  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
  bool runOnMachineFunction(MachineFunction &MF) override;

public:
  static char ID;
  WebAssemblyZeroLocalSets() : MachineFunctionPass(ID) {}
};
} // end anonymous namespace

char WebAssemblyZeroLocalSets::ID = 0;
INITIALIZE_PASS(WebAssemblyZeroLocalSets, DEBUG_TYPE,
                "Drop writes of zero to zero locals", false, false)

FunctionPass *llvm::createWebAssemblyZeroLocalSets() {
  return new WebAssemblyZeroLocalSets();
}

/// If MI writes a local, return its index and whether the value written is a
/// (bitwise) zero constant: the CONST instruction defining the stackified
/// operand, else nullptr.
static bool writesLocal(const MachineInstr &MI, const MachineRegisterInfo &MRI,
                        unsigned &Local, MachineInstr *&ZeroDef) {
  unsigned Opc = MI.getOpcode();
  bool Tee = WebAssembly::isLocalTee(Opc);
  if (!WebAssembly::isLocalSet(Opc) && !Tee)
    return false;
  unsigned LocalOp = Tee ? 1 : 0;
  Local = MI.getOperand(LocalOp).getImm();
  ZeroDef = nullptr;
  const MachineOperand &Src = MI.getOperand(LocalOp + 1);
  if (Src.isReg() && Src.getReg().isVirtual())
    if (MachineInstr *D = MRI.getUniqueVRegDef(Src.getReg())) {
      const MachineOperand &V = D->getNumOperands() > 1 ? D->getOperand(1)
                                                        : D->getOperand(0);
      switch (D->getOpcode()) {
      case WebAssembly::CONST_I32:
      case WebAssembly::CONST_I64:
        if (V.isImm() && V.getImm() == 0)
          ZeroDef = D;
        break;
      case WebAssembly::CONST_F32:
      case WebAssembly::CONST_F64:
        if (V.isFPImm() && V.getFPImm()->getValueAPF().isPosZero())
          ZeroDef = D;
        break;
      }
    }
  return true;
}

bool WebAssemblyZeroLocalSets::runOnMachineFunction(MachineFunction &MF) {
  if (!EnableZeroLocalSets)
    return false;
  const auto &MFI = *MF.getInfo<WebAssemblyFunctionInfo>();
  MachineRegisterInfo &MRI = MF.getRegInfo();
  unsigned NumParams = MFI.getParams().size();
  unsigned NumLocals = NumParams + MFI.getLocals().size();
  if (NumLocals == NumParams)
    return false;

  // In[MBB]: locals known to hold zero on entry to MBB.
  DenseMap<const MachineBasicBlock *, BitVector> In, Out;
  BitVector Top(NumLocals, true);
  BitVector EntryIn(NumLocals, true);
  EntryIn.reset(0, NumParams);
  ReversePostOrderTraversal<MachineFunction *> RPOT(&MF);
  for (MachineBasicBlock *MBB : RPOT)
    Out[MBB] = Top;
  auto Transfer = [&](const MachineBasicBlock &MBB, BitVector Z) {
    for (const MachineInstr &MI : MBB) {
      unsigned L;
      MachineInstr *ZeroDef;
      if (writesLocal(MI, MRI, L, ZeroDef) && L < NumLocals) {
        if (ZeroDef)
          Z.set(L);
        else
          Z.reset(L);
      }
    }
    return Z;
  };
  bool Iterate = true;
  while (Iterate) {
    Iterate = false;
    for (MachineBasicBlock *MBB : RPOT) {
      BitVector Z = MBB == &MF.front() ? EntryIn : Top;
      if (MBB != &MF.front() && MBB->pred_empty())
        Z.reset();
      for (MachineBasicBlock *P : MBB->predecessors())
        Z &= Out[P];
      if (MBB->isEHPad())
        Z.reset();
      In[MBB] = Z;
      BitVector O = Transfer(*MBB, Z);
      if (O != Out[MBB]) {
        Out[MBB] = O;
        Iterate = true;
      }
    }
  }

  bool Changed = false;
  for (MachineBasicBlock &MBB : MF) {
    BitVector Z = In.count(&MBB) ? In[&MBB] : BitVector(NumLocals);
    for (MachineInstr &MI : llvm::make_early_inc_range(MBB)) {
      unsigned L;
      MachineInstr *ZeroDef;
      if (!writesLocal(MI, MRI, L, ZeroDef) || L >= NumLocals)
        continue;
      if (!ZeroDef) {
        Z.reset(L);
        continue;
      }
      if (!Z.test(L) || WebAssembly::isLocalTee(MI.getOpcode())) {
        Z.set(L);
        continue;
      }
      Register R = MI.getOperand(1).getReg();
      MI.eraseFromParent();
      if (MRI.use_nodbg_empty(R))
        ZeroDef->eraseFromParent();
      Changed = true;
    }
  }
  return Changed;
}
