//===-- WebAssemblyFrameAddrCSE.cpp - Compute frame addresses once --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// PROTOTYPE. eliminateFrameIndex materializes `ADD $frame, (CONST off)` at
/// every non-memory use of a frame index (call arguments, stored pointers),
/// which costs `local.get $fp; i32.const off; i32.add` (4-6 bytes) each time.
/// When the frame register is defined only in the prologue, compute each
/// address used often enough once, right after the prologue, and reuse it.
///
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/WebAssemblyMCTargetDesc.h"
#include "WebAssembly.h"
#include "WebAssemblyFrameLowering.h"
#include "WebAssemblySubtarget.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/LEB128.h"
using namespace llvm;

#define DEBUG_TYPE "wasm-frame-addr-cse"

static cl::opt<bool> EnableFrameAddrCSE(
    "wasm-enable-frame-addr-cse", cl::Hidden, cl::init(false),
    cl::desc("WebAssembly: compute frequently used frame addresses once"));

namespace {
class WebAssemblyFrameAddrCSE final : public MachineFunctionPass {
  StringRef getPassName() const override {
    return "WebAssembly Frame Address CSE";
  }
  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
  bool runOnMachineFunction(MachineFunction &MF) override;

public:
  static char ID;
  WebAssemblyFrameAddrCSE() : MachineFunctionPass(ID) {}
};
} // end anonymous namespace

char WebAssemblyFrameAddrCSE::ID = 0;
INITIALIZE_PASS(WebAssemblyFrameAddrCSE, DEBUG_TYPE,
                "Compute frame addresses once", false, false)

FunctionPass *llvm::createWebAssemblyFrameAddrCSE() {
  return new WebAssemblyFrameAddrCSE();
}

bool WebAssemblyFrameAddrCSE::runOnMachineFunction(MachineFunction &MF) {
  if (!EnableFrameAddrCSE || !MF.getFunction().hasOptSize())
    return false;
  MachineRegisterInfo &MRI = MF.getRegInfo();
  const auto &ST = MF.getSubtarget<WebAssemblySubtarget>();
  const auto *TII = ST.getInstrInfo();
  Register Base = ST.getRegisterInfo()->getFrameRegister(MF);
  unsigned AddOpc = WebAssemblyFrameLowering::getOpcAdd(MF);
  unsigned ConstOpc = WebAssemblyFrameLowering::getOpcConst(MF);
  MachineBasicBlock &Entry = MF.front();

  // The frame register must only be defined in the entry block; the address
  // is computed after its last definition there.
  MachineBasicBlock::iterator InsertPt = Entry.begin();
  bool SawDef = false;
  for (MachineInstr &MI : Entry)
    if (MI.modifiesRegister(Base, /*TRI=*/nullptr)) {
      InsertPt = std::next(MI.getIterator());
      SawDef = true;
    }
  if (!SawDef)
    return false;
  for (MachineBasicBlock &MBB : MF)
    if (&MBB != &Entry)
      for (MachineInstr &MI : MBB)
        if (MI.modifiesRegister(Base, /*TRI=*/nullptr))
          return false;

  // Occurrences of `%a = ADD Base, %c` with `%c = CONST off` used once.
  MapVector<int64_t, SmallVector<MachineInstr *, 4>> Occ;
  bool BeforeInsertPt = true;
  for (MachineBasicBlock &MBB : MF)
    for (MachineInstr &MI : MBB) {
      if (&MBB == &Entry && BeforeInsertPt) {
        if (InsertPt != Entry.end() && &MI == &*InsertPt)
          BeforeInsertPt = false;
        else
          continue;
      }
      if (MI.getOpcode() != AddOpc || !MI.getOperand(1).isReg() ||
          MI.getOperand(1).getReg() != Base || !MI.getOperand(2).isReg())
        continue;
      Register C = MI.getOperand(2).getReg();
      if (!C.isVirtual() || !MRI.hasOneNonDBGUse(C))
        continue;
      MachineInstr *Def = MRI.getUniqueVRegDef(C);
      if (!Def || Def->getOpcode() != ConstOpc || !Def->getOperand(1).isImm())
        continue;
      Occ[Def->getOperand(1).getImm()].push_back(&MI);
    }

  bool Changed = false;
  for (auto &[Off, Adds] : Occ) {
    // n * (get + const + add) > (get + const + add) + set + n * get
    int64_t N = Adds.size();
    int64_t Pattern = 2 + 1 + getSLEB128Size(Off) + 1;
    if (N * Pattern <= Pattern + 2 + N * 2)
      continue;
    const TargetRegisterClass *RC = MRI.getRegClass(Adds[0]->getOperand(0).getReg());
    Register NewC = MRI.createVirtualRegister(RC);
    Register NewA = MRI.createVirtualRegister(RC);
    DebugLoc DL;
    BuildMI(Entry, InsertPt, DL, TII->get(ConstOpc), NewC).addImm(Off);
    BuildMI(Entry, InsertPt, DL, TII->get(AddOpc), NewA)
        .addReg(Base)
        .addReg(NewC);
    for (MachineInstr *Add : Adds) {
      Register Old = Add->getOperand(0).getReg();
      Register C = Add->getOperand(2).getReg();
      MachineInstr *Def = MRI.getUniqueVRegDef(C);
      MRI.replaceRegWith(Old, NewA);
      Add->eraseFromParent();
      Def->eraseFromParent();
    }
    Changed = true;
  }
  return Changed;
}
