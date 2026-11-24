//===-- WebAssemblyIfElse.cpp - Form if/else from block/br_if -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// PROTOTYPE. CFGStackify only places `block` and `loop`, so every forward
/// conditional skip is `block; <cond>; br_if 0; ...; end` (5 bytes of control)
/// and every diamond is `block; block; <cond>; br_if 0; B; br 1; end; C; end`
/// (10 bytes). After depths are final, rewrite them as
///   <!cond>; if; ...; end              (3 bytes)
///   <!cond>; if; B; else; C; end       (4 bytes)
/// Only void blocks, no EH instructions in the rewritten region.
///
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/WebAssemblyMCTargetDesc.h"
#include "MCTargetDesc/WebAssemblyMCTypeUtilities.h"
#include "WebAssembly.h"
#include "WebAssemblyMachineFunctionInfo.h"
#include "WebAssemblySubtarget.h"
#include "WebAssemblyUtilities.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/Support/CommandLine.h"
using namespace llvm;

#define DEBUG_TYPE "wasm-if-else"

static cl::opt<bool> EnableIfElse("wasm-enable-if-else", cl::Hidden, cl::init(false),
                                  cl::desc("WebAssembly: form if/else"));
static cl::opt<bool> EnableElse("wasm-if-else-diamonds", cl::Hidden,
                                cl::init(true),
                                cl::desc("WebAssembly: also form else arms"));

namespace {
class WebAssemblyIfElse final : public MachineFunctionPass {
  StringRef getPassName() const override { return "WebAssembly If/Else"; }
  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
  bool runOnMachineFunction(MachineFunction &MF) override;

public:
  static char ID;
  WebAssemblyIfElse() : MachineFunctionPass(ID) {}
};
} // end anonymous namespace

char WebAssemblyIfElse::ID = 0;
INITIALIZE_PASS(WebAssemblyIfElse, DEBUG_TYPE, "Form if/else", false, false)

FunctionPass *llvm::createWebAssemblyIfElse() { return new WebAssemblyIfElse(); }

using namespace WebAssembly;

static bool opensScope(unsigned Opc) {
  return Opc == BLOCK || Opc == LOOP || Opc == TRY || Opc == TRY_TABLE ||
         Opc == IF;
}
static bool closesScope(unsigned Opc) {
  return Opc == END_BLOCK || Opc == END_LOOP || Opc == END_TRY ||
         Opc == END_TRY_TABLE || Opc == END_IF || Opc == DELEGATE;
}
static bool isEH(unsigned Opc) {
  switch (Opc) {
  case TRY:
  case TRY_TABLE:
  case END_TRY:
  case END_TRY_TABLE:
  case DELEGATE:
  case RETHROW:
  case CATCH_LEGACY:
  case CATCH_ALL_LEGACY:
    return true;
  default:
    return false;
  }
}
static bool isMarkerOrBranch(const MachineInstr &MI) {
  unsigned Opc = MI.getOpcode();
  return opensScope(Opc) || closesScope(Opc) || Opc == ELSE || MI.isBranch() ||
         isEH(Opc);
}

static unsigned inverseCompare(unsigned Opc) {
  static const std::pair<unsigned, unsigned> Pairs[] = {
      {EQ_I32, NE_I32},     {GT_S_I32, LE_S_I32}, {GE_S_I32, LT_S_I32},
      {GT_U_I32, LE_U_I32}, {GE_U_I32, LT_U_I32}, {EQ_I64, NE_I64},
      {GT_S_I64, LE_S_I64}, {GE_S_I64, LT_S_I64}, {GT_U_I64, LE_U_I64},
      {GE_U_I64, LT_U_I64}};
  for (auto [A, B] : Pairs) {
    if (Opc == A)
      return B;
    if (Opc == B)
      return A;
  }
  return 0;
}

// Stack height effect of a (post-ExplicitLocals, fully stackified) instruction.
static int stackDelta(const MachineInstr &MI) {
  int D = 0;
  for (const MachineOperand &MO : MI.explicit_operands())
    if (MO.isReg())
      D += MO.isDef() ? 1 : -1;
  return D;
}

/// Index of the instruction closing the scope opened at Open.
static int findEnd(ArrayRef<MachineInstr *> L, int Open) {
  int Depth = 0;
  for (int I = Open; I < (int)L.size(); ++I) {
    unsigned Opc = L[I]->getOpcode();
    if (opensScope(Opc))
      ++Depth;
    else if (closesScope(Opc) && --Depth == 0)
      return I;
  }
  return -1;
}

/// Return the condition for `if` that runs the fallthrough of `br_if Cond`,
/// i.e. the negation of Cond, inverting a stackified compare for free.
static Register negate(MachineInstr &BrIf, MachineFunction &MF) {
  auto &MFI = *MF.getInfo<WebAssemblyFunctionInfo>();
  auto &MRI = MF.getRegInfo();
  const auto &TII = *MF.getSubtarget<WebAssemblySubtarget>().getInstrInfo();
  Register Cond = BrIf.getOperand(1).getReg();
  if (MFI.isVRegStackified(Cond) && MRI.hasOneNonDBGUse(Cond))
    if (MachineInstr *Def = MRI.getUniqueVRegDef(Cond)) {
      if (unsigned Inv = inverseCompare(Def->getOpcode())) {
        Def->setDesc(TII.get(Inv));
        return Cond;
      }
      if (Def->getOpcode() == EQZ_I32) {
        Register In = Def->getOperand(1).getReg();
        if (MFI.isVRegStackified(In)) {
          Def->eraseFromParent();
          return In;
        }
      }
    }
  Register Tmp = MRI.createVirtualRegister(&WebAssembly::I32RegClass);
  BuildMI(*BrIf.getParent(), BrIf, BrIf.getDebugLoc(), TII.get(EQZ_I32), Tmp)
      .addReg(Cond);
  MFI.stackifyVReg(MRI, Tmp);
  return Tmp;
}

// Depth operands of a branch instruction (post-CFGStackify immediates).
static SmallVector<MachineOperand *, 4> depthOperands(MachineInstr &MI) {
  SmallVector<MachineOperand *, 4> R;
  switch (MI.getOpcode()) {
  case BR:
  case BR_IF:
  case BR_UNLESS:
    R.push_back(&MI.getOperand(0));
    break;
  case BR_TABLE_I32:
  case BR_TABLE_I64:
    for (MachineOperand &MO : MI.explicit_uses())
      if (MO.isImm())
        R.push_back(&MO);
    break;
  default:
    break;
  }
  return R;
}

static bool tryForm(MachineFunction &MF, SmallVectorImpl<MachineInstr *> &L,
                    int I) {
  const auto &TII = *MF.getSubtarget<WebAssemblySubtarget>().getInstrInfo();
  MachineInstr *Outer = nullptr;
  int B = I;
  if (EnableElse && I + 1 < (int)L.size() && L[I + 1]->getOpcode() == BLOCK &&
      L[I]->getOperand(0).getImm() == int64_t(WebAssembly::BlockType::Void))
    Outer = L[I], B = I + 1;
  for (int Attempt = 0; Attempt < 2; ++Attempt) {
    if (Attempt == 1) {
      if (!Outer)
        return false;
      Outer = nullptr; // retry as a plain `if` on the inner block
    }
    MachineInstr *Block = L[B];
    if (Block->getOperand(0).getImm() != int64_t(WebAssembly::BlockType::Void))
      return false;
    // <cond> ... br_if 0, in the block's MBB, balanced.
    int J = B + 1, H = 0;
    while (J < (int)L.size() && L[J]->getParent() == Block->getParent() &&
           !isMarkerOrBranch(*L[J]))
      H += stackDelta(*L[J++]);
    if (J >= (int)L.size() || L[J]->getOpcode() != BR_IF ||
        L[J]->getParent() != Block->getParent() ||
        L[J]->getOperand(0).getImm() != 0 || H != 1)
      return false;
    MachineInstr *BrIf = L[J];
    int End = findEnd(L, B);
    if (End < 0 || L[End]->getOpcode() != END_BLOCK)
      return false;
    if (!Outer) {
      Register C = negate(*BrIf, MF);
      BuildMI(*BrIf->getParent(), BrIf, BrIf->getDebugLoc(), TII.get(IF))
          .addImm(int64_t(WebAssembly::BlockType::Void))
          .addReg(C);
      BrIf->eraseFromParent();
      Block->eraseFromParent();
      L[End]->setDesc(TII.get(END_IF));
      return true;
    }
    // Diamond: `br 1` right before the inner end, B balanced at it, and no
    // other branch in B to the inner block.
    MachineInstr *Br = L[End - 1];
    if (Br->getOpcode() != BR || Br->getOperand(0).getImm() != 1 ||
        End - 1 <= J)
      continue;
    {
      int H2 = 0;
      for (MachineInstr &MI : *Br->getParent()) {
        if (&MI == Br)
          break;
        H2 += isMarkerOrBranch(MI) ? 0 : stackDelta(MI);
      }
      if (H2 != 0 || Br->getParent()->isEHPad())
        continue;
    }
    int OuterEnd = findEnd(L, I);
    if (OuterEnd < 0 || L[OuterEnd]->getOpcode() != END_BLOCK)
      continue;
    bool OK = true;
    SmallVector<std::pair<MachineOperand *, int64_t>, 8> Fix;
    int K = 0;
    for (int X = J + 1; X < End - 1 && OK; ++X) {
      MachineInstr &MI = *L[X];
      unsigned Opc = MI.getOpcode();
      if (isEH(Opc)) {
        OK = false;
        break;
      }
      for (MachineOperand *MO : depthOperands(MI)) {
        int64_t D = MO->getImm();
        if (D == K)
          OK = false;
        else if (D > K)
          Fix.push_back({MO, D - 1});
      }
      if (opensScope(Opc))
        ++K;
      else if (closesScope(Opc))
        --K;
    }
    if (!OK)
      continue;
    for (auto [MO, D] : Fix)
      MO->setImm(D);
    Register C = negate(*BrIf, MF);
    BuildMI(*BrIf->getParent(), BrIf, BrIf->getDebugLoc(), TII.get(IF))
        .addImm(int64_t(WebAssembly::BlockType::Void))
        .addReg(C);
    BrIf->eraseFromParent();
    Br->eraseFromParent();
    Block->eraseFromParent();
    Outer->eraseFromParent();
    L[End]->setDesc(TII.get(ELSE));
    L[OuterEnd]->setDesc(TII.get(END_IF));
    return true;
  }
  return false;
}

bool WebAssemblyIfElse::runOnMachineFunction(MachineFunction &MF) {
  if (!EnableIfElse || !MF.getFunction().hasOptSize())
    return false;
  bool Changed = false;
  for (bool Again = true; Again;) {
    Again = false;
    SmallVector<MachineInstr *, 256> L;
    for (MachineBasicBlock &MBB : MF)
      for (MachineInstr &MI : MBB)
        if (!MI.isDebugInstr())
          L.push_back(&MI);
    for (int I = 0; I < (int)L.size(); ++I)
      if (L[I]->getOpcode() == BLOCK && tryForm(MF, L, I)) {
        Again = Changed = true;
        break;
      }
  }
  return Changed;
}
