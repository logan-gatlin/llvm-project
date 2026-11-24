//===-- WebAssemblyExplicitLocals.cpp - Make Locals Explicit --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file converts any remaining registers into WebAssembly locals.
///
/// After register stackification and register coloring, convert non-stackified
/// registers into locals, inserting explicit local.get and local.set
/// instructions.
///
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/WebAssemblyMCTargetDesc.h"
#include "WebAssembly.h"
#include "WebAssemblyDebugValueManager.h"
#include "WebAssemblyMachineFunctionInfo.h"
#include "WebAssemblySubtarget.h"
#include "WebAssemblyUtilities.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/CodeGen/MachineBlockFrequencyInfo.h"
#include "llvm/CodeGen/MachineFunctionAnalysisManager.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachinePassManager.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/IR/Analysis.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
using namespace llvm;

#define DEBUG_TYPE "wasm-explicit-locals"

namespace {
class WebAssemblyExplicitLocalsLegacy final : public MachineFunctionPass {
  StringRef getPassName() const override {
    return "WebAssembly Explicit Locals";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

public:
  static char ID; // Pass identification, replacement for typeid
  WebAssemblyExplicitLocalsLegacy() : MachineFunctionPass(ID) {}
};
} // end anonymous namespace

char WebAssemblyExplicitLocalsLegacy::ID = 0;
INITIALIZE_PASS(WebAssemblyExplicitLocalsLegacy, DEBUG_TYPE,
                "Convert registers to WebAssembly locals", false, false)

FunctionPass *llvm::createWebAssemblyExplicitLocalsLegacyPass() {
  return new WebAssemblyExplicitLocalsLegacy();
}

static void checkFrameBase(WebAssemblyFunctionInfo &MFI, unsigned Local,
                           unsigned Reg) {
  // Mark a local for the frame base vreg.
  if (MFI.isFrameBaseVirtual() && Reg == MFI.getFrameBaseVreg()) {
    LLVM_DEBUG({
      dbgs() << "Allocating local " << Local << "for VReg "
             << Register(Reg).virtRegIndex() << '\n';
    });
    MFI.setFrameBaseLocal(Local);
  }
}

/// Return a local id number for the given register, assigning it a new one
/// if it doesn't yet have one.
static unsigned getLocalId(DenseMap<unsigned, unsigned> &Reg2Local,
                           WebAssemblyFunctionInfo &MFI, unsigned &CurLocal,
                           unsigned Reg) {
  auto P = Reg2Local.insert(std::make_pair(Reg, CurLocal));
  if (P.second) {
    checkFrameBase(MFI, CurLocal, Reg);
    ++CurLocal;
  }
  return P.first->second;
}

/// Get the appropriate drop opcode for the given register class.
static unsigned getDropOpcode(const TargetRegisterClass *RC) {
  if (RC == &WebAssembly::I32RegClass)
    return WebAssembly::DROP_I32;
  if (RC == &WebAssembly::I64RegClass)
    return WebAssembly::DROP_I64;
  if (RC == &WebAssembly::F32RegClass)
    return WebAssembly::DROP_F32;
  if (RC == &WebAssembly::F64RegClass)
    return WebAssembly::DROP_F64;
  if (RC == &WebAssembly::V128RegClass)
    return WebAssembly::DROP_V128;
  if (RC == &WebAssembly::FUNCREFRegClass)
    return WebAssembly::DROP_FUNCREF;
  if (RC == &WebAssembly::EXTERNREFRegClass)
    return WebAssembly::DROP_EXTERNREF;
  if (RC == &WebAssembly::EXNREFRegClass)
    return WebAssembly::DROP_EXNREF;
  llvm_unreachable("Unexpected register class");
}

/// Get the appropriate local.get opcode for the given register class.
static unsigned getLocalGetOpcode(const TargetRegisterClass *RC) {
  if (RC == &WebAssembly::I32RegClass)
    return WebAssembly::LOCAL_GET_I32;
  if (RC == &WebAssembly::I64RegClass)
    return WebAssembly::LOCAL_GET_I64;
  if (RC == &WebAssembly::F32RegClass)
    return WebAssembly::LOCAL_GET_F32;
  if (RC == &WebAssembly::F64RegClass)
    return WebAssembly::LOCAL_GET_F64;
  if (RC == &WebAssembly::V128RegClass)
    return WebAssembly::LOCAL_GET_V128;
  if (RC == &WebAssembly::FUNCREFRegClass)
    return WebAssembly::LOCAL_GET_FUNCREF;
  if (RC == &WebAssembly::EXTERNREFRegClass)
    return WebAssembly::LOCAL_GET_EXTERNREF;
  if (RC == &WebAssembly::EXNREFRegClass)
    return WebAssembly::LOCAL_GET_EXNREF;
  llvm_unreachable("Unexpected register class");
}

/// Get the appropriate local.set opcode for the given register class.
static unsigned getLocalSetOpcode(const TargetRegisterClass *RC) {
  if (RC == &WebAssembly::I32RegClass)
    return WebAssembly::LOCAL_SET_I32;
  if (RC == &WebAssembly::I64RegClass)
    return WebAssembly::LOCAL_SET_I64;
  if (RC == &WebAssembly::F32RegClass)
    return WebAssembly::LOCAL_SET_F32;
  if (RC == &WebAssembly::F64RegClass)
    return WebAssembly::LOCAL_SET_F64;
  if (RC == &WebAssembly::V128RegClass)
    return WebAssembly::LOCAL_SET_V128;
  if (RC == &WebAssembly::FUNCREFRegClass)
    return WebAssembly::LOCAL_SET_FUNCREF;
  if (RC == &WebAssembly::EXTERNREFRegClass)
    return WebAssembly::LOCAL_SET_EXTERNREF;
  if (RC == &WebAssembly::EXNREFRegClass)
    return WebAssembly::LOCAL_SET_EXNREF;
  llvm_unreachable("Unexpected register class");
}

/// Get the appropriate local.tee opcode for the given register class.
static unsigned getLocalTeeOpcode(const TargetRegisterClass *RC) {
  if (RC == &WebAssembly::I32RegClass)
    return WebAssembly::LOCAL_TEE_I32;
  if (RC == &WebAssembly::I64RegClass)
    return WebAssembly::LOCAL_TEE_I64;
  if (RC == &WebAssembly::F32RegClass)
    return WebAssembly::LOCAL_TEE_F32;
  if (RC == &WebAssembly::F64RegClass)
    return WebAssembly::LOCAL_TEE_F64;
  if (RC == &WebAssembly::V128RegClass)
    return WebAssembly::LOCAL_TEE_V128;
  if (RC == &WebAssembly::FUNCREFRegClass)
    return WebAssembly::LOCAL_TEE_FUNCREF;
  if (RC == &WebAssembly::EXTERNREFRegClass)
    return WebAssembly::LOCAL_TEE_EXTERNREF;
  if (RC == &WebAssembly::EXNREFRegClass)
    return WebAssembly::LOCAL_TEE_EXNREF;
  llvm_unreachable("Unexpected register class");
}

/// Get the type associated with the given register class.
static MVT typeForRegClass(const TargetRegisterClass *RC) {
  if (RC == &WebAssembly::I32RegClass)
    return MVT::i32;
  if (RC == &WebAssembly::I64RegClass)
    return MVT::i64;
  if (RC == &WebAssembly::F32RegClass)
    return MVT::f32;
  if (RC == &WebAssembly::F64RegClass)
    return MVT::f64;
  if (RC == &WebAssembly::V128RegClass)
    return MVT::v16i8;
  if (RC == &WebAssembly::FUNCREFRegClass)
    return MVT::funcref;
  if (RC == &WebAssembly::EXTERNREFRegClass)
    return MVT::externref;
  if (RC == &WebAssembly::EXNREFRegClass)
    return MVT::exnref;
  llvm_unreachable("unrecognized register class");
}

/// Given a MachineOperand of a stackified vreg, return the instruction at the
/// start of the expression tree.
static MachineInstr *findStartOfTree(MachineOperand &MO,
                                     MachineRegisterInfo &MRI,
                                     const WebAssemblyFunctionInfo &MFI) {
  Register Reg = MO.getReg();
  assert(MFI.isVRegStackified(Reg));
  MachineInstr *Def = MRI.getVRegDef(Reg);

  // If this instruction has any non-stackified defs, it is the start
  for (auto DefReg : Def->defs()) {
    if (!MFI.isVRegStackified(DefReg.getReg())) {
      return Def;
    }
  }

  // Find the first stackified use and proceed from there.
  for (MachineOperand &DefMO : Def->explicit_uses()) {
    if (!DefMO.isReg())
      continue;
    return findStartOfTree(DefMO, MRI, MFI);
  }

  // If there were no stackified uses, we've reached the start.
  return Def;
}

// FAKE_USEs are no-ops, so remove them here so that the values used by them
// will be correctly dropped later.
static void removeFakeUses(MachineFunction &MF) {
  SmallVector<MachineInstr *> ToDelete;
  for (auto &MBB : MF)
    for (auto &MI : MBB)
      if (MI.isFakeUse())
        ToDelete.push_back(&MI);
  for (auto *MI : ToDelete)
    MI->eraseFromParent();
}

/// If MI writes a local, set Local to its index and ZeroDef to the CONST
/// instruction defining the value written if that value is a (bitwise) zero,
/// else nullptr.
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
  if (!Src.isReg() || !Src.getReg().isVirtual())
    return true;
  MachineInstr *D = MRI.getUniqueVRegDef(Src.getReg());
  if (!D)
    return true;
  switch (D->getOpcode()) {
  case WebAssembly::CONST_I32:
  case WebAssembly::CONST_I64:
    if (D->getOperand(1).isImm() && D->getOperand(1).getImm() == 0)
      ZeroDef = D;
    break;
  case WebAssembly::CONST_F32:
  case WebAssembly::CONST_F64:
    if (D->getOperand(1).isFPImm() &&
        D->getOperand(1).getFPImm()->getValueAPF().isPosZero())
      ZeroDef = D;
    break;
  }
  return true;
}

/// Wasm zero-initializes a function's non-parameter locals, so a write of zero
/// to a local that still holds zero on every path is a no-op. PHI elimination
/// emits such writes for incoming zero constants, e.g. a loop counter that
/// starts at 0. Remove them with a forward dataflow over local indices.
static bool removeZeroLocalSets(MachineFunction &MF) {
  const auto &MFI = *MF.getInfo<WebAssemblyFunctionInfo>();
  MachineRegisterInfo &MRI = MF.getRegInfo();
  unsigned NumParams = MFI.getParams().size();
  unsigned NumLocals = NumParams + MFI.getLocals().size();
  if (NumLocals == NumParams)
    return false;

  // Out[MBB]: locals known to hold zero on exit from MBB. Start optimistic and
  // iterate to the greatest fixed point.
  DenseMap<const MachineBasicBlock *, BitVector> In, Out;
  BitVector EntryIn(NumLocals, true);
  EntryIn.reset(0, NumParams);
  ReversePostOrderTraversal<MachineFunction *> RPOT(&MF);
  for (MachineBasicBlock *MBB : RPOT)
    Out[MBB] = BitVector(NumLocals, true);
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
  for (bool Iterate = true; Iterate;) {
    Iterate = false;
    for (MachineBasicBlock *MBB : RPOT) {
      BitVector Z = MBB == &MF.front() ? EntryIn : BitVector(NumLocals, true);
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
    // Blocks not reached in RPOT are unreachable; assume nothing.
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
      // A local.tee also leaves the value on the stack; keep it.
      if (!Z.test(L) || WebAssembly::isLocalTee(MI.getOpcode())) {
        Z.set(L);
        continue;
      }
      Register R = MI.getOperand(1).getReg();
      MI.eraseFromParent();
      if (MRI.use_nodbg_empty(R)) {
        SmallVector<MachineInstr *, 2> DbgUsers;
        for (MachineInstr &DbgMI : MRI.use_instructions(R))
          DbgUsers.push_back(&DbgMI);
        for (MachineInstr *DbgMI : DbgUsers)
          DbgMI->setDebugValueUndef();
        ZeroDef->eraseFromParent();
      }
      Changed = true;
    }
  }
  return Changed;
}

static bool explicitLocals(MachineFunction &MF) {
  LLVM_DEBUG(dbgs() << "********** Make Locals Explicit **********\n"
                       "********** Function: "
                    << MF.getName() << '\n');

  bool Changed = false;
  MachineRegisterInfo &MRI = MF.getRegInfo();
  WebAssemblyFunctionInfo &MFI = *MF.getInfo<WebAssemblyFunctionInfo>();
  const auto *TII = MF.getSubtarget<WebAssemblySubtarget>().getInstrInfo();

  removeFakeUses(MF);

  // Map non-stackified virtual registers to their local ids.
  DenseMap<unsigned, unsigned> Reg2Local;

  // Start assigning local numbers after the last parameter and after any
  // already-assigned locals.
  unsigned CurLocal = static_cast<unsigned>(MFI.getParams().size());
  CurLocal += static_cast<unsigned>(MFI.getLocals().size());

  // ABI contract: the async context must always be available in local 0. If
  // something else defines the register of the swiftasync argument too, for
  // example a value that register coalescing merged with it, that register
  // gets its own local.
  Register SwiftAsyncReg = WebAssembly::getSwiftAsyncArgReg(MF);
  bool CopySwiftAsyncArg = SwiftAsyncReg && !MRI.hasOneDef(SwiftAsyncReg);

  // Handle ARGUMENTS first to ensure that they get the designated numbers.
  for (MachineBasicBlock::iterator I = MF.begin()->begin(),
                                   E = MF.begin()->end();
       I != E;) {
    MachineInstr &MI = *I++;
    if (!WebAssembly::isArgument(MI.getOpcode()))
      break;
    Register Reg = MI.getOperand(0).getReg();
    assert(!MFI.isVRegStackified(Reg));
    auto Local = static_cast<unsigned>(MI.getOperand(1).getImm());
    if (Reg == SwiftAsyncReg && CopySwiftAsyncArg) {
      const TargetRegisterClass *RC = MRI.getRegClass(Reg);
      Register Tmp = MRI.createVirtualRegister(RC);
      BuildMI(*MI.getParent(), MI, DebugLoc(), TII->get(getLocalGetOpcode(RC)),
              Tmp)
          .addImm(Local);
      MFI.stackifyVReg(MRI, Tmp);
      Local = CurLocal++;
      BuildMI(*MI.getParent(), MI, DebugLoc(), TII->get(getLocalSetOpcode(RC)))
          .addImm(Local)
          .addReg(Tmp);
    }
    Reg2Local[Reg] = Local;
    checkFrameBase(MFI, Local, Reg);

    // Update debug value to point to the local before removing.
    WebAssemblyDebugValueManager(&MI).replaceWithLocal(Local);

    MI.eraseFromParent();
    Changed = true;
  }

  // Precompute the set of registers that are unused, so that we can insert
  // drops to their defs.
  // And unstackify any stackified registers that don't have any uses, so that
  // they can be dropped later. This can happen when transformations after
  // RegStackify remove instructions using stackified registers.
  BitVector UseEmpty(MRI.getNumVirtRegs());
  for (unsigned I = 0, E = MRI.getNumVirtRegs(); I < E; ++I) {
    Register Reg = Register::index2VirtReg(I);
    if (MRI.use_empty(Reg)) {
      UseEmpty[I] = true;
      MFI.unstackifyVReg(Reg);
    }
  }

  // Visit each instruction in the function.
  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &MI : llvm::make_early_inc_range(MBB)) {
      assert(!WebAssembly::isArgument(MI.getOpcode()));

      if (MI.isDebugInstr() || MI.isLabel())
        continue;

      if (MI.getOpcode() == WebAssembly::IMPLICIT_DEF) {
        MI.eraseFromParent();
        Changed = true;
        continue;
      }

      // Replace tee instructions with local.tee. The difference is that tee
      // instructions have two defs, while local.tee instructions have one def
      // and an index of a local to write to.
      //
      // - Before:
      // TeeReg, Reg = TEE DefReg
      // INST ..., TeeReg, ...
      // INST ..., Reg, ...
      // INST ..., Reg, ...
      // * DefReg: may or may not be stackified
      // * Reg: not stackified
      // * TeeReg: stackified
      //
      // - After (when DefReg was already stackified):
      // TeeReg = LOCAL_TEE LocalId1, DefReg
      // INST ..., TeeReg, ...
      // INST ..., Reg, ...
      // INST ..., Reg, ...
      // * Reg: mapped to LocalId1
      // * TeeReg: stackified
      //
      // - After (when DefReg was not already stackified):
      // NewReg = LOCAL_GET LocalId1
      // TeeReg = LOCAL_TEE LocalId2, NewReg
      // INST ..., TeeReg, ...
      // INST ..., Reg, ...
      // INST ..., Reg, ...
      // * DefReg: mapped to LocalId1
      // * Reg: mapped to LocalId2
      // * TeeReg: stackified
      if (WebAssembly::isTee(MI.getOpcode())) {
        assert(MFI.isVRegStackified(MI.getOperand(0).getReg()));
        assert(!MFI.isVRegStackified(MI.getOperand(1).getReg()));
        Register DefReg = MI.getOperand(2).getReg();
        const TargetRegisterClass *RC = MRI.getRegClass(DefReg);

        // Stackify the input if it isn't stackified yet.
        if (!MFI.isVRegStackified(DefReg)) {
          unsigned LocalId = getLocalId(Reg2Local, MFI, CurLocal, DefReg);
          Register NewReg = MRI.createVirtualRegister(RC);
          unsigned Opc = getLocalGetOpcode(RC);
          BuildMI(MBB, &MI, MI.getDebugLoc(), TII->get(Opc), NewReg)
              .addImm(LocalId);
          MI.getOperand(2).setReg(NewReg);
          MFI.stackifyVReg(MRI, NewReg);
        }

        // Replace the TEE with a LOCAL_TEE.
        unsigned LocalId =
            getLocalId(Reg2Local, MFI, CurLocal, MI.getOperand(1).getReg());
        unsigned Opc = getLocalTeeOpcode(RC);
        BuildMI(MBB, &MI, MI.getDebugLoc(), TII->get(Opc),
                MI.getOperand(0).getReg())
            .addImm(LocalId)
            .addReg(MI.getOperand(2).getReg());

        WebAssemblyDebugValueManager(&MI).replaceWithLocal(LocalId);

        MI.eraseFromParent();
        Changed = true;
        continue;
      }

      // Insert local.sets for any defs that aren't stackified yet.
      for (auto &Def : MI.defs()) {
        Register OldReg = Def.getReg();
        if (!MFI.isVRegStackified(OldReg)) {
          const TargetRegisterClass *RC = MRI.getRegClass(OldReg);
          Register NewReg = MRI.createVirtualRegister(RC);
          auto InsertPt = std::next(MI.getIterator());
          // When libcalls are emitted for thread context, the frame base vreg
          // has an implicit use in the DW_AT_frame_base debug info, so we
          // should not remove it.
          bool NeedsRegForDebug =
              MFI.isFrameBaseVirtual() && OldReg == MFI.getFrameBaseVreg() &&
              MF.getFunction().getSubprogram() &&
              MF.getSubtarget<WebAssemblySubtarget>().hasLibcallThreadContext();
          if (UseEmpty[OldReg.virtRegIndex()] && !NeedsRegForDebug) {
            unsigned Opc = getDropOpcode(RC);
            BuildMI(MBB, InsertPt, MI.getDebugLoc(), TII->get(Opc))
                .addReg(NewReg);
            if (MFI.isFrameBaseVirtual() && OldReg == MFI.getFrameBaseVreg())
              MFI.clearFrameBaseVreg();
          } else {
            unsigned LocalId = getLocalId(Reg2Local, MFI, CurLocal, OldReg);
            unsigned Opc = getLocalSetOpcode(RC);

            WebAssemblyDebugValueManager(&MI).replaceWithLocal(LocalId);

            BuildMI(MBB, InsertPt, MI.getDebugLoc(), TII->get(Opc))
                .addImm(LocalId)
                .addReg(NewReg);
          }
          // This register operand of the original instruction is now being used
          // by the inserted drop or local.set instruction, so make it not dead
          // yet.
          Def.setReg(NewReg);
          Def.setIsDead(false);
          MFI.stackifyVReg(MRI, NewReg);
          Changed = true;
        }
      }

      // Insert local.gets for any uses that aren't stackified yet.
      MachineInstr *InsertPt = &MI;
      for (MachineOperand &MO : reverse(MI.explicit_uses())) {
        if (!MO.isReg())
          continue;

        Register OldReg = MO.getReg();

        // Inline asm may have a def in the middle of the operands. Our contract
        // with inline asm register operands is to provide local indices as
        // immediates.
        if (MO.isDef()) {
          assert(MI.isInlineAsm());
          unsigned LocalId = getLocalId(Reg2Local, MFI, CurLocal, OldReg);
          // If this register operand is tied to another operand, we can't
          // change it to an immediate. Untie it first.
          MI.untieRegOperand(MO.getOperandNo());
          MO.ChangeToImmediate(LocalId);
          continue;
        }

        // If we see a stackified register, prepare to insert subsequent
        // local.gets before the start of its tree.
        if (MFI.isVRegStackified(OldReg)) {
          InsertPt = findStartOfTree(MO, MRI, MFI);
          continue;
        }

        // Our contract with inline asm register operands is to provide local
        // indices as immediates.
        if (MI.isInlineAsm()) {
          unsigned LocalId = getLocalId(Reg2Local, MFI, CurLocal, OldReg);
          // Untie it first if this reg operand is tied to another operand.
          MI.untieRegOperand(MO.getOperandNo());
          MO.ChangeToImmediate(LocalId);
          continue;
        }

        // Insert a local.get.
        unsigned LocalId = getLocalId(Reg2Local, MFI, CurLocal, OldReg);
        const TargetRegisterClass *RC = MRI.getRegClass(OldReg);
        Register NewReg = MRI.createVirtualRegister(RC);
        unsigned Opc = getLocalGetOpcode(RC);
        // Use a InsertPt as our DebugLoc, since MI may be discontinuous from
        // the where this local is being inserted, causing non-linear stepping
        // in the debugger or function entry points where variables aren't live
        // yet. Alternative is previous instruction, but that is strictly worse
        // since it can point at the previous statement.
        // See crbug.com/1251909, crbug.com/1249745
        InsertPt = BuildMI(MBB, InsertPt, InsertPt->getDebugLoc(),
                           TII->get(Opc), NewReg).addImm(LocalId);
        MO.setReg(NewReg);
        MFI.stackifyVReg(MRI, NewReg);
        Changed = true;
      }

      // Coalesce and eliminate COPY instructions.
      if (WebAssembly::isCopy(MI.getOpcode())) {
        MRI.replaceRegWith(MI.getOperand(1).getReg(),
                           MI.getOperand(0).getReg());
        MI.eraseFromParent();
        Changed = true;
      }
    }
  }

  // Define the locals.
  // TODO: Sort the locals for better compression.
  MFI.setNumLocals(CurLocal - MFI.getParams().size());
  for (unsigned I = 0, E = MRI.getNumVirtRegs(); I < E; ++I) {
    Register Reg = Register::index2VirtReg(I);
    auto RL = Reg2Local.find(Reg);
    if (RL == Reg2Local.end() || RL->second < MFI.getParams().size())
      continue;

    MFI.setLocal(RL->second - MFI.getParams().size(),
                 typeForRegClass(MRI.getRegClass(Reg)));
    Changed = true;
  }

  if (MF.getTarget().getOptLevel() != CodeGenOptLevel::None &&
      !MF.getFunction().hasOptNone())
    Changed |= removeZeroLocalSets(MF);

#ifndef NDEBUG
  // Assert that all registers have been stackified at this point.
  for (const MachineBasicBlock &MBB : MF) {
    for (const MachineInstr &MI : MBB) {
      if (MI.isDebugInstr() || MI.isLabel())
        continue;
      for (const MachineOperand &MO : MI.explicit_operands()) {
        assert(
            (!MO.isReg() || MRI.use_empty(MO.getReg()) ||
             MFI.isVRegStackified(MO.getReg())) &&
            "WebAssemblyExplicitLocals failed to stackify a register operand");
      }
    }
  }
#endif

  return Changed;
}

bool WebAssemblyExplicitLocalsLegacy::runOnMachineFunction(
    MachineFunction &MF) {
  return explicitLocals(MF);
}

PreservedAnalyses
WebAssemblyExplicitLocalsPass::run(MachineFunction &MF,
                                   MachineFunctionAnalysisManager &MFAM) {
  return explicitLocals(MF) ? getMachineFunctionPassPreservedAnalyses()
                                  .preserveSet<CFGAnalyses>()
                            : PreservedAnalyses::all();
}
