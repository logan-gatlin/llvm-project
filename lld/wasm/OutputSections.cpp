//===- OutputSections.cpp -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "OutputSections.h"
#include "InputChunks.h"
#include "InputElement.h"
#include "InputFiles.h"
#include "OutputSegment.h"
#include "WriterUtils.h"
#include "lld/Common/ErrorHandler.h"
#include "lld/Common/Memory.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/LEB128.h"

#define DEBUG_TYPE "lld"

using namespace llvm;
using namespace llvm::wasm;

namespace lld {

// Returns a string, e.g. "FUNCTION(.text)".
std::string toString(const wasm::OutputSection &sec) {
  if (!sec.name.empty())
    return (sec.getSectionName() + "(" + sec.name + ")").str();
  return std::string(sec.getSectionName());
}

namespace wasm {
StringRef OutputSection::getSectionName() const {
  return sectionTypeToString(type);
}

void OutputSection::createHeader(size_t bodySize) {
  raw_string_ostream os(header);
  debugWrite(os.tell(), "section type [" + getSectionName() + "]");
  encodeULEB128(type, os);
  writeUleb128(os, bodySize, "section size");
  log("createHeader: " + toString(*this) + " body=" + Twine(bodySize) +
      " total=" + Twine(getSize()));
}

void CodeSection::finalizeContents() {
  raw_string_ostream os(codeSectionHeader);
  writeUleb128(os, functions.size(), "function count");
  bodySize = codeSectionHeader.size();

  for (InputFunction *func : functions) {
    func->outputSec = this;
    func->outSecOff = bodySize;
    func->calculateSize();
    // All functions should have a non-empty body at this point
    assert(func->getSize());
    bodySize += func->getSize();
  }

  if (bodySize > UINT32_MAX) {
    error("section too large to encode: " + Twine(bodySize) + " bytes");
  }

  createHeader(bodySize);
}

void CodeSection::writeTo(uint8_t *buf) {
  log("writing " + toString(*this) + " offset=" + Twine(offset) +
      " size=" + Twine(getSize()));
  log(" headersize=" + Twine(header.size()));
  log(" codeheadersize=" + Twine(codeSectionHeader.size()));
  buf += offset;

  // Write section header
  memcpy(buf, header.data(), header.size());
  buf += header.size();

  // Write code section headers
  memcpy(buf, codeSectionHeader.data(), codeSectionHeader.size());

  // Write code section bodies
  for (const InputChunk *chunk : functions)
    chunk->writeTo(buf);
}

uint32_t CodeSection::getNumRelocations() const {
  uint32_t count = 0;
  for (const InputChunk *func : functions)
    count += func->getNumRelocations();
  return count;
}

void CodeSection::writeRelocations(raw_ostream &os) const {
  for (const InputChunk *c : functions)
    c->writeRelocations(os);
}

// Whether --pack-data-segments may drop the zero bytes of active segments:
// the memory must start zeroed (it is defined here, not imported) and the
// segments must be plain active segments at constant offsets.
static bool canPackDataSegments() {
  return ctx.arg.packDataSegments && !ctx.arg.relocatable &&
         !ctx.arg.emitRelocs && !ctx.isPic && !ctx.arg.memoryImport &&
         !ctx.arg.isMultithreaded();
}

static std::string segmentHeader(uint32_t initFlags, uint64_t va,
                                 uint32_t size) {
  std::string h;
  raw_string_ostream os(h);
  writeUleb128(os, initFlags, "init flags");
  if (initFlags & WASM_DATA_SEGMENT_HAS_MEMINDEX)
    writeUleb128(os, 0, "memory index");
  writeInitExpr(os, intConst(va, ctx.arg.is64.value_or(false)));
  writeUleb128(os, size, "segment size");
  return h;
}

// Render `segment`'s final bytes and split it into pieces that skip runs of
// zeros longer than the header a new piece costs. The engine limit on the
// number of data segments is respected by `budget`.
static void packSegment(OutputSegment *segment, unsigned &budget) {
  segment->packed.assign(segment->size, 0);
  for (InputChunk *inputSeg : segment->inputSegments) {
    inputSeg->outSecOff = inputSeg->outputSegmentOffset;
    inputSeg->writeTo(segment->packed.data());
  }
  const std::vector<uint8_t> &d = segment->packed;
  auto pieceCost = [&](uint32_t off, uint32_t size) {
    return segmentHeader(segment->initFlags, segment->startVA + off, size)
        .size();
  };
  uint32_t i = 0, n = d.size();
  while (i < n && d[i] == 0)
    ++i;
  while (n > i && d[n - 1] == 0)
    --n;
  uint32_t pieceStart = i;
  while (i < n) {
    if (d[i] != 0) {
      ++i;
      continue;
    }
    uint32_t j = i;
    while (j < n && d[j] == 0)
      ++j;
    // Splitting turns [pieceStart, j..) into two pieces; it pays off when
    // the zeros are longer than the extra header.
    if (budget > 1 && j - i > pieceCost(j, n - j) + 1) {
      segment->pieces.push_back({pieceStart, i - pieceStart, "", 0});
      --budget;
      pieceStart = j;
    }
    i = j;
  }
  if (n > pieceStart)
    segment->pieces.push_back({pieceStart, n - pieceStart, "", 0});
  for (OutputSegment::Piece &p : segment->pieces)
    p.header = segmentHeader(segment->initFlags, segment->startVA + p.offset,
                             p.size);
}

void DataSection::finalizeContents() {
  raw_string_ostream os(dataSectionHeader);
  bool pack = canPackDataSegments();
  if (pack) {
    // V8 accepts at most 100000 data segments.
    unsigned budget = 100000 - llvm::count_if(segments, [](OutputSegment *s) {
                        return s->requiredInBinary();
                      });
    for (OutputSegment *segment : segments)
      if (segment->requiredInBinary() && segment->isActive())
        packSegment(segment, budget);
  }
  unsigned segmentCount = 0;
  for (OutputSegment *segment : segments) {
    if (!segment->requiredInBinary())
      continue;
    segment->binaryIndex = segmentCount;
    segmentCount += segment->packed.empty() ? 1 : segment->pieces.size();
  }
#ifndef NDEBUG
  unsigned activeCount = llvm::count_if(segments, [](OutputSegment *segment) {
    return segment->requiredInBinary() && segment->isActive();
  });
#endif

  assert((!ctx.isPic || ctx.arg.extendedConst || activeCount <= 1) &&
         "output segments should have been combined by now");

  writeUleb128(os, segmentCount, "data count");
  bodySize = dataSectionHeader.size();
  bool is64 = ctx.arg.is64.value_or(false);

  for (OutputSegment *segment : segments) {
    if (!segment->requiredInBinary())
      continue;
    if (!segment->packed.empty()) {
      for (OutputSegment::Piece &p : segment->pieces) {
        p.sectionOffset = bodySize;
        bodySize += p.header.size() + p.size;
      }
      log("Data segment: size=" + Twine(segment->size) + ", startVA=" +
          Twine::utohexstr(segment->startVA) + ", name=" + segment->name +
          ", packed into " + Twine(segment->pieces.size()) + " pieces");
      continue;
    }
    raw_string_ostream os(segment->header);
    writeUleb128(os, segment->initFlags, "init flags");
    if (segment->initFlags & WASM_DATA_SEGMENT_HAS_MEMINDEX)
      writeUleb128(os, 0, "memory index");
    if (segment->isActive()) {
      if (ctx.isPic && ctx.arg.extendedConst) {
        writeU8(os, WASM_OPCODE_GLOBAL_GET, "global get");
        writeUleb128(os, ctx.sym.memoryBase->getGlobalIndex(),
                     "literal (global index)");
        if (segment->startVA) {
          writePtrConst(os, segment->startVA, is64, "offset");
          writeU8(os, is64 ? WASM_OPCODE_I64_ADD : WASM_OPCODE_I32_ADD, "add");
        }
        writeU8(os, WASM_OPCODE_END, "opcode:end");
      } else {
        WasmInitExpr initExpr;
        initExpr.Extended = false;
        if (ctx.isPic) {
          assert(segment->startVA == 0);
          initExpr.Inst.Opcode = WASM_OPCODE_GLOBAL_GET;
          initExpr.Inst.Value.Global = ctx.sym.memoryBase->getGlobalIndex();
        } else {
          initExpr = intConst(segment->startVA, is64);
        }
        writeInitExpr(os, initExpr);
      }
    }
    writeUleb128(os, segment->size, "segment size");

    segment->sectionOffset = bodySize;
    bodySize += segment->header.size() + segment->size;
    log("Data segment: size=" + Twine(segment->size) + ", startVA=" +
        Twine::utohexstr(segment->startVA) + ", name=" + segment->name);

    for (InputChunk *inputSeg : segment->inputSegments) {
      inputSeg->outputSec = this;
      inputSeg->outSecOff = segment->sectionOffset + segment->header.size() +
                            inputSeg->outputSegmentOffset;
    }
  }

  if (bodySize > UINT32_MAX) {
    error("section too large to encode: " + Twine(bodySize) + " bytes");
  }

  createHeader(bodySize);
}

void DataSection::writeTo(uint8_t *buf) {
  log("writing " + toString(*this) + " offset=" + Twine(offset) +
      " size=" + Twine(getSize()) + " body=" + Twine(bodySize));
  buf += offset;

  // Write section header
  memcpy(buf, header.data(), header.size());
  buf += header.size();

  // Write data section headers
  memcpy(buf, dataSectionHeader.data(), dataSectionHeader.size());

  for (const OutputSegment *segment : segments) {
    if (!segment->requiredInBinary())
      continue;
    if (!segment->packed.empty()) {
      for (const OutputSegment::Piece &p : segment->pieces) {
        uint8_t *at = buf + p.sectionOffset;
        memcpy(at, p.header.data(), p.header.size());
        memcpy(at + p.header.size(), segment->packed.data() + p.offset, p.size);
      }
      continue;
    }
    // Write data segment header
    uint8_t *segStart = buf + segment->sectionOffset;
    memcpy(segStart, segment->header.data(), segment->header.size());

    // Write segment data payload
    for (const InputChunk *chunk : segment->inputSegments)
      chunk->writeTo(buf);
  }
}
uint32_t DataSection::getNumRelocations() const {
  uint32_t count = 0;
  for (const OutputSegment *seg : segments)
    for (const InputChunk *inputSeg : seg->inputSegments)
      count += inputSeg->getNumRelocations();
  return count;
}

void DataSection::writeRelocations(raw_ostream &os) const {
  for (const OutputSegment *seg : segments)
    for (const InputChunk *c : seg->inputSegments)
      c->writeRelocations(os);
}

bool DataSection::isNeeded() const {
  for (const OutputSegment *seg : segments)
    if (seg->requiredInBinary())
      return true;
  return false;
}

// Lots of duplication here with OutputSegment::finalizeInputSegments
void CustomSection::finalizeInputSections() {
  SyntheticMergedChunk *mergedSection = nullptr;
  std::vector<InputChunk *> newSections;

  for (InputChunk *s : inputSections) {
    s->outputSec = this;
    MergeInputChunk *ms = dyn_cast<MergeInputChunk>(s);
    if (!ms) {
      newSections.push_back(s);
      continue;
    }

    if (!mergedSection) {
      mergedSection =
          make<SyntheticMergedChunk>(name, 0, WASM_SEG_FLAG_STRINGS);
      newSections.push_back(mergedSection);
      mergedSection->outputSec = this;
    }
    mergedSection->addMergeChunk(ms);
  }

  if (!mergedSection)
    return;

  mergedSection->finalizeContents();
  inputSections = std::move(newSections);
}

void CustomSection::finalizeContents() {
  finalizeInputSections();

  raw_string_ostream os(nameData);
  encodeULEB128(name.size(), os);
  os << name;

  for (InputChunk *section : inputSections) {
    assert(!section->discarded);
    payloadSize = alignTo(payloadSize, section->alignment);
    section->outSecOff = payloadSize;
    payloadSize += section->getSize();
  }

  if (payloadSize > UINT32_MAX) {
    error("section '" + name + "' too large to encode: " + Twine(payloadSize) +
          " bytes");
  }

  createHeader(payloadSize + nameData.size());
}

void CustomSection::writeTo(uint8_t *buf) {
  log("writing " + toString(*this) + " offset=" + Twine(offset) +
      " size=" + Twine(getSize()) + " chunks=" + Twine(inputSections.size()));

  assert(offset);
  buf += offset;

  // Write section header
  memcpy(buf, header.data(), header.size());
  buf += header.size();
  memcpy(buf, nameData.data(), nameData.size());
  buf += nameData.size();

  // Write custom sections payload
  for (const InputChunk *section : inputSections)
    section->writeTo(buf);
}

uint32_t CustomSection::getNumRelocations() const {
  uint32_t count = 0;
  for (const InputChunk *inputSect : inputSections)
    count += inputSect->getNumLiveRelocations();
  return count;
}

void CustomSection::writeRelocations(raw_ostream &os) const {
  for (const InputChunk *s : inputSections)
    s->writeRelocations(os);
}

} // namespace wasm
} // namespace lld
