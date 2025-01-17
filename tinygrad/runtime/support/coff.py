# tinygrad/runtime/support/coff.py

import struct
from dataclasses import dataclass
from typing import List, Tuple

##############################################################################
# Basic COFF structures and constants
##############################################################################

COFF_MACHINE_I386   = 0x14c
COFF_MACHINE_AMD64  = 0x8664
COFF_MACHINE_ARM64  = 0xaa64

# Minimal subset of AMD64 relocations in COFF
IMAGE_REL_AMD64_ADDR64    = 0x01  # 64-bit address
IMAGE_REL_AMD64_ADDR32    = 0x02  # 32-bit address
IMAGE_REL_AMD64_ADDR32NB  = 0x03  # 32-bit address, relative to image base
IMAGE_REL_AMD64_REL32     = 0x04  # 32-bit PC-relative

# Minimal subset of ARM64 relocations in COFF
IMAGE_REL_ARM64_BRANCH26        = 0x03  # used by B/BL
IMAGE_REL_ARM64_PAGEBASE_REL21  = 0x04  # like ELF's R_AARCH64_ADR_PREL_PG_HI21
IMAGE_REL_ARM64_REL21           = 0x05  # used by ADR (not ADRP)
IMAGE_REL_ARM64_PAGEOFFSET_12A  = 0x06  # like ELF's *_LO12_* relocations
IMAGE_REL_ARM64_PAGEOFFSET_12L  = 0x07  # often used by LDR literal

@dataclass
class CoffSection:
    name: str
    vaddr: int
    raw_data: bytes
    relocations: list  # list of (rva, sym_index, type)

@dataclass
class CoffSymbol:
    name: str
    value: int         # section-relative offset
    section_num: int   # 1-based section index, or special
    storage_class: int
    aux: bytes

##############################################################################
# Helpers for parsing COFF
##############################################################################

def _coff_str_from_bytes(b: bytes) -> str:
    """Utility to convert zero-terminated bytes or raw name to string."""
    return b.split(b'\x00', 1)[0].decode(errors='replace')

def _read_coff_header(blob: bytes, offset: int = 0):
    """
    Minimal read of IMAGE_FILE_HEADER (20 bytes):
      WORD  Machine;
      WORD  NumberOfSections;
      DWORD TimeDateStamp;
      DWORD PointerToSymbolTable;
      DWORD NumberOfSymbols;
      WORD  SizeOfOptionalHeader;
      WORD  Characteristics;
    """
    fmt = '<HHLLLHH'
    size = struct.calcsize(fmt)
    fields = struct.unpack(fmt, blob[offset:offset+size])
    return {
        'Machine': fields[0],
        'NumberOfSections': fields[1],
        'TimeDateStamp': fields[2],
        'PointerToSymbolTable': fields[3],
        'NumberOfSymbols': fields[4],
        'SizeOfOptionalHeader': fields[5],
        'Characteristics': fields[6],
    }, offset + size

def _read_section_headers(blob: bytes, offset: int, count: int) -> Tuple[List[CoffSection], int]:
    """
    Parse `count` IMAGE_SECTION_HEADER structures (each 40 bytes):
      BYTE  Name[8];
      union {
          DWORD PhysicalAddress;
          DWORD VirtualSize;
      };
      DWORD VirtualAddress;
      DWORD SizeOfRawData;
      DWORD PointerToRawData;
      DWORD PointerToRelocations;
      DWORD PointerToLinenumbers;
      WORD  NumberOfRelocations;
      WORD  NumberOfLinenumbers;
      DWORD Characteristics;
    """
    sections = []
    sh_size = 40
    for _ in range(count):
        hdr = blob[offset:offset+sh_size]
        name    = _coff_str_from_bytes(hdr[0:8])
        vsize   = struct.unpack('<I', hdr[8:12])[0]
        vaddr   = struct.unpack('<I', hdr[12:16])[0]
        rawsz   = struct.unpack('<I', hdr[16:20])[0]
        rawptr  = struct.unpack('<I', hdr[20:24])[0]
        relocptr= struct.unpack('<I', hdr[24:28])[0]
        n_reloc = struct.unpack('<H', hdr[32:34])[0]

        raw_data = blob[rawptr:rawptr+rawsz] if rawptr != 0 else b''

        # Read relocations: IMAGE_RELOCATION = 10 bytes each
        relocations = []
        if relocptr != 0 and n_reloc > 0:
            for r in range(n_reloc):
                reloc_data = blob[relocptr + 10*r : relocptr + 10*r + 10]
                va        = struct.unpack('<I', reloc_data[0:4])[0]  # VirtualAddress
                sym_index = struct.unpack('<I', reloc_data[4:8])[0]  # SymbolTableIndex
                rtype     = struct.unpack('<H', reloc_data[8:10])[0] # Type
                relocations.append((va, sym_index, rtype))

        sections.append(CoffSection(name, vaddr, raw_data, relocations))
        offset += sh_size
    return sections, offset

def _read_symbols(blob: bytes, symtab_offset: int, num_syms: int) -> List[CoffSymbol]:
    """
    Each COFF symbol is 18 bytes:
      union {
         BYTE   ShortName[8];
         struct {
            DWORD LongName[2];
         };
      } Name;
      DWORD  Value;
      SHORT  SectionNumber;
      WORD   Type;
      BYTE   StorageClass;
      BYTE   NumberOfAuxSymbols;

    Followed by AUX records (18 bytes each) if `NumberOfAuxSymbols` > 0.
    """
    syms = []
    off = symtab_offset
    i = 0
    while i < num_syms:
        chunk = blob[off:off+18]
        short_name = chunk[0:8]
        value      = struct.unpack('<I', chunk[8:12])[0]
        sec_num    = struct.unpack('<h', chunk[12:14])[0]
        # ignore 'Type' at 14..16
        storage_cls= chunk[16]
        aux_count  = chunk[17]

        sym_name = _coff_str_from_bytes(short_name)

        # read any AUX data
        aux_data = b''
        if aux_count > 0:
            aux_data = blob[off+18 : off+18+18*aux_count]

        syms.append(CoffSymbol(
            name = sym_name,
            value = value,
            section_num = sec_num,
            storage_class = storage_cls,
            aux = aux_data
        ))
        off += 18 + 18*aux_count
        i   += 1 + aux_count
    return syms

##############################################################################
# COFF Loader: build image + gather relocations
##############################################################################

def coff_loader(blob: bytes, force_section_align: int = 1):
    """
    Parse a COFF object file and produce:
      - memoryview(image)
      - list of CoffSection
      - list of relocations [(fixup_loc, target, rtype, addend)]
      - machine_type
    """
    # 1) File header
    hdr, off = _read_coff_header(blob, 0)
    num_sections = hdr['NumberOfSections']
    machine_type = hdr['Machine']

    # 2) Skip optional header (common in .exe/.dll, might be 0 in .obj)
    off += hdr['SizeOfOptionalHeader']

    # 3) Section headers
    sections, off = _read_section_headers(blob, off, num_sections)

    # 4) Symbol table
    symtab_offset = hdr['PointerToSymbolTable']
    num_syms      = hdr['NumberOfSymbols']
    symbols       = _read_symbols(blob, symtab_offset, num_syms)

    # 5) Build the image
    #    Reserve space for sections that have a nonzero VirtualAddress
    image_size = 0
    for s in sections:
        end = s.vaddr + len(s.raw_data)
        if end > image_size:
            image_size = end

    image = bytearray(image_size)

    # Place sections
    for s in sections:
        if s.vaddr != 0 and s.vaddr + len(s.raw_data) <= len(image):
            image[s.vaddr : s.vaddr + len(s.raw_data)] = s.raw_data
        else:
            # append at the end with alignment
            align = max(force_section_align, 1)
            pad_needed = (-len(image)) % align
            if pad_needed:
                image.extend(b'\0' * pad_needed)
            s.vaddr = len(image)
            image.extend(s.raw_data)

    # 6) Gather relocations as (fixup_loc, target_addr, type, addend)
    #    Basic COFF .obj doesn't store an explicit addend, so set 0
    relocs = []
    for sec in sections:
        for (rva, sym_idx, rtype) in sec.relocations:
            fixup_loc = sec.vaddr + rva
            sym = symbols[sym_idx]

            if 1 <= sym.section_num <= len(sections):
                tgt_section = sections[sym.section_num - 1]
                target = tgt_section.vaddr + sym.value
            else:
                # For extern/absolute, we just do sym.value here
                target = sym.value

            relocs.append((fixup_loc, target, rtype, 0))

    return memoryview(image), sections, relocs, machine_type

##############################################################################
# x64: Relocation patching
##############################################################################

def _relocate_coff_x64(instr: int, fixup_loc: int, target: int, r_type: int) -> int:
    """
    Minimal x86_64 COFF relocation logic, mirroring your ELF approach:
      - PC-relative 32 (REL32)
      - Absolute 32 (ADDR32)
      - Absolute 64 (ADDR64) -- partial if we only patch 4 bytes
    """
    if r_type == IMAGE_REL_AMD64_REL32:
        # 32-bit PC-relative => disp = target - (fixup_loc + 4)
        disp = (target - (fixup_loc + 4)) & 0xffffffff
        return disp
    elif r_type == IMAGE_REL_AMD64_ADDR32:
        # 32-bit absolute
        return target & 0xffffffff
    elif r_type == IMAGE_REL_AMD64_ADDR32NB:
        # 32-bit address, 'no base'
        return target & 0xffffffff
    elif r_type == IMAGE_REL_AMD64_ADDR64:
        # Actually needs 8-byte patch. We'll return the full 64 bits, but
        # if you're only writing 4 bytes back, you'll be storing the low half.
        return target
    else:
        raise NotImplementedError(f"Unknown x64 relocation type {r_type:#x}")

##############################################################################
# ARM64: Instruction decoding for LO12, etc.
##############################################################################

def _decode_arm64_load_store_shift(instr: int) -> int:
    """
    Attempt to decode how many bits to shift from 'target' when patching LO12.
    For standard LDR/STR:
      bits [31..30] = size => 0=>8-bit, 1=>16-bit, 2=>32-bit, 3=>64-bit
      bit [26] = V => if 1 => possibly 128-bit, shift=4
    This roughly mirrors your ELF code for R_AARCH64_LDST{16,32,64,128}.
    """
    v_bit = (instr >> 26) & 1
    if v_bit == 1:
        # naive assumption: 128-bit
        return 4
    else:
        size_field = (instr >> 30) & 0b11
        return size_field

def _relocate_coff_arm64(instr: int, fixup_loc: int, target: int, r_type: int) -> int:
    """
    ARM64 COFF relocation logic, similar to your ELF code.
    """
    if r_type == IMAGE_REL_ARM64_BRANCH26:
        # B/BL. PC-relative, 26-bit offset in [25..0]
        disp = (target - fixup_loc) >> 2
        disp_mask = disp & ((1 << 26) - 1)
        return (instr & 0xFC000000) | disp_mask

    elif r_type == IMAGE_REL_ARM64_PAGEBASE_REL21:
        # Like R_AARCH64_ADR_PREL_PG_HI21 (ADRP).
        #   offset = ((target & ~0xfff) - (fixup_loc & ~0xfff))
        rel_pg = ((target & ~0xfff) - (fixup_loc & ~0xfff))
        immhi = (rel_pg >> 14) & 0x7ffff
        immlo = (rel_pg >> 12) & 0x3
        instr &= 0x9F00001F  # clear relevant bits
        instr |= (immlo << 29)
        instr |= (immhi << 5)
        return instr

    elif r_type == IMAGE_REL_ARM64_REL21:
        # For ADR (non-ADRP), 21-bit PC-rel offset
        rel = (target - fixup_loc)
        immhi = (rel >> 2) & 0x7ffff
        immlo = rel & 0x3
        instr &= 0x9F00001F
        instr |= (immlo << 29)
        instr |= (immhi << 5)
        return instr

    elif r_type == IMAGE_REL_ARM64_PAGEOFFSET_12A:
        # Like various *_LO12_* relocations. We decode instruction size for shift.
        shift = _decode_arm64_load_store_shift(instr)
        lo12 = (target >> shift) & 0xfff
        instr_masked = instr & ~(0xfff << 10)
        instr = instr_masked | (lo12 << 10)
        return instr

    elif r_type == IMAGE_REL_ARM64_PAGEOFFSET_12L:
        # Similar to PAGEOFFSET_12A but might be used for LDR literal or other cases.
        shift = _decode_arm64_load_store_shift(instr)
        lo12 = (target >> shift) & 0xfff
        instr_masked = instr & ~(0xfff << 10)
        instr = instr_masked | (lo12 << 10)
        return instr

    else:
        raise NotImplementedError(f"Unknown ARM64 relocation type {r_type:#x}")

##############################################################################
# Main entry point: jit_loader_coff
##############################################################################

def jit_loader_coff(obj: bytes) -> bytes:
    """
    Build a runnable image from a Windows COFF .obj, applying relocations
    (x86_64 or ARM64) in a style similar to your ELF-based jit_loader.
    Returns the final bytes of the image.
    """
    image, sections, relocs, machine_type = coff_loader(obj)

    # Determine architecture
    if machine_type == COFF_MACHINE_AMD64:
        arch = 'x64'
    elif machine_type == COFF_MACHINE_ARM64:
        arch = 'arm64'
    else:
        raise NotImplementedError(f"Unsupported or unknown machine type {machine_type:#x}")

    # Apply relocations
    for (fixup_loc, target, rtype, addend) in relocs:
        if arch == 'x64':
            # For x64 relocations:
            #   - often 4-byte patches, but ADDR64 is actually 8 bytes.
            #   - For demonstration, we'll do a 4-byte read/write except for ADDR64 below.
            if rtype == IMAGE_REL_AMD64_ADDR64:
                # True 64-bit patch
                old_val = struct.unpack_from("<Q", image, fixup_loc)[0]
                new_val = _relocate_coff_x64(old_val & 0xffffffff, fixup_loc, target + addend, rtype)
                struct.pack_into("<Q", image, fixup_loc, new_val & 0xffffffffffffffff)
            else:
                instr = struct.unpack_from("<I", image, fixup_loc)[0]
                patched = _relocate_coff_x64(instr, fixup_loc, target + addend, rtype)
                struct.pack_into("<I", image, fixup_loc, patched & 0xffffffff)

        elif arch == 'arm64':
            # On ARM64, relocations typically patch a 4-byte instruction
            instr = struct.unpack_from("<I", image, fixup_loc)[0]
            patched = _relocate_coff_arm64(instr, fixup_loc, target + addend, rtype)
            struct.pack_into("<I", image, fixup_loc, patched & 0xffffffff)

    return bytes(image)