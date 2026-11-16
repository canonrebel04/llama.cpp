#!/usr/bin/env python3
"""Rewrite turbo-tan TQ3_4S tensors (stored as ggml type 46) to this fork's TQ3_4S id (51).

turbo-tan/llama.cpp-tq3 assigns id 46 to TQ3_4S; in this fork 46 is TQ4_1S and TQ3_4S is 51
(see docs/fork-types.md). Both use 32-value blocks but TQ3_4S blocks are 16 bytes and TQ4_1S
blocks are 20 bytes, so each tensor's on-disk span tells them apart. Only tensors whose span
matches TQ3_4S (and not TQ4_1S) are changed; the file is refused if any type-46 tensor looks
like a real TQ4_1S tensor or is ambiguous.

The type field is patched in place (4 bytes per tensor). Use --dry-run to only report, and
--backup to copy the file to <file>.bak first.
"""
from __future__ import annotations

import argparse
import os
import shutil
import struct
import sys

GGUF_MAGIC = b"GGUF"
TYPE_TQ4_1S_FORK = 46   # also turbo-tan's TQ3_4S id
TYPE_TQ3_4S_FORK = 51
QK = 32
BYTES_TQ3_4S = 16
BYTES_TQ4_1S = 20

# GGUF metadata value types -> fixed sizes (strings=8, arrays=9 handled separately)
_FIXED = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}


class _Reader:
    def __init__(self, f, endian: str):
        self.f = f
        self.e = endian

    def u32(self) -> int:
        return struct.unpack(self.e + "I", self.f.read(4))[0]

    def u64(self) -> int:
        return struct.unpack(self.e + "Q", self.f.read(8))[0]

    def string(self) -> bytes:
        n = self.u64()
        return self.f.read(n)

    def skip_value(self, vtype: int):
        if vtype in _FIXED:
            return self.f.read(_FIXED[vtype])
        if vtype == 8:
            return self.string()
        if vtype == 9:
            etype = self.u32()
            n = self.u64()
            if etype in _FIXED:
                self.f.seek(n * _FIXED[etype], os.SEEK_CUR)
            else:
                for _ in range(n):
                    self.skip_value(etype)
            return None
        raise ValueError(f"unknown GGUF value type {vtype}")


def pad(x: int, a: int) -> int:
    return (x + a - 1) // a * a


def scan(path: str):
    with open(path, "rb") as f:
        if f.read(4) != GGUF_MAGIC:
            raise ValueError("not a GGUF file")
        ver_le = struct.unpack("<I", f.read(4))[0]
        endian = "<" if ver_le < 65536 else ">"
        r = _Reader(f, endian)
        n_tensors = r.u64()
        n_kv = r.u64()
        alignment = 32
        for _ in range(n_kv):
            key = r.string()
            vtype = r.u32()
            val = r.skip_value(vtype)
            if key == b"general.alignment" and vtype == 4:
                alignment = struct.unpack(endian + "I", val)[0]
        tensors = []
        for _ in range(n_tensors):
            name = r.string().decode("utf-8", "replace")
            n_dims = r.u32()
            ne = [r.u64() for _ in range(n_dims)]
            type_pos = f.tell()
            ttype = r.u32()
            offset = r.u64()
            nel = 1
            for d in ne:
                nel *= d
            tensors.append(dict(name=name, type=ttype, type_pos=type_pos, offset=offset, nel=nel, ne0=ne[0] if ne else 1))
        data_start = pad(f.tell(), alignment)
        file_size = os.fstat(f.fileno()).st_size
    return endian, alignment, data_start, file_size, tensors


def classify(t, span: int, alignment: int, is_last: bool) -> str:
    if t["ne0"] % QK != 0:
        return "bad-shape"
    nblk = t["nel"] // QK
    s51, s46 = nblk * BYTES_TQ3_4S, nblk * BYTES_TQ4_1S
    if is_last:
        fits51 = s51 <= span < s46
        fits46 = span >= s46
    else:
        fits51 = span == pad(s51, alignment)
        fits46 = span == pad(s46, alignment)
    if fits51 and not fits46:
        return "tq3_4s"
    if fits46 and not fits51:
        return "tq4_1s"
    return "ambiguous"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", help="GGUF file to fix in place")
    ap.add_argument("--dry-run", action="store_true", help="only report what would change")
    ap.add_argument("--backup", action="store_true", help="copy the file to <model>.bak before patching")
    args = ap.parse_args()

    endian, alignment, data_start, file_size, tensors = scan(args.model)
    cand = [i for i, t in enumerate(tensors) if t["type"] == TYPE_TQ4_1S_FORK]
    if not cand:
        print("no type-46 tensors found, nothing to do")
        return 0

    order = sorted(range(len(tensors)), key=lambda i: tensors[i]["offset"])
    nxt = {order[k]: order[k + 1] for k in range(len(order) - 1)}
    fix, bad = [], []
    for i in cand:
        t = tensors[i]
        is_last = i not in nxt
        end = (data_start + tensors[nxt[i]]["offset"]) if not is_last else file_size
        span = end - (data_start + t["offset"])
        kind = classify(t, span, alignment, is_last)
        print(f"  {t['name']}: type 46, {t['nel']} elements, span {span} bytes -> {kind}")
        (fix if kind == "tq3_4s" else bad).append(i)

    if bad:
        print(f"refusing to patch: {len(bad)} type-46 tensor(s) are not unambiguously TQ3_4S", file=sys.stderr)
        return 1
    if args.dry_run:
        print(f"dry run: would retype {len(fix)} tensor(s) 46 -> {TYPE_TQ3_4S_FORK}")
        return 0
    if args.backup:
        shutil.copy2(args.model, args.model + ".bak")
        print(f"backup written to {args.model}.bak")
    with open(args.model, "r+b") as f:
        for i in fix:
            f.seek(tensors[i]["type_pos"])
            f.write(struct.pack(endian + "I", TYPE_TQ3_4S_FORK))
    print(f"retyped {len(fix)} tensor(s) 46 -> {TYPE_TQ3_4S_FORK}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
