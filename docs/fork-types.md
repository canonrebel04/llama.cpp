# Fork ggml_type ID registry

Every custom `ggml_type` in this fork must be listed here **before** it is added to
`ggml/include/ggml.h`, `gguf-py/gguf/constants.py` and `include/llama.h` (ftype).
IDs are stored in GGUF files: a GGUF written by a fork that uses different IDs
for the same type is **not** loadable here without remapping.

| ID | ggml_type | Origin | Block (elems / bytes) | Notes |
|----|-----------|--------|-----------------------|-------|
| 0-39 | upstream types | ggml-org/llama.cpp | | |
| 40 | NVFP4 | upstream | | |
| 41 | Q1_0 | upstream | | |
| 42 | Q2_0 | upstream | | |
| 43 | TURBO2_0 | TheTom turboquant | 32 / 10 | KV cache |
| 44 | TURBO3_0 | TheTom turboquant | | KV cache |
| 45 | TQ3_1S | TheTom/turbo-tan | 32 / 16 | weights |
| 46 | TQ4_1S | TheTom/turbo-tan | 32 / 20 | weights |
| 47 | TURBO4_0 | TheTom turboquant | | KV cache |
| 48 | Q8_CR | ConvRot | | weights |
| 49 | Q5_CR | ConvRot | | weights |
| 50 | Q6_CR | ConvRot | | weights |
| 51 | TQ3_4S | turbo-tan/llama.cpp-tq3 | 32 / 16 | weights; **turbo-tan uses ID 46** (collides with our TQ4_1S) |
| 52 | Q6_0 | Anbeeld/beellama.cpp | 32 / 26 | KV cache; beellama ID 43 |
| 53 | Q6_1 | Anbeeld/beellama.cpp | 32 / 28 | KV cache; beellama ID 44 |
| 54 | Q3_0 | Anbeeld/beellama.cpp | 32 / 14 | KV cache; beellama ID 45 |
| 55 | Q3_1 | Anbeeld/beellama.cpp | 32 / 16 | KV cache; beellama ID 46 |
| 56 | Q2_0S | Anbeeld/beellama.cpp | 32 / 10 | KV cache (beellama's "q2_0" for KV; upstream Q2_0=42 is a different format); beellama ID 47 |
| 57 | Q2_1 | Anbeeld/beellama.cpp | 32 / 12 | KV cache; beellama ID 48 |

| 137 | IQ2_K | ikawrakow/ik_llama.cpp | 256 / 76 | weights, 2.375 bpw; **same ID as ik_llama.cpp** |
| 138 | IQ3_K | ikawrakow/ik_llama.cpp | 256 / 110 | weights, 3.44 bpw; **same ID as ik_llama.cpp** |
| 139 | IQ4_K | ikawrakow/ik_llama.cpp | 256 / 144 | weights, 4.5 bpw; **same ID as ik_llama.cpp** |
| 140 | IQ5_K | ikawrakow/ik_llama.cpp | 256 / 176 | weights, 5.5 bpw; **same ID as ik_llama.cpp** |
| 141 | IQ6_K | ikawrakow/ik_llama.cpp | 256 / 212 | weights, 6.625 bpw; **same ID as ik_llama.cpp** |
| 142 | PQ2_0 | PrismML llama.cpp fork | 128 / 34 | weights; **same ID as PrismML** so their GGUFs load unchanged |
| 143 | PTQ1_0 | PrismML llama.cpp fork | 128 / 28 | weights (ternary, 1.75 bpw); **same ID as PrismML** |
| 144 | IQ4_KS | ikawrakow/ik_llama.cpp | 256 / 136 + 4 B/row | weights, 4.25 bpw, row-scaled; **same ID as ik_llama.cpp** |
| 145 | IQ2_KS | ikawrakow/ik_llama.cpp | 256 / 70 + 2 B/row | weights, 2.1875 bpw, row-scaled; **same ID** |
| 146 | IQ4_KSS | ikawrakow/ik_llama.cpp | 256 / 128 + 4 B/row | weights, 4.0 bpw, row-scaled; **same ID** |
| 152 | IQ5_KS | ikawrakow/ik_llama.cpp | 256 / 168 + 4 B/row | weights, 5.25 bpw, row-scaled; **same ID** |
| 156 | IQ3_KS | ikawrakow/ik_llama.cpp | 256 / 102 + 2 B/row | weights, 3.1875 bpw, row-scaled; **same ID** |
| 157 | IQ2_KL | ikawrakow/ik_llama.cpp | 256 / 86 + 2 B/row | weights, 2.6875 bpw, row-scaled; **same ID** |

Next free ID: **58** (58-136 free; 137-141 taken by ik_llama.cpp IQ_K types; 142-143 by Prism types; 144-146/152/156/157 by ik row-scaled types; 147-151/153-155/158 reserved for the remaining ik types (IQ*_KT etc.); GGML_TYPE_COUNT is 158). Reserve ranges per branch below and keep them in sync.

## llama_ftype additions

| ftype | value |
|-------|-------|
| MOSTLY_Q8_CR | 42 |
| MOSTLY_TQ3_1S | 43 |
| MOSTLY_TQ4_1S | 44 |
| MOSTLY_Q5_CR | 45 |
| MOSTLY_Q6_CR | 46 |
| MOSTLY_TQ3_4S | 47 |
| MOSTLY_PQ2_0 | 141 (PrismML value) |
| MOSTLY_PQ2_0_LEGACY | 142 (older PrismML value, same format) |
| MOSTLY_PTQ1_0 | 143 (PrismML value) |
| MOSTLY_IQ2_K .. MOSTLY_IQ6_K | 148-152 (ik_llama.cpp uses 138-142, which collide with the Prism values; ftype is informational only) |
| MOSTLY_IQ4_KS, IQ2_KS, IQ4_KSS, IQ5_KS, IQ3_KS, IQ2_KL | 153-158 (ik values 137-146 collide with ours/Prism) |

## GGUF compatibility / remapping

- turbo-tan TQ3_4S GGUFs store tensor type 46 (our TQ4_1S). Two ways to handle them:
  1. **Fix the file (recommended):** `python gguf-py/gguf/scripts/gguf_fix_tq3_4s.py model.gguf [--dry-run] [--backup]`
     patches the 4-byte type field 46 -> 51 in place. It checks each type-46 tensor's on-disk
     span (offset to the next tensor, or to EOF for the last one) against the TQ3_4S size
     (16 B / 32 values) and the TQ4_1S size (20 B / 32 values), and refuses to touch the file if any
     type-46 tensor is a real TQ4_1S tensor or ambiguous.
  2. **Loader auto-detection:** `gguf_init` compares the stored offsets of type-46 tensors with both
     sizes. If at least one tensor only fits TQ3_4S and none only fits TQ4_1S, every type-46 tensor
     in the file is treated as TQ3_4S and a warning is logged. It cannot decide from a single type-46
     tensor that is the last tensor in the file (no following offset); fix such files with the script.

- beellama KV types are runtime KV-cache types only (never written into model GGUFs
  by this fork), so the ID difference vs beellama only matters for saved session/state files.

## ik_llama.cpp IQ_K types (forks/ik-quants)

Ported from [ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp) (Iwan Kawrakow, MIT);
croco.cpp carries the same code and was used as a cross-reference.

Scope (kept small on purpose):

- IQ2_K, IQ3_K, IQ4_K, IQ5_K, IQ6_K: block layouts, value tables, imatrix-aware
  quantization, dequantization (`ggml/src/ggml-iqk-quants.cpp`, part of ggml-base) and
  `llama-quantize` ftypes.
- CPU: scalar `vec_dot_*_q8_k` against Q8_K activations. IQ4_K/IQ5_K use ik's scalar
  dot products. IQ2_K/IQ3_K/IQ6_K only have SIMD (`iqk_mul_mat`) dot products in ik, so they
  use a dequantize-and-dot fallback here. Correct but slow: the iqk_mul_mat GEMM kernels are **not** ported.
- CUDA: dequantize (to F16/F32, contiguous and non-contiguous) and GET_ROWS. MUL_MAT goes
  through dequantize + cuBLAS. No MMQ/MMVQ kernels, and MUL_MAT_ID is reported unsupported,
  so MoE experts in these types run on the CPU.
- Not ported: IQ*_KS/KSS/KL, the trellis IQ*_KT types, the _r4/_r8 interleaved
  repacks, Vulkan/Metal kernels.
- GGUFs made by ik_llama.cpp with these five types load unchanged (same type IDs).

## Row-scaled types (row metadata)

IQ4_KS, IQ2_KS, IQ4_KSS, IQ5_KS, IQ3_KS and IQ2_KL store a per-row scale (`float` or `half`,
`ggml_type_traits.row_meta_size` bytes) in front of each row's blocks, exactly like ik_llama.cpp.
`ggml_row_size()` / `nb[1]` include it, so `to_float` / `from_float` / `vec_dot` must be called on
whole rows (`ggml_row_size(type, ne)` with `ne` > one row is not meaningful for these types).
CPU only (scalar; dequant fallback dot for IQ2_KL / IQ3_KS / IQ4_KSS); no GPU backend support yet.
