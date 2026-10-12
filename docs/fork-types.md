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

Next free ID: **52**. Reserve ranges per branch below and keep them in sync.

## llama_ftype additions

| ftype | value |
|-------|-------|
| MOSTLY_Q8_CR | 42 |
| MOSTLY_TQ3_1S | 43 |
| MOSTLY_TQ4_1S | 44 |
| MOSTLY_Q5_CR | 45 |
| MOSTLY_Q6_CR | 46 |
| MOSTLY_TQ3_4S | 47 |

## GGUF compatibility / remapping

- turbo-tan TQ3_4S GGUFs store tensor type 46. Loading them here would misread
  them as TQ4_1S (different block size; the loader rejects them on size mismatch).
  Remap by rewriting the tensor-info type field 46 -> 51 (for example with a small
  gguf-py script that copies tensors unchanged and sets `GGMLQuantizationType.TQ3_4S`).
