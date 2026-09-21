# Running Qwen3.8-Flash-Next (qwen4exp) on consumer hardware

Measured deployment notes for the `qwen4exp` architecture
(Qwen3.8-Flash-Next, 125B MoE / 6B active + 51B n-gram PLE table), from a
real deployment on an RTX 3060 12GB / 31GB RAM / NVMe box. Every claim below
was measured, not estimated.

## Required flags

- `-fit off` — the automatic parameter fit mis-sizes `qwen4exp` and fails
  to allocate. Always set ctx and `-ngl` explicitly.
- `-ot per_layer_token_embd=CPU` — on CUDA, with `-ngl` high enough to cover
  layer 2 (where the PLE table lives), the engine issues one huge
  `cudaMalloc` for the whole n-gram table (35.8 GB on the Q4 build) and
  dies with OOM. On Metal the table self-hosts; on CUDA it needs the
  explicit override. Never use `--no-mmap` with this architecture.

## The PLE n-gram table and prefill

`per_layer_token_embd.weight` (~27-38 GB depending on quant) is read through
mmap, one page fault per token row. Whether those pages sit in the OS page
cache decides prefill speed: cold prompts measured 3-7 tok/s, warm 47-60
tok/s on the same server. This is the "prompt speed swings 3x" effect.

## Expert placement: what actually wins

All figures measured on one RTX 3060 12GB with a REAP-320 pruned
512->320-expert Q2 build (22.9 GB expert set) and 31 GB system RAM:

| configuration | decode | prefill | context |
| --- | --- | --- | --- |
| all experts CPU, no cache | 9.95 | 43 | 131k |
| static `--moe-cache-slots 32` | 9.6-12.4 | 43-95 | 32k-131k |
| **LRU `--moe-expert-cache-size 64` + `--moe-expert-cache-host-pinned-mb 8192`** | **12.4** | **95.5** | 32k |
| LRU cache 96 or 128 slabs | cudaMalloc OOM | - | - |
| LRU 64 slabs at 48k+ context | cudaMalloc OOM | - | - |

Rules of thumb:

- Do not combine the LRU cache with `--cpu-moe`/`--n-cpu-moe` tensor
  overrides for expert tensors; the cache owns expert placement. (Mixing
  them aborted at load in our tests.) The PLE `-ot` override above is fine -
  the table is not an expert tensor.
- The LRU cache and large context do not mix on a 12 GB card: 32k context
  is the practical ceiling at 64 slabs. Without the cache, 128k works.
- Thread count: 8 (physical cores) is optimal; `-t 16` measured 8x SLOWER
  decode (0.83 vs 6.79 tok/s) on the same machine.

## TurboQuant KV is not usable for qwen4exp

`-ctk turbo4 -ctv turbo4` and `turbo2` abort at load:

```
ggml-cpu/ops.cpp:5238: unsupported GET_ROWS source CPU#cache_idx_k_l3 (view)
```

The QSA indexer side-cache is a view tensor with no turbo GET_ROWS path.
Use standard cache types (`f16`, `q8_0`).

## MTP on a pruned trunk

The unsloth shared-Q8_0 MTP head loads and drafts on the REAP-320 build
(55% acceptance at n-max 4), but measured SLOWER than no speculation at all
(7.5 vs 9.95 tok/s at 64k ctx) because the draft runs on the same saturated
CPU. It also costs ~3.2 GB VRAM, which does not fit next to a 128k KV cache.
Skip it on CPU-bound expert deployments; it belongs on VRAM-resident setups.

## What pruning buys

REAP-320 (512->320 experts, multi-domain calibration, publisher AnonimousA)
cuts the expert set 44.5 GB -> 26.5 GB (Q3) / 22.9 GB (Q2). Decode gains come
from the expert set fitting the page cache, not from cheaper math: with all
experts host-side, decode is CPU-compute bound (~6B active params) and
prefill is page-fault bound. Measured on this box: unpruned 5.9 tok/s ->
pruned Q3 6.5 -> pruned Q2 with LRU cache 12.4, with prefill 3-7 -> 43-95
tok/s. The PLE table and expert set together decide everything.

## Density vs sparsity on small cards

For comparison, a dense 27B (`qwen3_5`, unsloth/Qwen3.8-27B-GGUF) at
UD-Q2_K_XL (9.8 GB) fits fully VRAM-resident on 12 GB and measured 15.8 tok/s
decode / 401 tok/s prefill with a draft head at 16k context - 4-13x faster
prefill than the MoE builds because nothing streams. Choose density for
interactive loops, sparsity for context and peak knowledge.
