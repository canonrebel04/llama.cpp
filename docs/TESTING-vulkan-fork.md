# Fork test plan: Vulkan fixes (AMD), plus optional CUDA and Metal

> **Start with [TESTING.md](TESTING.md).** It's the single guide for the whole #62–#88 stack: which PRs need
> your AMD GPU, the test order, and how to bisect. This file holds the model list, reference numbers and known issues it refers to.

This is for testing the fork's stacked PRs #85, #86 and #87 on real hardware.
The branch `sync-upstream-20261102` (PR #87) contains everything from #62 through #87.

The build box only has lavapipe (a CPU Vulkan driver), so none of this has run on a real GPU yet.
Expected values below come from lavapipe `test-backend-ops` runs and from CPU-backend perplexity
runs on the same branch on 2026-11-02.

## 1. Get the branch

```sh
git clone https://github.com/canonrebel04/llama.cpp.git   # or: cd llama.cpp && git fetch origin
cd llama.cpp
git fetch origin
git checkout sync-upstream-20261102
git pull --ff-only
git log -1 --oneline          # should be the newest commit on that branch
```

## 2. Build

Vulkan (main target). You need the Vulkan SDK or distro packages: `libvulkan-dev`, `glslc`, `spirv-headers`.

```sh
cmake -B build-vk -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-vk -j"$(nproc)"
./build-vk/bin/llama-cli --list-devices     # your AMD GPU should be listed as Vulkan0
```

CPU-only reference build (used to compare against):

```sh
cmake -B build-cpu -DCMAKE_BUILD_TYPE=Release
cmake --build build-cpu -j"$(nproc)"
```

Optional, CUDA (it has only been compile-checked, never run):

```sh
cmake -B build-cuda -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=native -DCMAKE_BUILD_TYPE=Release
cmake --build build-cuda -j"$(nproc)"
```

Optional, Metal on a Mac (Metal is on by default; it has never been built for this fork):

```sh
cmake -B build-mtl -DCMAKE_BUILD_TYPE=Release
cmake --build build-mtl -j"$(sysctl -n hw.ncpu)"
```

For CUDA or Metal, run the same commands as below with `build-cuda` or `build-mtl`, and use `-b CUDA0` or `-b MTL0`.

If you have more than one Vulkan device, an iGPU and a dGPU for example, choose one with
`GGML_VK_VISIBLE_DEVICES=<index>`.

## 3. Op tests (test-backend-ops, GPU vs CPU)

Each command compares the GPU result against the CPU backend.

**Expected result for every command:** the last line is `OK` and `N/N tests passed`, with no `FAIL`.
Lines that say `not supported` are fine; that op falls back to the CPU.

```sh
B=./build-vk/bin/test-backend-ops
```

### 3a. TQ3_1S / TQ4_1S MUL_MAT through the dequant path (fix 35b3c10bc, PR #85)

```sh
$B -b Vulkan0 -o MUL_MAT    -p 'type_a=tq3_1s'
$B -b Vulkan0 -o MUL_MAT    -p 'type_a=tq4_1s'
$B -b Vulkan0 -o MUL_MAT_ID -p 'type_a=tq(3|4)_1s'
```

Before the fix, about 41 of these failed for n>8 with large NMSE. Expected now: all OK.

### 3b. turbo / TQ4_1S SET_ROWS (fix 308b9e035, PR #85)

```sh
$B -b Vulkan0 -o SET_ROWS
$B -b Vulkan0 -o SET_ROWS_TURBO3,SET_ROWS_TURBO4,SET_ROWS_TQ4_1S
```

Expected: all OK. On AMD RDNA, where 32-wide subgroups are available, the turbo and TQ4_1S cases
should actually run and report OK. On lavapipe they show `not supported`. Before the fix, the
process aborted in `ggml_vk_create_pipeline_func`. An abort or crash here is a failure.

### 3c. MUL_MAT_ID with skipped experts (id = -1), hot/cold MoE packs (fix 6b5c893d5, PR #85)

```sh
$B -b Vulkan0 -o MUL_MAT_ID -p 'skip=1'          # n=1, 4 and 64 for q4_0, q8_0, f16, f32
$B -b Vulkan0 -o MUL_MAT_ID -p 'n=64,.*skip=1'   # just the n=64 (mmq / matmul) cases
$B -b Vulkan0 -o MUL_MAT_ID                      # everything (lavapipe: 1238/1238)
```

Expected: all OK. Before the fix, skipped slots kept stale data, which showed up as large NMSE.

### 3d. FLASH_ATTN_EXT with the fork KV types (turbo #66/#84, beellama #80)

```sh
$B -b Vulkan0 -o FLASH_ATTN_EXT -p 'type_K=(turbo3|turbo4),'
$B -b Vulkan0 -o FLASH_ATTN_EXT -p 'type_K=(q6_0|q6_1|q3_0|q3_1|q2_0s|q2_1),'
$B -b Vulkan0 -o FLASH_ATTN_EXT -p 'type_K=(f16|q8_0),'                    # upstream baseline
$B -b Vulkan0 -o FLASH_ATTN_EXT -p 'turbo4_vec'                            # turbo4 decode vec path
```

Expected: all OK. Results on lavapipe:
- turbo3/turbo4: 752/752 each.
- Each beellama type: 976/976.
- q8_0: 996/996.
- f16: 1004/1004.

These runs are slow. Add `-j 4` to run several cases at once.

turbo2 and tq4_1s KV have no FLASH_ATTN_EXT cases in test-backend-ops. Section 4 covers them
with real models instead.

### 3e. Whole-suite sweep (optional, long)

```sh
$B -b Vulkan0 2>&1 | tee tbo-vulkan-full.log
grep -E 'FAIL|not supported' tbo-vulkan-full.log | grep -v 'not supported' | head -50
```

## 4. Real-model runs

These models were checked to exist on Hugging Face on 2026-11-02. All of them use head_dim 128,
which the turbo types need.

| Role | HF repo | File | Size |
|---|---|---|---|
| Dense, small | `ggml-org/Qwen3-0.6B-GGUF` | `Qwen3-0.6B-Q8_0.gguf` | 0.8 GB |
| Dense, mid | `Qwen/Qwen3-8B-GGUF` | `Qwen3-8B-Q4_K_M.gguf` | ~5 GB |
| Source for TQ weight quants | `unsloth/Qwen3-1.7B-GGUF` | `Qwen3-1.7B-BF16.gguf` | 3.4 GB |
| MoE, small | `allenai/OLMoE-1B-7B-0125-Instruct-GGUF` | `OLMoE-1B-7B-0125-Instruct-Q4_K_M.gguf` | ~4 GB |
| MoE, large (if VRAM+RAM allow) | `unsloth/Qwen3-30B-A3B-GGUF` | `Qwen3-30B-A3B-Q4_K_M.gguf` | ~18.6 GB |

```sh
mkdir -p models && cd models
for u in \
  ggml-org/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q8_0.gguf \
  Qwen/Qwen3-8B-GGUF/resolve/main/Qwen3-8B-Q4_K_M.gguf \
  unsloth/Qwen3-1.7B-GGUF/resolve/main/Qwen3-1.7B-BF16.gguf \
  allenai/OLMoE-1B-7B-0125-Instruct-GGUF/resolve/main/OLMoE-1B-7B-0125-Instruct-Q4_K_M.gguf ; do
  curl -L -O "https://huggingface.co/$u"
done
sh ../scripts/get-wikitext-2.sh     # creates wikitext-2-raw/wiki.test.raw
cd ..
```

KV cache flags (from `common/arg.cpp`):
- `-ctk` / `--cache-type-k` and `-ctv` / `--cache-type-v`. Values: `f16 q8_0 q4_0 ... turbo2 turbo3 turbo4 q6_0 q6_1 q3_0 q3_1 q2_0s q2_1`.
- `-fa on` (`--flash-attn`). The turbo types turn FA on automatically and print a warning.

Set a few shell variables first:

```sh
M=models/Qwen3-0.6B-Q8_0.gguf
W=models/wikitext-2-raw/wiki.test.raw
PPL="./build-vk/bin/llama-perplexity -m $M -f $W -c 512 --chunks 16 -ngl 99 -fa on"
```

### 4a. Dense model, KV-type perplexity (Vulkan vs CPU reference)

The test is whether Vulkan matches the CPU numbers. The absolute value matters less.

```sh
for kv in "f16 f16" "q8_0 q8_0" "q8_0 turbo4" "q8_0 turbo3" "q8_0 turbo2" "q8_0 q6_0" "q8_0 q3_0" \
          "turbo4 turbo4" "turbo3 turbo3" "q6_0 q6_0" "q3_0 q3_0"; do
  set -- $kv; echo -n "K=$1 V=$2: "
  $PPL -ctk $1 -ctv $2 2>&1 | grep 'Final estimate'
done
```

CPU reference on this branch (Qwen3-0.6B-Q8_0, `-c 512 --chunks 16`):

| K / V | CPU PPL | Expected on Vulkan |
|---|---|---|
| f16 / f16 | 25.12 ± 1.33 | within ±1% of CPU |
| q8_0 / q8_0 | 25.56 ± 1.35 | within ±1% of CPU; this is the **baseline** |
| q8_0 / turbo4 | 25.69 ± 1.36 | within ±1% of CPU, ≤ +2% vs baseline |
| q8_0 / turbo3 | 26.05 ± 1.38 | within ±1% of CPU, ≤ +3% vs baseline |
| q8_0 / q3_0 | 26.03 ± 1.38 | within ±1% of CPU, ≤ +3% vs baseline |
| q6_0 / q6_0 | 27.08 ± 1.42 | within ±2% of CPU |
| q4_0 / q4_0 (upstream type) | 126.4 | about the same as CPU; this model breaks with any 4-bit K |
| turbo4 / turbo4 | 389.8 | about the same as CPU (see note) |
| turbo3 / turbo3 | 2860 | about the same as CPU (see note) |
| q3_0 / q3_0 | 1378 | about the same as CPU |

**Note on quantized K with Qwen3-0.6B:** on this model, 4-bit-or-lower K cache blows up perplexity
on the CPU backend too, even with upstream's own q4_0. Use **q8_0 K with turbo or beellama V** as the
pass/fail config. Treat the symmetric rows only as Vulkan == CPU checks.

The symmetric turbo4 result (390, versus 126 for q4_0) is worse than expected even for a sensitive
model. It's tracked as a separate follow-up. If Qwen3-8B with `turbo4/turbo4` also gives more than
+10% vs q8_0, report it.

The same loop with `M=models/Qwen3-8B-Q4_K_M.gguf` is a better quality test. Expect
q8_0/turbo4 within +1% of the q8_0/q8_0 baseline and q8_0/turbo3 within +3%.

### 4b. Dense model, generation sanity (no gibberish)

```sh
./build-vk/bin/llama-completion -m $M -ngl 99 -fa on -ctk q8_0 -ctv turbo3 -no-cnv -n 128 --temp 0 \
  -p "Explain in three sentences why the sky is blue."
./build-vk/bin/llama-completion -m $M -ngl 99 -fa on -ctk turbo4 -ctv turbo4 -no-cnv -n 128 --temp 0 \
  -p "Explain in three sentences why the sky is blue."
```

Expected: coherent English about light scattering in the atmosphere, with no repeated tokens and no `?????`
or garbage. With `--temp 0`, the q8_0/turbo3 output should start with the same words as
`-ctk q8_0 -ctv q8_0`.

For an interactive check, use `./build-vk/bin/llama-cli -m $M -ngl 99 -ctk q8_0 -ctv turbo3 -st -p "..."`
(`-st` = single turn).

### 4c. TQ3_1S / TQ4_1S weights end to end (exercises fix 35b3c10bc in a real model)

```sh
./build-vk/bin/llama-quantize models/Qwen3-1.7B-BF16.gguf models/Qwen3-1.7B-TQ4_1S.gguf TQ4_1S
./build-vk/bin/llama-quantize models/Qwen3-1.7B-BF16.gguf models/Qwen3-1.7B-TQ3_1S.gguf TQ3_1S
./build-vk/bin/llama-quantize models/Qwen3-1.7B-BF16.gguf models/Qwen3-1.7B-Q8_0.gguf   Q8_0
for q in Q8_0 TQ4_1S TQ3_1S; do
  echo "== $q vulkan"; ./build-vk/bin/llama-perplexity  -m models/Qwen3-1.7B-$q.gguf -f $W -c 512 --chunks 16 -ngl 99 2>&1 | grep 'Final estimate'
  echo "== $q cpu";    ./build-cpu/bin/llama-perplexity -m models/Qwen3-1.7B-$q.gguf -f $W -c 512 --chunks 16        2>&1 | grep 'Final estimate'
done
```

Expected:
- Vulkan and CPU agree within 1% for each file.
- TQ4_1S is ≤ +5% vs Q8_0 and TQ3_1S is ≤ +15% vs Q8_0, as rough targets.
- Before 35b3c10bc, TQ prompt processing (batches > 8) produced garbage on Vulkan, so a PPL in
  the hundreds means that bug is back.

### 4d. MoE model, including hot/cold expert packs (exercises fix 6b5c893d5)

```sh
MOE=models/OLMoE-1B-7B-0125-Instruct-Q4_K_M.gguf
# plain full offload
./build-vk/bin/llama-perplexity -m $MOE -f $W -c 512 --chunks 16 -ngl 99 -fa on 2>&1 | grep 'Final estimate'
# experts on CPU with a GPU hot-expert cache -> MUL_MAT_ID gets id=-1 slots
./build-vk/bin/llama-perplexity -m $MOE -f $W -c 512 --chunks 16 -ngl 99 -fa on \
  --n-cpu-moe 99 --moe-cache-slots 16 2>&1 | grep -E 'moe|cache|Final estimate'
# same, with turbo V cache
./build-vk/bin/llama-perplexity -m $MOE -f $W -c 512 --chunks 16 -ngl 99 -fa on \
  --n-cpu-moe 99 --moe-cache-slots 16 -ctk q8_0 -ctv turbo3 2>&1 | grep 'Final estimate'
```

Expected: all three PPLs within ±1–2% of each other, and of the CPU build, with no crash at load.
For `--moe-cache-mib` on Vulkan, see the known-issues section.

### 4e. Speed (llama-bench), q8_0 KV baseline vs fork KV types

```sh
./build-vk/bin/llama-bench -m models/Qwen3-8B-Q4_K_M.gguf -ngl 99 -fa 1 -p 512 -n 128 -r 3 \
  -ctk q8_0,q8_0,q8_0,turbo4,q6_0 -ctv q8_0,turbo4,turbo3,turbo4,q6_0
./build-vk/bin/llama-bench -m $MOE -ngl 99 -fa 1 -p 512 -n 128 -r 3 -ctk q8_0 -ctv q8_0,turbo3
./build-vk/bin/llama-bench -m models/Qwen3-1.7B-Q8_0.gguf,models/Qwen3-1.7B-TQ4_1S.gguf,models/Qwen3-1.7B-TQ3_1S.gguf \
  -ngl 99 -p 512 -n 128 -r 3
```

Expected, as a rough guide since no AMD numbers exist yet:
- **tg128**: turbo and beellama V types within about 15% of the q8_0/q8_0 row.
- **pp512**: turbo3/turbo4 can be 10–30% slower than q8_0 at short context. The quantized-KV
  prefill path dequantizes only q8_0 to f16; the other types run the slower direct path.
- **TQ weights**: TQ4_1S/TQ3_1S pp512 will be clearly slower than Q8_0, because non-id MUL_MAT goes
  through the dequant path on purpose since 35b3c10bc. tg128 should be close to Q8_0 or faster.
- Anything **under 50%** of the baseline, or a crash, is worth reporting.

## 5. Known issues that are not fork bugs

- **lavapipe coopmat1 MUL_MAT mismatch.** On lavapipe, about 120 MUL_MAT cases fail with NMSE
  around 6e-4 to 2e-2, across all types and on plain upstream too. This is upstream issue
  https://github.com/ggml-org/llama.cpp/issues/30563: a Mesa llvmpipe barrier problem in the
  coopmat1 shader. `GGML_VK_DISABLE_COOPMAT=1` makes it pass. It should **not** happen on a real AMD
  GPU; if MUL_MAT fails there, it's real.
- **`test-fp8-requantize` flakes.** It sometimes fails on plain upstream as well (new upstream FP8
  type). A rerun usually passes.
- Other baseline ctest failures that aren't code problems:
  - `test-tokenizers-ggml-vocabs` needs git-lfs vocab files.
  - `test-arg-parser` needs network.
  - `test-llama-context-model` is flaky under parallel load.
  - `test-jinja-py` and `test-conversion-fp8` need Python `jinja2` / `torch`.
- **Upstream Vulkan issues you might hit:**
  - `--moe-cache-mib` crashes at load on some AMD cards: https://github.com/ggml-org/llama.cpp/issues/30697
  - Random all-NaN logits on some RDNA2 iGPUs with integer dot; work around with
    `GGML_VK_DISABLE_INTEGER_DOT_PRODUCT=1`: https://github.com/ggml-org/llama.cpp/issues/31509
  - Very long context with a quantized KV cache can fail to allocate the FA dequant scratch:
    https://github.com/ggml-org/llama.cpp/issues/30048. A fix is staged locally on `vulkan-next`.

## 6. What to send back if something fails

1. `git log -1 --oneline` and the build directory you used.
2. Device info: the first lines of any run (`ggml_vulkan: 0 = ...`), the GPU model, the driver
   (RADV/Mesa version or AMDVLK/proprietary), and the OS.
3. For test-backend-ops: the exact command plus the failing lines, for example:
   ```sh
   $B -b Vulkan0 -o FLASH_ATTN_EXT -p 'type_K=turbo3,' 2>&1 | grep -B2 -A6 FAIL > fail.log
   ```
   If it crashes, include the last ~30 lines and whether `GGML_VK_DISABLE_COOPMAT=1` or
   `GGML_VK_DISABLE_INTEGER_DOT_PRODUCT=1` changes the result.
4. For real models: the full command, the `Final estimate` line for both the Vulkan and CPU builds,
   and 2–3 lines of the generated text if it's gibberish.
5. For speed: the llama-bench table, with the q8_0 baseline row included.
