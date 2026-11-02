# Testing the fork stack (#62–#88): start here

Open PRs #62 through #88 form one linear stack, with each PR based on the one before it.
The `vulkan-next` branch (PR #88) is at the top, so it contains everything. **Test it once there.**
You only need the other branches for bisecting (section 5).

Model downloads, the perplexity reference numbers, and the known non-fork issues are in
[TESTING-vulkan-fork.md](TESTING-vulkan-fork.md). This guide refers to its sections as "VF §N".

The build box has no GPU. Everything below has been compiled for CPU, Vulkan and CUDA, and the Vulkan op
tests have run on lavapipe, which is a CPU Vulkan driver. None of it has run on a real AMD GPU.

## 1. One-time setup

```sh
git clone https://github.com/canonrebel04/llama.cpp.git && cd llama.cpp   # or: cd llama.cpp && git fetch origin
git checkout vulkan-next && git pull --ff-only

# Vulkan (main target). Needs libvulkan-dev, glslc, spirv-headers (or the Vulkan SDK).
cmake -B build-vk  -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-vk -j"$(nproc)"
./build-vk/bin/llama-cli --list-devices            # your AMD GPU should show up as Vulkan0

# CPU reference build (for comparing perplexity numbers against)
cmake -B build-cpu -DCMAKE_BUILD_TYPE=Release
cmake --build build-cpu -j"$(nproc)"
```

Optional builds. They use the same commands below, with `-b CUDA0`, `-b ROCm0` or `-b MTL0`:

```sh
cmake -B build-cuda -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=native -DCMAKE_BUILD_TYPE=Release   # NVIDIA
cmake -B build-hip  -DGGML_HIP=ON  -DGPU_TARGETS=gfx1100 -DCMAKE_BUILD_TYPE=Release              # ROCm; use your gfx (gfx1151 = Strix Halo)
cmake -B build-mtl  -DCMAKE_BUILD_TYPE=Release                                                   # macOS, Metal is on by default
```

Shortcuts used below:

```sh
B=./build-vk/bin/test-backend-ops     # every command compares Vulkan0 against the CPU backend
```

If you have more than one Vulkan device, pick one with `GGML_VK_VISIBLE_DEVICES=<index>`.

**Pass criteria for any `$B` command:** the run ends with `N/N tests passed` and the backend
line says `OK`. `not supported` lines are fine, because those cases fall back to the CPU. Any `FAIL`, abort
or crash is a bug.

## 2. Which PRs need your AMD GPU

"Vulkan" means the PR changes Vulkan code, so it needs your GPU. In the hardware column, "no"
means any machine can test it; a CPU-only build is enough.

| PR | Branch | What it touches | Needs AMD GPU? |
|---|---|---|---|
| #62 | `sync-upstream-20261011` | Upstream sync (Vulkan, CUDA, SYCL, Metal, CPU) and **Metal port of turbo/TQ/CR kernels** | **yes** (Vulkan). The Metal part needs a Mac |
| #63 | `forks/turbo-tan` | CUDA TurboQuant kernels; one Vulkan header include fix | no (CUDA). Vulkan is compile-only |
| #64 | `forks/thetom-fixes` | **Vulkan turbo3 codebook, turbo FA kept off coopmat2, MoE-cache queue sync**; CUDA wave64; server | **yes** |
| #65 | `forks/prism-fixes` | kv-cache / server / sampling | no |
| #66 | `forks/thetom-perf` | CPU FA; **Vulkan coopmat2 turbo decode** (NVIDIA-only path); HIP D=576 | no on RADV (no coopmat2). Yes on NVIDIA Vulkan |
| #67 | `forks/tq3-4s` | TQ3_4S weight type (CPU, CUDA, quantize) | no |
| #68 | `forks/strix-vulkan` | **Vulkan coopmat1 FA wave32 pin, RDNA3 MMQ stores, RMS_NORM fusions, top_k**; ggml-alloc | **yes**, ideally on Strix Halo |
| #69 | `forks/beellama` | Beellama KV types (CPU), draft batch sizes, DFlash, reasoning-loop guard | no |
| #70 | `forks/prism-ternary` | PQ2_0 / PTQ1_0 types (CPU, CUDA) | no |
| #71 | `forks/kv-stream` | Block KV streaming, **CUDA only** (`--kv-stream-arena-mib`) | no (needs NVIDIA) |
| #72 | `forks/mtp-mm` | MTP/NextN CLI compatibility, scripts, docs | no (any backend) |
| #73 | `forks/rknpu` | RKNPU2 backend, off by default | no (needs a Rockchip board) |
| #74 | `forks/ik-quants` | IQ2_K..IQ6_K (CPU, CUDA dequant) | no |
| #75 | `forks/gpu-kernels-newtypes` | **Vulkan IQ_K dequant / MUL_MAT**; CUDA IQ_K MMVQ and beellama KV | **yes** |
| #76 | `forks/tq3-4s-fast` | CUDA TQ3_4S MMVQ | no (needs NVIDIA) |
| #77 | `forks/ik-quants-ext` | IQ_KS / KSS / KL types (CPU) | no |
| #78 | `forks/skipped-picks` | F16 norm cast; CUDA FA long-context OOM | no |
| #79 | `forks/vulkan-forktypes` | **Vulkan TQ3_4S / PQ2_0 / PTQ1_0 dequant** | **yes** |
| #80 | `forks/vulkan-beellama` | **Vulkan SET_ROWS / dequant / FA for q6_0..q2_1 KV** | **yes** |
| #81 | `forks/iqk-simd` | CPU AVX2/NEON IQ_K dot products | no |
| #82 | `forks/iqk-kt` | IQ1_KT..IQ4_KT (CPU); CUDA kernels not enabled | no |
| #83 | `forks/vulkan-iqks` | **Vulkan IQ_KS dequant** | **yes** |
| #84 | `forks/thetom-ports` | SYCL TQ KV dequant; HIP/MUSA hostalloc; deepseek4 meta splits | no (SYCL = Intel GPU; HIP part = `build-hip`) |
| #85 | `sync-upstream-20261023` | Upstream sync, plus **3 Vulkan fixes** (TQ MUL_MAT, turbo SET_ROWS, MUL_MAT_ID skip) | **yes** |
| #86 | `sync-upstream-20261101` | Upstream sync, plus kv-stream/turbo no_alloc fix (TURBO2_0 → id 58) | **yes** (upstream Vulkan changes) |
| #87 | `sync-upstream-20261102` | Upstream sync, plus the Vulkan test-plan doc | **yes** (upstream Vulkan changes) |
| #88 | `vulkan-next` | **Vulkan FA K+V scratch size check** (upstream #30487), plus this guide | **yes** |

To sum up, the AMD GPU is needed for #62, #64, #68, #75, #79, #80, #83, #85–#88. A Mac is needed for #62's Metal port.
An NVIDIA GPU is needed for #63, #71, #76 and #78, and the CUDA parts of #64, #66, #75 and #84. Everything else
runs on CPU, so the CPU-only check in Group E covers it.

## 3. Recommended test order

Do the quick op tests first: they run in minutes and cover the riskiest changes. Each step lists the PRs it covers.

### Group A: Vulkan fixes and the newest change (#85, #88, #64, #80)

```sh
# #85: TQ weights through the dequant path, turbo SET_ROWS, skipped MoE experts
$B -b Vulkan0 -o MUL_MAT    -p 'type_a=tq(3|4)_1s'
$B -b Vulkan0 -o MUL_MAT_ID -p 'type_a=tq(3|4)_1s'
$B -b Vulkan0 -o SET_ROWS
$B -b Vulkan0 -o MUL_MAT_ID -p 'skip=1'

# #88: FA dequant scratch check. Run once normally, then with the limits forced small so the new fallback is taken
$B -b Vulkan0 -o FLASH_ATTN_EXT -p 'hsk=128,.*kv=512,.*type_K=(f16|q8_0|q6_0),'
GGML_VK_FORCE_MAX_ALLOCATION_SIZE=786432 GGML_VK_FORCE_MAX_BUFFER_SIZE=786432 \
  $B -b Vulkan0 -o FLASH_ATTN_EXT -p 'hsk=128,.*kv=512,.*type_K=(q8_0|q6_0),'

# #64 / #80: fork KV types in flash attention (turbo, beellama)
$B -b Vulkan0 -o FLASH_ATTN_EXT -p 'hsk=128,.*kv=512,.*type_K=(turbo3|turbo4),'
$B -b Vulkan0 -o FLASH_ATTN_EXT -p 'hsk=128,.*kv=512,.*type_K=(q6_0|q6_1|q3_0|q3_1|q2_0s|q2_1),'
$B -b Vulkan0 -o FLASH_ATTN_EXT -p 'turbo4_vec'
```

Expected: everything OK.
- On RDNA, which has 32-wide subgroups, the turbo/TQ4_1S `SET_ROWS` cases should actually **run** and pass. On lavapipe they show as not supported.
- Under the forced limits, some FA cases report `not supported`. That's expected.
- The filters above are kept narrow on purpose. For a full sweep, drop the `hsk=128,.*kv=512,.*` part, and add `-j 4`, because it takes a long time.

Lavapipe reference, `hsk=128,kv=512`:
- f16/q8_0: 1122/1122.
- q8_0 with forced limits: 804/804. q6_0 with forced limits: 150/150.
- turbo3 and turbo4: 280/280 each, or 196/196 each under forced limits, with `nb=(1|8|75)`.

### Group B: new weight types on Vulkan (#75, #79, #83)

```sh
$B -b Vulkan0 -o MUL_MAT,GET_ROWS -p 'type_a=(iq2_k|iq3_k|iq4_k|iq5_k|iq6_k),'                    # #75
$B -b Vulkan0 -o MUL_MAT,GET_ROWS -p 'type_a=(tq3_4s|pq2_0|ptq1_0),'                               # #79
$B -b Vulkan0 -o MUL_MAT,GET_ROWS -p 'type_a=(iq4_ks|iq2_ks|iq4_kss|iq5_ks|iq3_ks|iq2_kl),'        # #83
$B -b Vulkan0 -o MUL_MAT_ID -p 'type_a=(iq4_k|iq4_ks|tq3_4s),'
```

Expected: all OK or `not supported`; no FAIL. The IQ1_KT..IQ4_KT types (#82) have no Vulkan kernels yet,
so they should show `not supported` and fall back to the CPU.

If a MUL_MAT case fails, rerun the same filter with `type_a=q4_0` and with `GGML_VK_DISABLE_COOPMAT=1`.
If upstream's q4_0 fails the same way, or the failure goes away with coopmat off, it is the coopmat1 issue
(VF §5), not a fork-type bug. Please still report it. On lavapipe, the `m=1,n=64` cases fail for *all* types,
upstream ones included, and pass with coopmat off.

### Group C: Strix Halo / RDNA3 tuning (#68)

```sh
$B -b Vulkan0 -o MUL_MAT -p 'type_a=(q4_0|q8_0|q4_K),'      # int8 MMQ vec4 store path
$B -b Vulkan0 -o RMS_NORM,RMS_NORM_MUL_ADD,RMS_NORM_MUL_ROPE,TOP_K
$B -b Vulkan0 -o FLASH_ATTN_EXT -p 'hsk=(64|128),.*type_K=(f16|q8_0),'     # coopmat1 FA, wave32 pin
GGML_VK_FA_WAVE32=0 $B -b Vulkan0 -o FLASH_ATTN_EXT -p 'hsk=(64|128),.*type_K=(f16|q8_0),'

M8=models/Qwen3-8B-Q4_K_M.gguf
for w in 1 0; do GGML_VK_FA_WAVE32=$w ./build-vk/bin/llama-bench -m $M8 -ngl 99 -fa 1 -p 512 -n 128 -d 0,8192 -r 3 -ctk q8_0 -ctv q8_0; done
```

Expected:
- All op tests OK.
- With `GGML_VK_FA_WAVE32=1`, which is the default, pp512 at `-d 8192` is the same as `=0` or faster, by up to ~10%.
- tg128 is within ~2%.

If `=1` is clearly slower on your card, report it.

### Group D: real models (#62, #64, #80, #85–#88)

Download the models first (VF §4). Then run:
- **KV-type perplexity** on Vulkan vs CPU (VF §4a): `-ctk/--cache-type-k`, `-ctv/--cache-type-v` with
  `turbo2 turbo3 turbo4 q6_0 q6_1 q3_0 q3_1 q2_0s q2_1`, plus `-fa on`.
  Pass = Vulkan within 1% of the CPU build.
- **Generation sanity check** (VF §4b): `llama-completion ... -ctk q8_0 -ctv turbo3 -no-cnv --temp 0`.
- **TQ weights end to end** (VF §4c).
- **MoE with the hot/cold expert cache** (VF §4d): `--n-cpu-moe 99 --moe-cache-slots 16`.
- **New weight types end to end** (#75, #79, #83). Pass = Vulkan within 1% of CPU for each file:
  ```sh
  W=models/wikitext-2-raw/wiki.test.raw
  for q in IQ4_K IQ4_KS TQ3_4S; do
    ./build-vk/bin/llama-quantize models/Qwen3-1.7B-BF16.gguf models/Qwen3-1.7B-$q.gguf $q
    for b in build-vk build-cpu; do echo -n "$q $b: "
      ./$b/bin/llama-perplexity -m models/Qwen3-1.7B-$q.gguf -f $W -c 512 --chunks 16 -ngl 99 2>&1 | grep 'Final estimate'
    done
  done
  ```
- **Long context, #88's actual bug** (upstream issue #30048). Use a prompt file that fills most of the context:
  ```sh
  ./build-vk/bin/llama-completion -m $M8 -ngl 99 -fa on -ctk q8_0 -ctv q8_0 -c 131072 -f long-prompt.txt -n 64 -no-cnv
  ```
  Run it on `vulkan-next`, then on `sync-upstream-20261102` (the baseline without the fix).
  Expected:
  - The baseline may fail with `ErrorOutOfDeviceMemory` during prompt processing. If it doesn't, raise `-c`.
  - `vulkan-next` finishes and the output is coherent.
  - Repeat on `vulkan-next` with `-ctk turbo3 -ctv turbo3`.
- **Speed** (VF §4e). Anything under 50% of the q8_0 baseline row, or a crash, is a bug.

### Group E: no GPU needed (#65, #67, #69, #70, #72, #74, #77, #81, #82)

On any machine, with `build-cpu`:

```sh
ctest --test-dir build-cpu -j"$(nproc)" --output-on-failure
./build-cpu/bin/test-quantize-fns
```

Expected: everything passes, except the known baseline failures listed in VF §5:
- `test-tokenizers-ggml-vocabs`
- `test-arg-parser`
- `test-llama-context-model`, which is flaky
- sometimes `test-fp8-requantize`
- `test-jinja-py`, and `test-conversion-fp8` without Python deps

Optional feature checks:
- **#69 draft batch sizes:** `llama-server -m <target> -md <draft> -ubd 64 -bd 256`.
- **#69 reasoning-loop guard:** `--reasoning-loop-guard`.
- **#72 MTP/NextN:** `llama-server -m <qwen3.6 or gemma4> --mtp-head <head.gguf> --spec-type draft-mtp`. See `scripts/run-qwen36-*-nextn-server.sh`.

Group E also covers the NVIDIA-only PRs (#63, #71, #76, #78): run Groups A–D on `build-cuda` with `-b CUDA0`. kv-stream (#71) is enabled with `--kv-stream-arena-mib N`; see `docs/kv-stream.md`.

## 4. What to send back

Send the following (details in VF §6):
- `git log -1 --oneline`
- The `ggml_vulkan: 0 = ...` line and your driver (RADV/Mesa version or AMDVLK)
- The exact command, plus the `FAIL` lines or the last ~30 lines before a crash
- Whether `GGML_VK_DISABLE_COOPMAT=1` or `GGML_VK_DISABLE_INTEGER_DOT_PRODUCT=1` changes the result

## 5. Bisecting a failure

Each PR's branch contains every PR below it. So the first branch, in stack order, that fails a test is the one that broke it.

```sh
for b in sync-upstream-20261011 forks/strix-vulkan forks/thetom-ports sync-upstream-20261023 \
         sync-upstream-20261101 sync-upstream-20261102 vulkan-next; do
  git checkout -q origin/$b && cmake --build build-vk -j"$(nproc)" >/dev/null &&
  echo -n "$b: " && $B -b Vulkan0 -o <OP> -p '<filter>' 2>&1 | tail -1
done
```

1. Start with the coarse list above.
2. Narrow down inside the failing range using the branch names in the table (section 2).
3. Report the first failing branch.

Some branches are only worth testing when the failure matches what they change:
- Vulkan op failures: the Vulkan-touching PRs #64, #68, #75, #79, #80, #83, and the syncs.
- Failures that also happen on the CPU build: the CPU PRs.

Note: anything that doesn't reproduce on #62's branch, but fails on a sync branch (#85–#87), may be upstream's bug.
Check it against plain upstream with `git checkout <upstream commit named in that PR's title>`.
