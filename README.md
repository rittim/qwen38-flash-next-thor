# Qwen3.8-Flash-Next on a Jetson AGX Thor (with vision)

I got Qwen3.8-Flash-Next running on my AGX Thor 128GB, image input included. I couldn't
find anyone else who had done this on a Thor so I'm writing up what worked, since it took
a whole evening of trial and error to get here.

Short version: 15-18.5 tok/s decode on free-form text, vision works, fits in memory with
room to spare. My setup is L4T R38.2.2, CUDA 13.0, MAXN power mode, stock PSU.

## Why it fits at all

The model is 180B params on disk (125B MoE with 6B active, plus a 51B n-gram embedding
table and a 4B MTP head). The Q4_K_XL quant is 111GB which obviously doesn't leave much
on a 128GB board. Two things save it:

1. The 51B n-gram table never goes to the GPU. This trick comes from
   [0xBakeer's DGX Spark writeup](https://github.com/0xBakeer/qwen38-flash-next-spark):
   `-ot per_layer_token_embd=CPU -lm mmap` leaves the table on NVMe and lets the page
   cache deal with it. Resident memory ends up around 80GB.
2. A graph reuse patch. The qwen4exp branch rebuilds the whole compute graph every
   token, and on Thor that costs about 25%. `canreuse-v2.patch` in this repo fixes it.
   It's based on the (now closed) llama.cpp PR #27774 but that patch as-is will
   segfault on newer commits, see the use-after-free section below.

## Steps

```bash
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
git fetch origin pull/27742/head:qwen4exp
git checkout d4a943f
git cherry-pick -n 24ea62d           # crash fix, you want this
git apply canreuse-v2.patch

cmake -S . -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j $(nproc)
```

Models, ~112GB total:
- unsloth/Qwen3.8-Flash-Next-GGUF, the UD-Q4_K_XL folder (111.3GB in 4 shards)
- AtomicChat/Qwen3.8-Flash-Next-GGUF, just `mmproj-Qwen3.8-Flash-Next-F16.gguf` (0.9GB).
  Unsloth doesn't have an mmproj at the time of writing, AtomicChat's works fine.

Serve:

```bash
./build/bin/llama-server \
  -m Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf \
  --mmproj mmproj-Qwen3.8-Flash-Next-F16.gguf \
  --image-min-tokens 1024 \
  -lm mmap -ot per_layer_token_embd=CPU \
  --n-gpu-layers 999 --ctx-size 65536 --parallel 1 \
  --spec-type ngram-mod \
  --temp 1.0 --top-p 0.95 --top-k 20 \
  --host 0.0.0.0 --port 8080
```

After every server start it's worth warming the n-gram table with `warm_table.py` from
the Spark repo (one sequential read, ~30s, cuts page faults a lot). I run it from
ExecStartPost in my systemd unit, `flashnext.service.example` in this repo.

## Numbers

All with UD-Q4_K_XL, table warmed, MAXN:

| build | decode tok/s |
|---|---|
| branch head, no patch | 12.6 |
| branch head + patch | 16.3-17.8 |
| d4a943f + 24ea62d + patch (what I run) | 15.3-18.5 |

Notes:
- there's easily 1-2 tok/s of run to run noise from cache state, don't read too much
  into small differences
- prefill is 90-170 tok/s depending on caching. Images cost a one-off 1-2s to encode,
  decode speed is unaffected
- ngram-mod speculation is free (no draft model) and does a lot on copy/edit-heavy
  prompts, nothing on prose
- the 120W power mode costs about 10% vs MAXN

## Why commit d4a943f and not the branch head

The PR moves fast and the maintainers took it over recently. I bisected and found the
head was about 1 tok/s slower than d4a943f (looks like it came in with the quantized-KV
and tensor parallel commits, didn't narrow it further, it's within noise on any single
run). d4a943f already has the image support, and cherry-picking 24ea62d on top gets you
a real crash fix (iterator invalidation in the PLE history when sequences get removed,
which a server does constantly). Re-check all of this once the PR merges.

## The use-after-free that cost me an hour

If you port the graph reuse patch to a different commit: both graph input classes in
`src/models/qwen4exp.cpp` need to re-bind `this->mctx` from `params.mctx` inside
`can_reuse()` before the shape checks. Since commit d22d2be the PLE n-gram history
lives on the per-context memory object, so a reused graph with the old pointer reads a
freed context and you get a segfault on basically the first generation. The original
PR #27774 patch is from before that commit and doesn't do the re-bind for the PLE
input. That's the difference in the v2 patch here.

## Other things I hit

- Don't load the whole model GPU-resident (i.e. without the -ot/mmap trick). A ~94GB
  cudaMalloc against ~90GB free wedged my board hard enough to trigger a hardware reset
  with nothing in the logs. I chased phantom PSU problems for hours because of this.
- Stay at or below the native 262144 context. There's a known CUDA crash on this branch
  when YaRN-extended prefill crosses that boundary.
- `--parallel 1`. Concurrent requests aren't stable on this branch yet.
- KV cache is tiny (~24KB/token) thanks to the hybrid linear attention, so big contexts
  are cheap.
- If you put Open WebUI in front of it, turn off title/tag/follow-up generation,
  otherwise every message triggers hidden extra requests against the model and evicts
  your prompt cache.

## Credit where due

- danielhanchen/Unsloth for the qwen4exp llama.cpp bringup (PR #27742) and the quants
- jkyamog for the original graph reuse patch (PR #27774)
- 0xBakeer for the Spark recipe this is built on
- AtomicChat for the mmproj conversion
