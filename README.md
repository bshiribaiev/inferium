# inferium

Run LLMs on your own machine, serve them over HTTP, and spread requests across several
nodes.

It implements the whole path itself — GGUF parsing, quantized weights, the transformer
forward pass — but the interesting part is what makes it fast: a KV-cache, int8 weights
with a NEON `sdot` matmul, a spin-then-sleep worker team, an arena allocator, and
continuous batching so concurrent requests share one pass over the weights. Efficiency notes covers where the time actually goes; it turned out to
be memory bandwidth rather than cores.

Today that means Llama-architecture models in `Q4_K`/`Q6_K`, tested against TinyLlama-1.1B.
Broader architecture support is on the todo list.

## install

```
cmake -S . -B build -DCMAKE_PREFIX_PATH=/opt/homebrew
cmake --build build -j
```

You need a C++17 compiler and CMake 3.20+. gRPC and protobuf are optional — without them
the distributed targets get skipped and everything else still builds.

Grab a model. Anything Llama-architecture in `Q4_K_M` or `Q6_K` should work, but
TinyLlama-1.1B is what I tested against:

```
curl -LO https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
```

## quick start

```
./build/inferium tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf "The capital of France is"
```

You get ` Paris.` and then some rambling about ancient Rome, at ~80 tok/s on an M4.

Threads default to core count, override with `INFERIUM_THREADS=8`. Eight is the sweet
spot; past that it slows down (see efficiency notes).

To run on the GPU instead, set `INFERIUM_BACKEND=metal` (or pass `--backend metal` to the
server, worker and bench). CPU is the default.

## serving

```
./build/inferium_server tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf --port 8080 --batch 8
```

```
curl -X POST http://127.0.0.1:8080/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"The capital of France is","max_tokens":32}'
```

```json
{"text":" Paris.\n\n2. B.C. ...","prompt_tokens":6,"generated_tokens":32}
```

Fire 8 of those at once and they join a running batch, sharing one pass over the weights.

One scheduler thread owns all the model compute; request threads enqueue and wait on a
future. The matmul already fans out across the worker team, so if requests also ran *on*
that team, a blocked worker could end up waiting on subtasks it was itself supposed to
run. One scheduler thread sidesteps that, and keeps the weights hot in cache across
sequences.

## going distributed

Two workers behind a coordinator:

```
./build/inferium_worker model.gguf --port 50051 --batch 4 --threads 4 &
./build/inferium_worker model.gguf --port 50052 --batch 4 --threads 4 &
./build/inferium_coordinator --workers 127.0.0.1:50051,127.0.0.1:50052 --port 9090
```

The coordinator round-robins and speaks the same HTTP API, so `curl` treats it like any
single node. Kill a worker mid-run and requests keep succeeding — the round-robin cursor
advances on every retry, so a retry never lands back on the worker that just died. Kill
both and you get a `503` with the actual gRPC error instead of a hang.

Two workers on *one* machine buy redundancy. They share the same memory bus, and memory is
the bottleneck, so speed needs separate machines.

## benchmarks

Apple M4 (4P + 6E cores, 24GB LPDDR5, ~120 GB/s), TinyLlama-1.1B `Q4_K_M`.

Single-stream decode, over the life of the project:

| stage | tok/s |
| --- | --- |
| naive (no KV-cache, scalar, 1 thread) | 0.13 |
| \+ parallel matmul | 0.42 |
| \+ KV-cache | ~2.5 |
| \+ NEON SIMD | 6.5 |
| \+ arena + batched matmul | 9.2 |

Then a second round, measured in one sitting (64 tokens, average of 3 runs; the machine was
busier, so the previous stage re-measured at 16):

| stage | tok/s |
| --- | --- |
| arena + batched matmul, re-measured | 16 |
| \+ panel-packed weights, SME2 matmul for batches | 19 |
| \+ `float16` weights | 33 |
| \+ spin-then-sleep worker team | 36 |
| \+ attention heads split across threads | 42 |
| \+ `int8` weights with an `sdot` kernel | 74 |
| \- SME2 (int8 for every batch size) | **84** |

Thread count, int8 build: 4 → 60, 6 → 74, 8 → 81, 10 → 64 tok/s.

Throughput vs batch size, 8 threads:

| batch | tok/s | p50 latency |
| --- | --- | --- |
| 1 | 77 | 0.41s |
| 4 | 216 | 0.58s |
| 8 | **259** | 0.94s |
| 16 | 265 | 1.89s |

3.4x throughput at batch 8, average of 2 runs, 16 requests. Peak RSS is 2.3GB.
A ~250-token prompt prefills in 0.9s.

CPU vs Metal. The CPU column is from a quiet run; the Metal one is from a later run while
the machine was busy, which barely touches the GPU but cut the CPU's numbers by ~40%:

| | CPU | Metal |
| --- | --- | --- |
| single-stream, 64 tokens | 70–83 tok/s | 48–65 tok/s |
| batch 8 | 244 tok/s | **238–271 tok/s** |
| batch 16 | 243–264 tok/s | **260–298 tok/s** |
| ~250-token prefill | 0.90s | **0.40s** |

GPU time per decode step, which load doesn't distort:

| kernel | batch 1 | 2 | 4 | 8 | 16 |
| --- | --- | --- | --- | --- | --- |
| matrix-vector | **13.9ms** | 15.2ms | 24.8ms | 45.9ms | 90.9ms |
| 8-wide tiles | 14.7ms | **14.8ms** | **16.5ms** | **19.0ms** | **30.1ms** |
| 32-wide tiles | 31.9ms | 31.3ms | 33.7ms | 34.3ms | 43.6ms |

At ~1000 tokens of context a Metal decode step takes 15.9ms (31.2ms before the attention
rewrite) and a 1000-token prefill 1.7s.

Quality, scoring 4,088 tokens of Shakespeare with `inferium_perplexity`:

| | perplexity | top-1 accuracy |
| --- | --- | --- |
| CPU, int8 activations | 52.28 | 35.9% |
| Metal, float activations | 52.27 | 35.8% |

The two pick the same top token 98.3% of the time, with a mean KL divergence of 0.0005 nats,
so rounding activations to int8 costs about nothing.

Run it yourself:

```
./build/inferium_bench model.gguf --requests 16 --concurrency 8 --batch 8 --threads 8
./build/inferium_bench model.gguf --requests 16 --concurrency 8 --batch 8 --backend metal
./build/inferium_perplexity model.gguf some_text.txt --window 512 --max-tokens 4096
```

## efficiency notes

Early on, 8 threads gave a 3.4x speedup. After I added the KV-cache and NEON SIMD, the
same 8 threads gave 1.43x. Optimizing the compute *moved the wall closer*.

At batch size 1, decode reads every weight in the model from memory and uses each one for
exactly one multiply, so bytes per token sets the ceiling and cores are irrelevant to it.
At `float32` that was ~4.4GB per token; `float16` halved it and nearly doubled tok/s;
`int8` with a `float16` scale per 32 weights brings it to ~1.2GB. At 84 tok/s that's
~100 GB/s, close to the ~120 GB/s the bus can do. Ten threads is slower than eight; the
efficiency cores add contention while bandwidth stays flat.

The int8 kernel quantizes the activations the same way, so one `sdot` multiplies four
weight rows by four columns in integers, and the float scales are applied once per block.
Weights are packed in 16-row panels so each thread streams one contiguous run of memory.

Batching amortizes the weight *load*: each panel is read once and used for every sequence
in the batch, which is why batch 8 gets ~3.4x the throughput of batch 1.

SME2 (the M4's matrix unit) was faster than NEON on `float16` weights for batches of 8+,
but it can only read float16, twice the bytes of int8. Once the weights went to int8 the
NEON path won at every batch size, so SME was removed, which also dropped the float16
copy and halved memory. An int8 SME kernel (`SMOPA`) is the way to bring it back.

Matmuls are short (tens of microseconds), so waking threads through a mutex and condition
variable cost a real share of each one. Workers now spin briefly on a generation counter
and only sleep when idle, and the calling thread does a share of the work instead of
waiting.

Attention still grows with sequence length: at 256 tokens single-stream drops to ~60 tok/s.

The Metal backend shares the CPU's memory (unified memory), so the weights and KV cache are
used in place, with no copy, and single-stream hits the same ~100 GB/s ceiling. Its first
version ran each matmul as its own GPU job and spent ~200µs per round trip waiting, 20
tok/s in total; encoding a whole step into one command buffer removed that. Batches and
prefill are compute-bound, so they use tiled kernels that stage weight and activation
tiles in threadgroup memory and multiply them on the GPU's 8x8 matrix units, 8 batch rows
wide up to batch 31 and 32 wide beyond, with the columns split across threadgroups so
small batches still fill all ten GPU cores.

Metal attention gives each group of query heads that share a KV head one threadgroup, so
they read each K and V row together instead of 8 times over, and uses an online softmax
so nothing is stored per position. Decode has only one row, so its positions are also split
across threadgroups and merged, otherwise 4 threadgroups would leave most of the GPU idle.

Metal keeps activations in float while the CPU rounds them to int8, so the two can pick a
different token on a near tie; 3 of 8 bench prompts diverge partway through.

## todos

- 4-bit weights with a native 4-bit kernel. Bytes per token is still the ceiling, so
  halving them again should be the biggest remaining single-stream win. Dequantizing `Q4_K`
  on the fly in the matmul was tried and was 27% slower; the unpacking has to be as cheap
  as the int8 path's.
- Vectorize attention and store the KV cache as `float16` or `int8`. It's still scalar and
  rereads the full history every step, which is why long sequences slow down.
- Metal: single-stream still trails the CPU slightly; the matrix-vector kernel reaches
  ~94 GB/s, so the rest is in the small per-step kernels.
- Stream tokens back as they are generated.
- More architectures. `general.architecture` is already parsed out of the GGUF and unused,
  so that is the seam. Llama-family variants need QKV bias support and more quant formats;
  GPT-2 and Falcon additionally need LayerNorm, a non-gated GELU MLP, and byte-level BPE.
- Temperature, top-k, and top-p sampling.
- Share one set of weights across workers on the same box instead of each loading its own.

## troubleshooting

**`unsupported dtype`** — only `Q4_K` (12) and `Q6_K` (14) are implemented. Most
`Q4_K_M` files mix both, which is fine. `Q8_0`, `Q5_K` etc. will hit this.

**`matrix with N rows is not a multiple of 16`** / **`columns is not a multiple of 32`** —
weights are packed in 16-row panels and quantized in 32-column blocks. Every Llama size
fits; an odd custom model may not.

**`prompt exceeds max_seq`** — bump `--max-seq`. It's 512 by default because each slot
preallocates its own KV cache and they add up.

**`arena exhausted`** — the activation arena was sized wrong for your batch. File an issue
with your `--batch` and `--max-seq`.

**gRPC targets missing** — `find_package` didn't locate gRPC. On macOS,
`brew install grpc protobuf` and pass `-DCMAKE_PREFIX_PATH=/opt/homebrew`.
