# inferium

The simplest CPU inference engine for Llama-style LLMs, written from scratch in C++17.

Every layer is built directly on the standard library and ARM NEON intrinsics: GGUF
parsing, `Q4_K`/`Q6_K` dequantization, RMSNorm, RoPE, grouped-query attention, SwiGLU,
greedy sampling. It `mmap`s the model file, serves requests over HTTP with continuous
batching, and spreads them across worker nodes over gRPC.

I wrote this to find out where inference actually spends its time. The short answer: not
where I expected. See efficiency notes.

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

You get ` Paris.` and then some rambling about ancient Rome, at ~9 tok/s on an M4. It's a
1.1B model doing greedy sampling on a CPU — set expectations accordingly.

Threads default to core count, override with `INFERIUM_THREADS=8`. Eight is the sweet
spot; past that it slows down (again, efficiency notes).

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
That's the whole point of the thing.

One scheduler thread owns all the model compute; request threads just enqueue and wait on
a future. This is on purpose. The matmul already fans out across the thread pool, so if
requests also ran *on* that pool, a blocked worker could end up waiting on subtasks it was
itself supposed to run. One scheduler thread sidesteps that, and keeps the weights hot in
cache across sequences.

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

Worth saying plainly: two workers on *one* machine buy you redundancy. They share the same
memory bus, and memory is the bottleneck, so speed needs separate machines.

## benchmarks

Apple M4 (4P + 6E cores, 24GB LPDDR5, ~120 GB/s), TinyLlama-1.1B `Q4_K_M`.

Single-stream decode, over the life of the project:

| stage | tok/s |
| --- | --- |
| naive (no KV-cache, scalar, 1 thread) | 0.13 |
| \+ parallel matmul | 0.42 |
| \+ KV-cache | ~2.5 |
| \+ NEON SIMD | 6.5 |
| \+ arena + batched matmul | **9.2** |

About 70x, though the naive baseline had no KV-cache so it got *worse* the longer it ran.

Throughput vs batch size, 8 threads:

| batch | tok/s | p50 latency |
| --- | --- | --- |
| 1 | 6.2 | 3.4s |
| 4 | 14.5 | 5.7s |
| 8 | **15.5** | 12.6s |
| 16 | 17.1 | 22.4s |

~2.5x throughput at batch 8, median of 3 runs, ±15% run to run. Note the latency column —
that's the trade, and it's not subtle.

Run it yourself:

```
./build/inferium_bench model.gguf --requests 16 --concurrency 8 --batch 8 --threads 8
```

## efficiency notes

The interesting finding, and the reason the rest of the design looks the way it does.

Early on, 8 threads gave me a 3.4x speedup. After I added the KV-cache and NEON SIMD, the
same 8 threads gave 1.43x. Optimizing the compute *moved the wall closer*.

At batch size 1, decode reads every weight in the model from memory and uses each one for
exactly one multiply. 1.1B params at 4 bytes is ~4.4GB per token. At ~120 GB/s that's your
ceiling, and cores are irrelevant to it. Ten threads is actually slower than eight; the
efficiency cores add contention while bandwidth stays flat.

Which tells you what batching is for: it amortizes the weight *load*. `mat_mat` pulls one
weight vector into a register and fires four FMAs against four different sequences, so 8
sequences cost roughly the memory traffic of 2.
That's where the 2.5x comes from.

Gains flatten past batch 4-8 because attention is per-sequence work that batching can't
share, and every sequence's KV cache is fighting for the same cache lines.

## todos

- Keep weights in `Q4_K` and dequantize per-block inside the matmul. Right now they're
  dequantized to `float32` at load, which costs 4.8GB RSS for a 638MB file. Since the
  bottleneck is bytes moved, this should cut memory ~4x *and* make decode faster. Easily
  the highest-value thing left.
- Vectorize attention. It's still scalar and rescans the full history every step, which is
  most of why batch gains flatten.
- Stream tokens back instead of making you wait for the whole completion.
- Temperature, top-k, and top-p sampling.
- Share one set of weights across workers on the same box instead of each loading its own.

## troubleshooting

**`unsupported dtype`** — only `Q4_K` (12) and `Q6_K` (14) are implemented. Most
`Q4_K_M` files mix both, which is fine. `Q8_0`, `Q5_K` etc. will hit this.

**`prompt exceeds max_seq`** — bump `--max-seq`. It's 512 by default because each slot
preallocates its own KV cache and they add up.

**`arena exhausted`** — shouldn't happen; means the activation arena was sized wrong for
your batch. File an issue with your `--batch` and `--max-seq`.

**gRPC targets missing** — `find_package` didn't locate gRPC. On macOS,
`brew install grpc protobuf` and pass `-DCMAKE_PREFIX_PATH=/opt/homebrew`.
