# vference technical reference

Status: architecture/feasibility reference  
Initial architecture adapter and validation target: Qwen3.5-35B-A3B, text-only,
batch 1, Apple M2 MacBook Air with 8 GB unified memory  
Primary execution stack: MLX/Metal; Core ML is an optional, separately proven
acceleration path

This document is the source of truth for the initial design. Future agents
should update facts in place, add dates and source revisions when upstream
behavior changes, and preserve the distinction between measured results,
derived estimates, and proposals.

## 1. Goal and non-goals

The goal is useful local inference for a sparse model whose complete checkpoint
cannot fit in unified memory. The design keeps the text model's always-used
weights and recurrent state resident, stores routed experts in a runtime-specific
on-disk layout, and maintains a bounded expert working set in unified memory.
MLX executes the model math and Metal kernels. Native CPU code owns expert I/O,
cache bookkeeping, prediction, and memory-pressure response.

The intended product is a general runtime for oversized sparse generative
models. Qwen3.5 is the first architecture adapter because its expert structure
and hybrid backbone make it a demanding, useful validation target. Qwen tensor
names, layer types, routing rules, and state layouts belong behind an adapter;
the artifact store, memory controller, expert cache, I/O scheduler, telemetry,
and correctness harness should remain architecture-neutral. Generality does not
mean designing every abstraction before Qwen works: a second model family is
the test that the boundary is real.

### 1.1 Definition of usable

For the initial 8 GB M2 target, the provisional acceptance metric is **at least
2.0 sustained decode tokens/second** at batch 1 with an admitted 8K context and
at least 256 newly generated tokens, after warm-up, without swap growth, memory
pressure termination, thermal collapse, or any quality-invariant failure.
Report the median and p95 token latency over a sustained run, not a short burst.

The runtime must also show a material measured improvement over both baselines
on the same machine and exact model artifact:

1. the full conventional model left to macOS/MLX oversubscription and swapping,
   if it can complete the benchmark; and
2. correctness-first synchronous demand streaming with no expert prediction or
   asynchronous overlap.

Stage 0 measured the internal storage and synchronous runtime limits. The
absolute threshold is now frozen at 2.0 tok/s. For an optimized policy,
“material improvement” is provisionally frozen as at least **10% higher decode
throughput and 10% lower exposed expert-execution time per generated token**
than synchronous demand streaming on the same context workload and artifact.
Confidence intervals still require repeated runs. If conventional
oversubscription cannot complete without termination or uncontrolled swap,
record that failure and compare the completing runtime against synchronous
streaming. Any threshold revision requires a dated rationale and fixed
benchmark manifest. A system that merely completes inference but misses these
thresholds is a feasibility demonstration, not a successful runtime.

### 1.2 Artifact acquisition policy

Runtime development starts from the approximately 20 GB
`mlx-community/Qwen3.5-35B-A3B-4bit` artifact. It is the input to the first
artifact builder and the model whose math the streaming runtime must reproduce.
The immediate objective is to repack that existing quantized artifact, build
the runtime, and get Qwen generating correctly on the 8 GB target.

Do **not** download the approximately 67 GiB official BF16 checkpoint during
Stages 0–5. Acquire it only when either of these explicit triggers is reached:

1. Stage 6 begins BF16-versus-quantized quality evaluation; or
2. the project begins generating its own quantized or mixed-precision artifact
   from the official weights.

Reaching either trigger is not implicit authorization to download it. Tell the
user that BF16 is now required, and give them the command to download it.
The BF16 checkpoint is a later quality/conversion source;
it is not the artifact the 8 GB runtime will keep resident or stream directly.

The first release is text-only, single-request, batch-1 inference. Vision, MTP, training, and ANE acceleration are follow-on work. This
ordering matters: each adds independent memory, correctness, and scheduling
variables that would obscure whether expert streaming itself works.

This is not ordinary virtual-memory oversubscription. Letting macOS page a full
MLX model implicitly gives neither predictable eviction nor a protected state
budget. It can also cause the GPU working set and filesystem cache to compete
with each other. vference explicitly chooses which expert bytes occupy its
bounded slots and when they move.

## 2. Non-negotiable quality invariant

The runtime may change **where and when** a weight is stored or transferred. It
must not change **which computation the model performs**.

The production/default path therefore has these rules:

- Use the checkpoint's exact router, top-K of 8, routing weights, shared expert,
  and layer order.
- A cache miss stalls until every selected expert is ready. Never replace a
  missing expert, redirect it to a resident expert, skip it, reduce K, or use a
  lower-precision emergency copy.
- Prefetch prediction affects latency only. A wrong prediction may waste I/O
  and cache capacity, but cannot affect logits.
- Preserve the reference expert-output reduction/accumulation order. Parallel
  or expert-major implementations must demonstrate equivalent intermediate
  states and token output before replacing the reference path.
- Repacking is lossless. Every tensor payload is checksummed before and after
  conversion, and reconstructed tensors must match the input artifact.
- Quantization is a model conversion decision, not a transparent runtime
  optimization. Runtime correctness is judged against a fully resident run of
  the **same quantized artifact**, MLX version, prompt, sampler, and seed.

The last point avoids an impossible comparison. An 8 GB machine requires a
quantized text trunk and quantized experts; quantization itself can change model
quality relative to BF16. A candidate quantization recipe must separately pass
quality evaluation against BF16 before it becomes a supported artifact. Runtime
work cannot use quantization changes to conceal storage/scheduling regressions.

Quality-affecting ideas such as top-K reduction, expert substitution, expert
remapping, layer dropping, lossy state compression, or approximate routing do
not belong in the default product. If researched, they must live behind an
explicit experimental build flag and may never be used to report the main
runtime's throughput.

## 3. Verified Qwen3.5-35B-A3B facts

The official configuration at Hugging Face revision
[`59d61f3`](https://huggingface.co/Qwen/Qwen3.5-35B-A3B/blob/59d61f3ce65a6d9863b86d2e96597125219dc754/config.json)
defines the following text model:

| Property                          |                               Value | Runtime consequence                                     |
| --------------------------------- | ----------------------------------: | ------------------------------------------------------- |
| Hidden width                      |                               2,048 | Common residual width and expert input/output           |
| Decoder layers                    |                                  40 | Every layer contains a routed MoE block                 |
| Layer pattern                     | 3 Gated DeltaNet : 1 full attention | 30 recurrent layers, 10 KV-cached layers                |
| Routed experts/layer              |                                 256 | 10,240 independently addressable expert records         |
| Selected experts/token/layer      |                                   8 | 320 expert uses per generated token before cache hits   |
| Expert intermediate width         |                                 512 | Each expert has gate, up, and down projections          |
| Shared expert width               |                                 512 | Always executed and therefore part of resident trunk    |
| Full-attention Q heads / KV heads |                              16 / 2 | GQA keeps KV growth smaller than ordinary MHA           |
| Head dimension                    |                                 256 | Used in KV sizing                                       |
| Native maximum context            |                             262,144 | Architectural capability, not an 8 GB operating promise |
| Vocabulary                        |                             248,320 | Embedding and untied LM head are each large             |
| Stored dtype                      |     BF16, with 4,800 F32 parameters | Original checkpoint is about 66.96 GiB                  |

The model is natively multimodal and includes a vision tower and one MTP layer,
but current MLX-LM text loading removes both. The official model description
also confirms the 3:1 Gated DeltaNet/attention backbone and 8 routed plus 1
shared expert design. See the
[Qwen model card](https://huggingface.co/Qwen/Qwen3.5-35B-A3B),
[Qwen3.5 announcement](https://qwen.ai/blog?id=qwen3.5), and
[Transformers architecture documentation](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/qwen3_5_moe.md).

### 3.1 Checkpoint byte accounting

The figures below were derived from the safetensors headers, not inferred from
the marketing parameter count. The official index reports 71,903,655,008 bytes
of payload across 1,811 tensors and 14 shards.

| Checkpoint region            | BF16/F32 payload | Treatment in the first runtime                                           |
| ---------------------------- | ---------------: | ------------------------------------------------------------------------ |
| Main text routed experts     |       60.000 GiB | Repack, quantize as a qualified artifact, keep on SSD with bounded slots |
| Main text non-routed weights |        4.560 GiB | Quantized and resident                                                   |
| Vision tower                 |        0.832 GiB | Excluded from text-only runtime                                          |
| MTP layer                    |        1.573 GiB | Excluded initially                                                       |
| Total                        |       66.966 GiB | Matches official index payload                                           |

For one routed expert:

```text
gate/up: 2 * 2048 * 512 parameters
down:        2048 * 512 parameters
total:    3,145,728 parameters
BF16:             6 MiB
ideal 4-bit:    1.5 MiB, before scales/biases/alignment
```

All main routed experts are therefore exactly `40 * 256 * 6 MiB = 60 GiB` in
BF16. With no cache hits, decode requests `40 * 8 * 6 MiB = 1.875 GiB` of BF16
expert payload per token, or an ideal 480 MiB/token at 4-bit. This is the central
I/O number. Sparse activation reduces arithmetic, but useful speed is possible
only if quantization, cache hits, and overlapped reads reduce the _exposed_
bytes per token.

The 4.560 GiB non-routed BF16 text region also disproves the idea that the whole
trunk can remain BF16 on an 8 GB Mac. Its ideal lower bounds are 2.280 GiB at
8-bit and 1.140 GiB at 4-bit, before group scales, allocator padding, temporary
buffers, tokenizer/runtime memory, recurrent state, and macOS headroom.
Selective higher precision is possible only after a measured budget exists.

### 3.2 State growth

Only the 10 full-attention layers have a conventional KV cache. At BF16 and
batch 1, the theoretical payload is:

```text
per token = 10 layers * 2 (K,V) * 2 KV heads * 256 dims * 2 bytes
          = 20 KiB/token
```

That is about 640 MiB at 32K tokens, 1.25 GiB at 64K, 2.5 GiB at 128K, and
5 GiB at 256K, excluding capacity rounding and allocator overhead. The 30
Gated DeltaNet layers instead maintain fixed-size convolution/recurrent state;
the exact MLX allocation must be measured because implementation dtype and
layout, not just config fields, determine the real footprint. The expected
order of magnitude for the matrix state is roughly 60 MiB at batch 1 if it is
FP32 (`30 * 32 * 128 * 128 * 4` bytes), plus small convolution state.

Consequently, context length is a configured memory contract. Native 262K
support does not mean 262K is feasible on the 8 GB target.

### 3.3 Pinned 4-bit runtime artifact (local measurement, 2026-09-01 SGT)

Stage 1 converted source revision
`1e20fd8d42056f870933bf98ca6211024744f7ec` into the text-only
`vference.runtime-model.v1` artifact on the target machine. The converter
excluded the vision tower without changing any retained tensor payload.

| Region | Tensors/records | Payload/file bytes |
| --- | ---: | ---: |
| Text core payload | 1,397 tensors | 1,378,869,376 |
| `core.safetensors` including header | 1 file | 1,379,054,296 |
| Routed experts | 360 stacked source tensors / 10,240 records | 18,119,393,280 |
| One expert record | 9 quantized components | 1,769,472 |

The expert record is exactly 432 4-KiB blocks and contains the existing affine
4-bit weights, BF16 scales, and BF16 biases without requantization. A full
independent verification reread the source, reconstructed every stacked expert
tensor from all 10,240 records, checked all 1,757 retained source tensors, and
reported zero failures. The pack SHA-256 is
`9b26ebdf13a2863d607e1c3ea6e2528ff0267b7f472f8fd37fd20d3adb497e18`.

## 4. Upstream MLX behavior and required seam

MLX is a strong base because it uses unified memory, lazy evaluation, dynamic
graphs, GPU/CPU streams, and custom Metal kernels. The relevant upstream facts
are documented in [MLX lazy evaluation](https://github.com/ml-explore/mlx/blob/main/docs/src/usage/lazy_evaluation.rst),
[streams](https://ml-explore.github.io/mlx/build/html/usage/using_streams.html),
[custom Metal kernels](https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html), and
[custom extensions](https://ml-explore.github.io/mlx/build/html/dev/extensions.html).

As verified at MLX-LM commit
[`77c33b1`](https://github.com/ml-explore/mlx-lm/tree/77c33b14373ac70d7abd6f82af15962852adadbb),
Qwen3.5 MoE sanitization transforms the checkpoint's stacked
`experts.gate_up_proj` and `experts.down_proj` tensors into `SwitchGLU` arrays.
The standard loader then evaluates all model parameters unless `lazy=True`.
The standard `SwitchGLU` contract still presents all 256 experts as normal
model parameters. Lazy loading by itself is therefore not an expert cache.

vference needs a narrow model seam:

1. The router remains the upstream MLX computation.
2. `SparseMoeBlock` returns or exposes exact top-K indices and routing weights.
3. Expert execution calls an `ExpertStore.execute(layer, x, indices)` operation
   rather than indexing a permanently resident 256-expert tensor.
4. The shared expert and router stay as ordinary resident MLX modules.
5. A native MLX extension/custom primitive owns stable slot arrays and launches
   fused dequantize-plus-SwiGLU/matvec work without rebuilding an MLX module or
   a K-expert tensor on every layer.

The Python-level version is acceptable for a correctness prototype. The steady
state should be native because Python futures, per-expert MLX array creation,
frequent `mx.eval`, and allocator churn can erase any I/O overlap. A custom MLX
primitive is appropriate only after the behavior is correct; it must integrate
with lazy scheduling and stream dependencies rather than reaching around them.

Pin MLX and MLX-LM revisions in each benchmark artifact. Current upstream issue
reports show that hybrid-cache correctness, speculative decoding, descriptor
growth, and decode throughput can change between releases. Treat dependency
updates as performance and correctness changes, not routine package bumps.

## 5. Runtime architecture

### 5.1 Offline artifact builder

The official safetensors layout contains whole stacked expert tensors as large
as 1 GiB, optimized for conventional loading rather than individual experts.
The artifact builder should stream source ranges and write:

```text
runtime-model/
  manifest.json
  core-*.safetensors       # text-only, non-routed, selected quantization
  experts.pack            # aligned fixed/known-size expert records
  experts.index            # (layer, expert) -> offset, length, format, digest
  tokenizer files
```

Each expert record contains its gate, up, and down weights plus quantization
metadata in the order expected by the fused kernel. Records should be at least
4 KiB aligned; the pack should support coalesced/read-vector requests for the
eight experts of a layer. A single pack minimizes file descriptors, while the
index permits the format to evolve. The builder must operate shard-by-shard so
conversion itself never requires the whole model in memory.

`manifest.json` records source repository and revision, source file hashes,
config hash, tokenizer hash, tensor inventory, quantization mode/group size per
tensor class, byte order, alignment, converter version, and output hashes.
Preparation fails closed on any unknown/missing tensor.

Format v1 is now implemented for the Qwen3.5 adapter. Records are ordered by
layer, then expert ID. Within a record, gate/up/down weight, scale, and bias
components have fixed indexed offsets. Conversion reads each stacked source
tensor sequentially and writes expert slices directly to their final offsets,
so it does not create a source-shard copy or hold a layer in memory. Output is
built under a `.partial` name and atomically published only after the core,
pack, index, support files, tensor digests, and whole-file hashes complete. The
builder refuses internal placement if the estimated completed artifact would
leave less than 30 GiB free.

### 5.2 Memory controller

Memory is divided explicitly:

```text
safe process budget B
  = resident core P
  + KV/DeltaNet state S(context)
  + expert slots C
  + I/O staging I
  + MLX temporaries/allocator cache T
  + native/Python/tokenizer overhead H
  + safety margin M
```

Solve for `C`; never assume it. `B` is below physical RAM and adapts to current
pressure. Apple documents `os_proc_available_memory()` as advisory and
changeable, so it should inform gradual shrinking, not be cached or used to
consume every available byte. MLX's active/peak memory counters, device
`max_recommended_working_set_size`, process footprint/RSS, swap activity, and
memory-pressure events are all telemetry inputs.

Reserve state for the configured maximum context at session creation. If that
reservation leaves too little expert cache, reject or lower the requested
context before inference; do not discover the conflict halfway through a chat.
On pressure, cancel low-priority prefetches, release staging buffers, shrink
expert slots, and clear unused allocator cache in that order. Never evict live
state or a pinned/in-flight expert.

### 5.3 Expert slot pool

Use stable, preallocated slot buffers addressable by the Metal kernel. Maintain
an authoritative mapping `(layer, expert_id) -> slot_id` and a state machine:

```text
EMPTY -> LOADING -> READY -> IN_USE -> READY -> EVICTING -> EMPTY
```

Each slot also has a generation counter, pin/use count, last/next-use metadata,
frequency score, digest/format metadata, and completion event. Generation
counters prevent an asynchronous read for an old occupant from publishing into
a reused slot. The demand path deduplicates concurrent requests for the same
expert. Eviction is allowed only from READY with a zero pin count.

A purely per-layer cache is simple but strands capacity when layers have
different locality. A fully global pool is efficient but needs guardrails so
one layer cannot evict the minimum working set of another. Start with an
8-slot-per-layer protected floor (enough for one decode step), then lend the
remaining global slots according to trace-derived miss cost and reuse. This is
a proposal to benchmark, not a predetermined winner.

### 5.4 Storage and I/O

The native I/O engine issues aligned `pread`/`preadv` work on a bounded queue,
deduplicates demand and prefetch requests, and supports cancellation before a
read starts. Demand reads outrank predictions. Multiple selected records should
be coalesced when offsets make that cheaper than separate operations.

macOS offers `mmap`, `madvise`, `MAP_NOCACHE`, and `F_NOCACHE`, but they are
hints with different page-cache behavior, not a replacement for the explicit
slot policy. Benchmark at least:

- normal buffered `pread` into reusable aligned slots;
- `F_NOCACHE` reads to avoid duplicating the explicit cache in filesystem cache;
- read-only mapping plus `MADV_RANDOM`/`WILLNEED`/`DONTNEED`.

Apple recommends aligned buffers for uncached reads and documents mapped I/O
for random access. Benchmark the internal SSD using the actual vference access
pattern: cold and warm expert-sized reads, coalesced reads, sustained reads, and
end-to-end exposed expert stall time. Do not use advertised sequential bandwidth
as the runtime model.

The storage policy for the current machine is:

- `/Volumes/veer/vference/models/source/qwen3.5-35b-a3b-4bit` remains the
  authoritative downloaded source checkpoint.
- `VEER` holds inactive experiment output, alternate packs/layouts,
  quantizations, and overflow that is not latency-critical.
- The current repacked expert store and other inference-critical artifacts
  preferentially live on the internal SSD.
- Internal placement is admitted only while a configured reserve remains for
  macOS, swap, normal applications, conversion temporaries, and safe runtime
  operation. Move inactive artifacts back to `VEER` as needed.
- The USB-connected external SSD is not a required performance dependency for
  the main runtime. Do not benchmark it again unless internal capacity becomes
  insufficient or another concrete design requires it on the inference-critical
  path.

The already-measured USB2 result remains in experiment history because failed
or ruled-out approaches are evidence, not because it is the selected runtime
path.

#### Target internal-SSD measurement (2026-09-01 SGT)

These measurements used the verified 18,119,393,280-byte `experts.pack` above,
not a synthetic file or copied source shard. Hardware was Mac14,2, Apple M2,
8 GiB RAM, internal APFS `/dev/disk3s5` over Apple Fabric, macOS 26.6.2. Each
result is recorded in `experiments/results.jsonl` at Git commit `3068cbf` with
the pack SHA-256 and free-space snapshot.

| Access pattern | Mode | Total read | Throughput | p50 / p95 request latency |
| --- | --- | ---: | ---: | ---: |
| 512 random one-expert reads (1,769,472 B) | `F_NOCACHE` | 905,969,664 B | 1.451 GB/s | 1.204 / 1.315 ms |
| 64 random contiguous eight-record reads (14,155,776 B) | `F_NOCACHE` | 905,969,664 B | 1.670 GB/s | 8.504 / 8.817 ms |
| 64 sequential 64-MiB reads | `F_NOCACHE` | 4 GiB | 1.643 GB/s | 39.584 / 49.041 ms |
| Random one-expert trace, buffered pass 1 | buffered | 905,969,664 B | 1.433 GB/s | 1.250 / 1.374 ms |
| Identical immediate warm repeat | buffered | 905,969,664 B | 4.018 GB/s | 0.428 / 0.565 ms |

The contiguous eight-record result is an upper bound, not an assumption that a
router's eight IDs are adjacent in the pack. The one-record random result is
the conservative storage baseline for exact routed misses. Exact K=8 requests
`40 * 8 * 1,769,472 = 566,231,040` expert bytes/token at a zero-percent hit
rate. Therefore the measured storage-only zero-hit ceiling is about 2.56 tok/s
before model compute, router synchronization, scheduling overhead, or thermal
effects. This does not yet constitute an end-to-end throughput claim. It shows
that the 2.0 tok/s provisional goal is physically plausible but requires cache
reuse and/or useful I/O overlap; the synchronous runtime and oversubscription
baselines still determine whether the target and required improvement threshold
are finally frozen.

Track physical/read-request bytes separately from logical expert bytes. A cache
hit has zero expert-file I/O; a correct speculative prefetch that is later
consumed hides latency but still costs physical bytes; a wrong prefetch is
write amplification against the cache and SSD.

### 5.5 Decode pipeline

Autoregressive dependencies limit how much work is truly independent. For each
layer:

1. MLX executes its Gated DeltaNet or full-attention block and normalization.
2. MLX computes router logits and exact top-8 indices/weights.
3. The small routing result becomes visible to the cache manager. This is a
   necessary control-flow synchronization point unless routing and demand
   submission are integrated into a native primitive.
4. The cache manager pins hits, raises demand reads for misses, and waits only
   for missing selected experts.
5. The fused Metal operation reads stable slots, applies the artifact's exact
   dequantization and SwiGLU projections, combines outputs in reference order,
   and adds the resident shared-expert result.
6. Slots are unpinned after GPU completion, not merely after command encoding.

While step 5 runs, CPU/native threads can read predictions for later layers.
However, their exact routers depend on hidden states that do not exist yet.
Prediction must therefore use trace history or a separately validated predictor;
it cannot pretend future exact routes are already known.

### 5.6 Prefetch and eviction

Start with policies whose errors affect performance only:

- static per-layer popularity learned from a representative calibration set;
- prompt/session exponentially weighted frequency;
- cross-layer transition tables conditioned on the current layer's expert set;
- adjacent-token history for the same layer;
- a next-use/least-stale score when route traces demonstrate predictable reuse.

Do not assume LRU or LFU is good. Apple's 2026 SpecMD study reports that MoE
access does not generally follow ordinary temporal-locality assumptions and
shows a least-stale policy outperforming LRU in its evaluated regime. MoE-Infinity
and HOBBIT provide useful activation-trace, prefetch, mixed-precision, and
multi-timescale cache ideas, but their hardware/models differ from this target.
Replay real Qwen3.5 route traces through candidate policies before integrating
them.

Prefetch has an explicit byte and concurrency budget. It yields immediately to
demand I/O and must not evict an expert whose predicted next use is sooner than
the candidate's. Report precision, recall, useful-prefetch rate, late-prefetch
rate, cache pollution, and physical bytes/token—not just hit rate.

The first offline replay implements static per-layer popularity and a
prefill-trained cross-layer transition table under an intentionally optimistic
assumption: predicted reads finish before the next layer demands them. On the
repetitive 7,936-token qualification trace with eight slots per layer, a
one-record transition budget avoided 5,133 of 71,027 exposed misses, achieved
86.2% useful-prefetch reads, and used 0.5% fewer total physical reads. A
two-record budget avoided 10,022 misses with 0.9% read amplification. This did
not generalize to a 35-token technical prompt with 16 slots per layer: the
one-record policy added 40 exposed misses and 4.9% physical reads, while the
two-record policy avoided only 201 misses at 10.0% amplification. Static
popularity polluted both caches. That fixed table was rejected.

A causal online-adaptive version then trained the same transition counts only
after each exact target route became known. With a one-record budget it reduced
exposed misses on all six tested traces: four 128-token code, Japanese, JSON,
and reasoning cases plus the two earlier traces. Worst-case ideal-replay read
amplification was 3.33%; useful-prefetch rate ranged from 51.2% to 86.4%. This
cleared the offline gate for a bounded live experiment.

The live implementation uses one worker and one 1,769,472-byte CPU staging
record. Prediction never writes an MLX slot or evicts a resident expert. Only
an exact later router request can publish staged bytes; wrong or failed reads
are discarded and synchronous exact demand remains the fallback. Against a
contemporaneous 128-token code control, two prefetch runs averaged 3.78% more
throughput and 6.31% lower p95 latency with 2.56% more physical bytes. At 8K it
improved throughput 4.40%, mean latency 4.21%, and p95 3.31%, with 0.47% read
amplification, identical output tokens, and no swap growth. The 8K prefetch run
measured 1.944 tok/s versus a 1.862 tok/s control; both are below the 2 tok/s
absolute gate. The Mac was observed on battery after the runs, but power source
is benchmark metadata rather than a separate qualification target or an
assumed cause of the difference from historical measurements.
`adaptive_cross` with budget one remains opt-in and the default remains no
prefetch until it passes the sustained gate in the Mac's current operating
state. See
`experiments/runtime/stage4-prefetch-replay-2026-09-01.json` and
`experiments/runtime/stage4-live-prefetch-2026-09-01.json`.

The live policy now accepts an explicit bounded staging budget. With two
records, two output-exact 8K/256-token runs measured 2.159 and 2.495 decode
tok/s (mean 2.327), versus the contemporaneous no-prefetch result of 1.862
tok/s. Mean physical-read amplification versus that control was 1.34%, and
82.7% of completed speculative records were useful. The captured repetition
reported 193.682 seconds model-ready TTFT, 395 ms median and 432 ms p95 decode
latency, 2.903 GB peak MLX memory, an 8.4 MB decrease in swap occupancy, and
6.6 MB of system-wide swap-out. Both runs produced the accepted canonical
256-token hash, and the five-domain corpus again had exact tokens and zero
initial-logit error. The earlier repetition increased swap occupancy by 145.6
MB and throughput varied materially, so budget two remains opt-in rather than
becoming the default. See
`experiments/runtime/stage4-prefetch-budget2-2026-09-01.json`.

Unrestricted budgets three and four are rejected without live testing. On the
code trace, budget three made exposed misses worse than no prefetch and raised
physical reads 22.2%; budget four raised them 36.3%. The selected refinement
instead requires eight causal transition observations before publishing a
prediction candidate (the score threshold scales with route width, so this is
64 for Qwen's top-8 router). With budget two this reduced exposed misses on all
six stored traces and capped worst-case ideal-replay amplification at 5.07%; it
did not suppress any prediction on the strongly trained 8K trace. Three exact
128-token code runs measured 2.731, 2.872, and 2.466 tok/s (mean 2.690, sample
standard deviation 0.206), 14.5% above the no-prefetch control and 10.3% above
the budget-one mean. Mean physical-read amplification was 3.13%, every run had
flat swap occupancy, and the five-domain corpus again had exact tokens and zero
logit error. Eight observations is now the safer default threshold when
adaptive prefetch is explicitly enabled; no prefetch remains the runtime
default because performance still varies with host state. See
`experiments/runtime/stage4-confidence-gated-prefetch-2026-09-01.json`.

### 5.7 Prefill is a separate operating mode

Prefill routes many prompt tokens at once and can touch a much wider union of
experts than single-token decode. Treating it like decode may overflow activation
memory or thrash every slot.

The proposed bounded path processes prompt chunks, obtains exact routes, groups
token rows by expert, and evaluates experts in expert-major microbatches through
a transient slot bank. Output accumulation must reproduce reference route rank
order. Prefill reports time-to-first-token, peak memory, expert union size, and
bytes read separately from decode throughput. Cache seeds retained at the
prefill/decode boundary should be selected from the tail of the route trace and
the decode policy, not merely the most frequent expert over the entire prompt.

## 6. Quantization policy

The first practical artifact is the pinned MLX community 4-bit conversion. Use
its existing representation for both the resident trunk and routed experts,
because it provides a conventional fully resident reference on a larger-memory
Mac and a realistic 8 GB budget. Do not begin with BF16 acquisition, a novel
asymmetric recipe, and an untested runtime simultaneously.

After the exact streaming path passes, evaluate sensitivity by tensor class:
embeddings, LM head, routers, shared experts, attention/DeltaNet projections,
and routed experts. Candidate mixed precision is admitted only if it fits the
budget and passes the BF16 quality suite. Router weights deserve particular
care because small perturbations can change discrete expert selection. Keeping
routers at 8-bit or BF16 is cheap (40 MiB BF16 total) relative to the model and
is a sensible initial choice.

Store quantized experts in the representation consumed by the fused Metal
kernel. Expanding a 4-bit expert to FP16 in a staging buffer doubles traffic and
adds memory. Prefer fused unpack/dequantize-matvec, with scales read once and
weights never materialized as a full FP16 expert.

## 7. Core ML and the Neural Engine

Core ML is an experiment after the MLX/Metal baseline, not a premise required
for feasibility. Apple exposes ANE through Core ML compute-unit choices; there
is no supported public API for arbitrary ANE kernels. `MLComputeUnits.all` lets
Core ML choose CPU/GPU/ANE, while `cpuAndNeuralEngine` excludes the GPU but still
does not promise that every operation executes on ANE. Core ML dynamically
partitions graphs. Use Xcode performance reports and `MLComputePlan` to inspect
supported/preferred devices and measured operation cost.

Good candidates have fixed or enumerated shapes, dense operations, meaningful
work per invocation, and minimal boundary traffic. Decode shape `(1, 1, 2048)`
may be too small to amortize Core ML dispatch; prefill buckets may be better.
Apple recommends `EnumeratedShapes` for finite optimized shapes. Stateful ML
Programs are available from macOS 15, but duplicating KV/recurrent state between
MLX and Core ML would violate the memory budget.

The decoder is mostly sequential: an ANE subgraph whose output feeds the next
GPU subgraph is not independent work. Any claimed overlap must identify a real
dependency-free interval. Crossing MLX/Core ML boundaries may introduce copies,
format conversion, dispatch latency, weight duplication, or unified-memory
contention. A Core ML candidate is accepted only when an end-to-end layer/token
benchmark, including boundary costs and peak memory, wins over MLX and passes
the same intermediate/logit tests. ANE numerical differences are not exempt
from the quality invariant.

Primary references:
[MLComputeUnits](https://developer.apple.com/documentation/coreml/mlcomputeunits),
[Core ML typed execution](https://apple.github.io/coremltools/docs-guides/source/typed-execution.html),
[flexible shapes](https://apple.github.io/coremltools/docs-guides/source/flexible-inputs.html),
[stateful models](https://apple.github.io/coremltools/docs-guides/source/stateful-models.html), and
[MLComputePlan/performance reports](https://developer.apple.com/videos/play/wwdc2024/10161/).

## 8. Correctness and quality qualification

### 8.1 Reference hierarchy

Maintain three clearly named references:

1. **BF16 quality reference:** official text checkpoint on hardware that fits it.
2. **Quantized model reference:** standard fully resident MLX-LM using the exact
   candidate converted artifact.
3. **Streaming runtime candidate:** vference using that same artifact.

BF16 vs quantized measures model conversion quality. Quantized resident vs
streaming measures runtime correctness. Never combine the two deltas.

### 8.2 Layerwise oracle tests

For fixed prompts/seeds, capture or compare online:

- router logits, exact top-8 indices, route weights, and route order per token/layer;
- cache hit/miss outcome (informational only) and expert record digest;
- each routed expert output before weighting;
- shared-expert output and combined MoE output;
- post-layer hidden state;
- final logits, sampled token, and recurrent/KV state checksums.

Lossless BF16 repacking should be bit-identical when executing the same kernels
and order. Quantized/custom fused kernels may require tight dtype-aware numeric
tolerances, followed by exact greedy-token equality over long generations.
Tolerance alone is insufficient if token sequences diverge.

#### Stage 2 evidence on the target (local measurement, 2026-09-01 SGT)

The correctness-first Python path now replaces only `switch_mlp`; upstream MLX
still computes router softmax, exact top-8 IDs and scores, shared-expert output,
route-weighted reduction, and layer order. The store blocks on exact records and
performs no prediction, substitution, or top-K change.

Two real-weight expert-operation oracles passed bit-for-bit against MLX
`gather_qmm` using slices read independently from the source checkpoint:
layer 0 with expert IDs 0–7, and layer 17 with shuffled IDs
`[255, 17, 203, 42, 99, 0, 128, 7]`. Each compared 16,384 BF16 output elements;
both maximum and mean absolute error were zero. The synthetic forced-eviction
unit test is also bit-exact.

Loading the resident core alone measured 1,378,869,384 MLX active bytes and did
not read or materialize any expert records. A one-token, eight-slot forced-miss
forward used all 320 exact expert routes, read 566,231,040 logical bytes, peaked
at 1,464,980,934 MLX bytes, and completed in 1.83 seconds including first-use
costs.

A subsequent short warm chat smoke run with a 320-record cache and thinking
disabled generated `Hello from vference!<|im_end|>` for the instruction to
reply with exactly that phrase. It measured 2.40 decode model calls/s over only
five calls, 2,038 expert-cache hits, 6,282 misses, and 2,018,563,520 peak MLX
bytes. This proves end-to-end operation but is **not** the acceptance result: it
used a 21-token prompt, buffered reads, and too few output tokens to establish
sustained speed, 8K state behavior, thermal stability, or physical bytes/token.
The detailed record is `experiments/runtime/stage2-smoke-2026-09-01.json`.

Attempting to evaluate an entire original 256-expert stacked layer directly
from the external source shard produced a Metal command-buffer GPU timeout on
the 8 GB machine. The successful oracle instead reads the selected source
slices into a small resident stacked reference before calling `gather_qmm`.
This failed attempt is retained in the correctness experiment record and is
further evidence that ordinary full-layer faulting is not a viable runtime.

#### Sustained synchronous baseline (local measurement, 2026-09-01 SGT)

With a 35-token chat prompt, 256 generated tokens, `F_NOCACHE`, an initially
empty 320-record explicit LRU, and no prefetch or asynchronous I/O, decode
sustained 2.400 model calls/s. Median latency was 410 ms and p95 was 512 ms;
peak MLX memory was 2.024 GB. The run made 92,800 exact expert requests, hit
28.9%, and physically read 116.70 GB. Of 116.83 seconds inside expert
execution, measured `pread` occupied 83.23 seconds (71.2%), router/previous
graph waits 23.08 seconds (19.8%), and record-to-MLX materialization 9.30
seconds (8.0%). Effective expert-read bandwidth was 1.402 GB/s, consistent with
the standalone internal-SSD result. `pmset` reported no thermal or performance
warning after the runs.

This clears the provisional 2.0 tok/s number for sustained short-context decode,
but not the complete acceptance gate: context was only 35 prompt tokens, no
pre-run swap snapshot was captured, and one run does not establish confidence
intervals or 8K context behavior.

For runs at and after commit `5cf71cf`, record both system-wide swap occupancy
and cumulative swap-in/swap-out byte deltas. They answer different questions:
occupancy shows whether the run left more data in swap, while the I/O counters
show traffic during the window. Both are global macOS counters, not bytes
attributable solely to vference. A one-token telemetry smoke test measured
1.43 MB of swap-out with exactly zero occupancy change, so nonzero swap I/O by
itself is not evidence that the model exceeded its memory budget. The frozen
acceptance wording "without swap growth" continues to mean no positive
before/after occupancy delta; swap I/O and available-memory snapshots are
reported as supporting pressure diagnostics rather than silently replacing
that gate.

The identical route trace replayed at larger LRU capacities predicted 59.2%
hits at 1,024 records instead of 28.9% at 320. End-to-end validation preserved
all 256 greedy tokens and every route and reduced physical reads from 116.70 GB
to 66.96 GB. Nevertheless throughput improved only from 2.400 to 2.424 tok/s,
while peak MLX memory rose to 3.270 GB and p95 latency worsened to 608 ms.
Materialization time increased from 9.30 to 34.46 seconds, offsetting the 28.45
seconds saved in `pread`. Therefore a larger cache of independently allocated
MLX expert arrays is rejected as the primary optimization. Stable preallocated
slots/native execution are required before spending substantially more RAM on
expert residency.

#### Stable native-slot results (local measurement, 2026-09-01 SGT)

A nanobind/C++ extension now owns fixed MLX component pools and issues one
`preadv` per missing record directly into the nine final typed slot regions.
The Python scheduler maps exact `(layer, expert)` identities to those slots;
MLX `gather_qmm` executes the unchanged 4-bit affine expert operation. Before a
slot is overwritten, evaluation of the dependent router result provides the
required completion barrier. The cache protects every expert in the currently
known top-8 route while selecting victims. It never substitutes or remaps an
expert, and an oversized multi-token working set falls back to exact tokenwise
execution.

The native path passed a real-weight layer-0 oracle bit-for-bit for experts
0–7: all 16,384 BF16 output elements matched, with zero maximum and mean error.
A synthetic forced-eviction test also verified stable pool addresses and exact
outputs across slot reuse. End-to-end greedy runs at capacities 320, 640, and
1,024 produced the same 256 output token IDs as the Stage 2 Python reference
(token-list SHA-256
`e3a136ca3c3eb2c71a59d6df4a6829a6ec58c3f1ad7be35ea482b1200315aaa4`).

Using the same 35-token prompt, 256 generated tokens, `F_NOCACHE`, and an empty
cache, the measured capacity sweep was:

| Stable slots | Pool bytes | Hit rate | Physical reads | Decode tok/s | p50 / p95 | Peak MLX | Observed system swap delta |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 320 | 566.23 MB | 39.1% | 100.08 GB | 2.408 | 410 / 531 ms | 2.023 GB | -8.4 MB |
| 640 | 1.132 GB | 48.2% | 85.11 GB | 2.672 | 367 / 497 ms | 2.590 GB | -8.4 MB |
| 1,024 | 1.812 GB | 59.2% | 66.93 GB | 3.046 | 325 / 442 ms | 3.269 GB | +499.1 MB |

The 640-slot run improved throughput 11.3% over the capacity-320 Python
baseline without measured swap growth. The 1,024-slot run improved 26.9% over
that baseline and 25.6% over the same-capacity Python cache, but its larger pool
coincided with 499 MB of additional system-wide swap. Because swap is a global
counter and the short prompt reserves almost no long-context state, this is an
observation rather than proof of direct attribution. Capacity 1,024 must not be
made the 8 GB default until the admitted-context memory controller and an 8K
run show adequate headroom.

Stable slots reduced materialization time to zero. At capacity 1,024, `preadv`
occupied 49.84 seconds and the router/previous-graph completion barrier 40.35
seconds of 90.74 seconds inside expert execution. This establishes that the
larger stable cache is useful, while also identifying synchronization and
unoverlapped reads—not MLX-array construction—as the next exposed bottlenecks.
The complete reproducible record is
`experiments/runtime/stage3-stable-slots-2026-09-01.json`.

An exact trace replay then showed that dividing 640 slots evenly into 16 slots
per MoE layer should raise hit rate from 48.2% to 51.5%, because a global LRU
otherwise lets traversal of later layers displace the earlier layers' working
sets. The live run matched the replay exactly: 47,802 hits, 44,998 misses, and
79.62 GB read. It sustained 3.262 tok/s with 301 ms median and 407 ms p95
latency, 2.590 GB peak MLX memory, no observed swap growth, and all 256 output
tokens unchanged. This is 35.9% faster than the original capacity-320 Python
baseline and 7.1% faster than the 1,024-slot global native cache while using
680 MB fewer expert slots. The layer-partitioned 640-slot policy is therefore
the current best short-context decode configuration, subject to the 8K memory
gate. See `experiments/runtime/stage3-layer-partition-2026-09-01.json`.

Bounded prefill is now explicit. When a chunk routes to more unique experts
than its cache or per-layer partition can hold, the runtime divides consecutive
tokens into maximal groups whose exact expert union fits, evaluates each group
before slot reuse, and concatenates the outputs in original order. A 35-token
chunk-size-32 regression preserved all greedy tokens and reduced prefill from
14.96 to 6.95 seconds. A separate synthetic 512-token, single-chunk stress
probe completed in 14.00 seconds with 2.568 GB peak MLX memory and no observed
swap growth. These are feasibility measurements, not the required meaningful
8K retrieval/quality evaluation.

The first 8K-total feasibility workload used 7,936 synthetic repeated raw
prompt tokens followed by 256 greedy output tokens. With 640 slots and no MLX
allocator-cache clearing it decoded at 2.388 tok/s but increased system swap by
1.96 GB, so that configuration was rejected. Reducing to 320 slots, clearing
only unused allocator buffers after each completed 512-token prefill chunk,
then repartitioning from global prefill slots to eight slots per layer for
decode produced the exact same 256 tokens. It measured 191.75 seconds prefill,
2.281 decode tok/s, 433 ms median and 482 ms p95 decode latency, 2.903 GB peak
MLX memory, and a -151 MB system swap delta. `pmset` reported no thermal or
performance warning.

Three identical runs of the passing configuration measured 2.2812, 2.2767,
and 2.2754 decode tok/s (mean 2.2778, sample standard deviation 0.0031,
small-sample 95% t-interval 2.2702–2.2854). All produced identical token IDs,
the same 2.903 GB peak MLX memory, negative swap deltas, and no thermal warning.
The two memory policies also produced identical token-list SHA-256
`f4b04a69f0bb425f65d7103580aedf78b786bc324b6c2100083314a9a76b88bd`.

A deterministic retrieval probe then placed `ORCHID-7319` one quarter into a
7,936-token context and asked for the code at the end. The response began
exactly `ORCHID-7319`; prefill took 198.86 seconds, peak MLX memory was again
2.903 GB, and swap decreased 50 MB. This is meaningful long-range state use,
though still a synthetic regression rather than a broad quality corpus.

The provisional absolute 8K throughput/no-swap gate is therefore repeatably
met. Admission control now derives the exact Qwen cache allocation: 64,389,120
fixed bytes for 30 DeltaNet layers plus 5,242,880 bytes for every 256-token KV
allocation step across 10 attention layers. It adds current resident MLX bytes
and a conservative chunk-transient reserve fitted above the measured 512- and
2,048-token probes. The default budget is 37.5% of physical memory (3 GiB on
this Mac). The passing configuration estimated 3,183,894,664 bytes and was
admitted; the previously swap-heavy 640-slot 8K configuration estimated 3.49
GiB and was rejected before prefill. The estimator exactly matched measured
state allocations at 1, 513, and 8,192 tokens.

Full release qualification still requires the wider deterministic and
stochastic regression corpus plus failure injection. See
`experiments/runtime/stage5-8k-feasibility-2026-09-01.json`.

The first split-state and cross-domain runtime corpus now compare the stable
native store directly with the Python exact-expert reference under identical
MLX operations. A two-segment prefix followed by 32 greedy tokens had zero
initial-logit error, identical 69,632,000-byte cache states, and identical
continuations. Five chat cases—plain exact response, Python code, Japanese,
compact JSON, and arithmetic—each had zero initial-logit error and exact output
tokens. This confirms that storage/cache implementation does not worsen these
outputs; it does not replace BF16-versus-quantized quality evaluation. The
tracked corpus is `experiments/corpus/runtime-v1.json`, and results are in
`experiments/runtime/runtime-correctness-v1-2026-09-01.json`.

Seeded categorical generation is now part of the architecture-neutral runtime
surface. Greedy remains the default; explicit temperature, top-p, top-k, and
seed parameters use the pinned MLX-LM sampler after model logits are complete
and therefore cannot affect routing or cache behavior. A fixed 64-token sample
matched exactly across the Python expert reference, stable layer cache, forced
100% decode misses, and confidence-gated adaptive prefetch.

The tracked stochastic corpus extends this across explanation, code, Japanese,
structured JSON, and reasoning prompts with five distinct seeds and sampler
settings. For all 158 generated steps, the complete vocabulary-logit vector
was bit-exact between the stable adaptive-prefetch path and the Python
exact-expert reference; every sampled token also matched. Peak MLX memory was
2,633,133,216 bytes and system-wide swap occupancy decreased by 3,801,088
bytes. Exact per-step logits prove that the runtime leaves the categorical
distribution unchanged, which is stronger than failing to detect drift in a
finite frequency test. This is runtime correctness for the pinned 4-bit
artifact, not BF16-versus-quantized quality evidence. See
`experiments/runtime/runtime-stochastic-v1-2026-09-02.json`.

For the frozen same-context baseline, the runtime used the identical safe
7,936-token prefill and then cleared expert identities before every decoded
layer, forcing all exact top-8 experts to be synchronously read. This baseline
uses stable native buffers, so it is stronger than the Python materializing
path, and it preserved all 256 output tokens. It sustained 1.937 tok/s versus
the optimized three-run mean of 2.278 tok/s: a 17.6% improvement. Mean token
latency fell 15.0%.

An otherwise identical prefill-only run isolated 185.16 seconds of expert
execution before decode. Subtracting it from the complete measurements gives
505.1 ms of expert execution per demand-baseline model call and a three-run
optimized mean of 425.8 ms, a 15.7% reduction. Both frozen 10% material-
improvement thresholds therefore pass at the 8K context. Neither policy grew
swap or changed output tokens, and `pmset` reported no thermal warning.

Only 3.13% of sorted within-layer selected-ID pairs were adjacent in the v1
pack. Blindly reading the span between the minimum and maximum of eight routed
IDs would therefore amplify I/O; coalescing must operate on genuinely adjacent
ranges or use a trace-qualified physical reordering.

An attempted per-record CRC32 integrity path was rejected from the default
runtime. The implementation detected injected bit flips and preserved the
accepted 8K workload's exact 256-token output, but its sustained 7,936-token
prefill plus 256-token decode run achieved only 1.824 tok/s and grew swap by
221,642,752 bytes. This fails both provisional gates. A contiguous-record CPU
microbenchmark had predicted 33.68 GiB/s checksum throughput, but that result
did not capture MLX/unified-memory pipeline effects or the host-state drift
seen in contemporaneous A/B runs. The runtime therefore continues to rely on
the complete pack SHA-256 checked while constructing/verifying the immutable
active artifact; it does not checksum records in the token-critical path.
Future integrity work must run asynchronously or otherwise demonstrate the
full sustained gates before adoption. Measurements and the rejected variants
are recorded in
`experiments/runtime/rejected-crc-on-load-2026-09-01.json`.

### 8.3 Runtime release gates

A storage/scheduler change is releasable only if:

- route indices and expert identities match the resident quantized reference;
- greedy output tokens match on the full deterministic regression corpus;
- sampler distribution tests show no statistically meaningful drift for
  stochastic generation with identical logits/tolerances;
- results are invariant across forced cache capacities, forced 100% misses,
  delayed I/O, cache warm/cold states, and disabled prefetch;
- prefill chunk sizes produce the same tokens and within-tolerance states;
- long-context, multi-turn prefix/state handling matches reference behavior;
- corruption, short reads, failed digests, and out-of-memory conditions fail
  explicitly rather than producing output.

The regression corpus should cover plain chat, reasoning, code, multilingual
text, JSON/schema output, tool calls, very short and long prompts, repeated
tokens, and adversarial routes that churn the cache.

### 8.4 Quantization release gates

Choose public, reproducible datasets before tuning. At minimum compare BF16 and
candidate artifacts on held-out language-model perplexity, task accuracy,
instruction/tool-call validity, long-context retrieval, and human/blind output
preference where automatic metrics are weak. Define thresholds before running
the final evaluation. If the project promise is literally “no worse,” prefer a
known community conversion with existing evidence or keep the new recipe
experimental until the confidence interval excludes the agreed degradation.

## 9. Performance model and telemetry

The M2 Air has 100 GB/s advertised unified-memory bandwidth and a 16-core Neural
Engine, but internal SSD performance varies by storage configuration and thermal
state. Measure the actual target. The basic decode bound is:

```text
expert_bytes/token = sum(missed expert record bytes) + wasted prefetch bytes
I/O floor seconds/token = physical expert bytes/token / sustained measured SSD B/s
token latency = max/overlap(compute, useful I/O, preparation) + exposed stalls + sync
```

At ideal 4-bit, 480 MiB/token with zero hits is already demanding. If sustained
random/coalesced reads measure 1.5 GiB/s, the I/O-only ceiling is roughly 3
tokens/s before quantization metadata, read amplification, compute, and stalls.
This is an example, not a claimed M2 SSD measurement.

Record per phase and per layer:

- TTFT/prefill tok/s and decode tok/s (p50/p95, cold and warm);
- logical and physical expert bytes/token;
- demand hit rate, stall time, outstanding reads, read latency/size distribution;
- prefetch usefulness, accuracy, lateness, pollution, and cancellation;
- slot occupancy/evictions and per-layer working-set size;
- MLX active/peak/cache memory, process footprint/RSS, available memory, swap;
- GPU/CPU/ANE time, synchronization count/time, kernel count;
- temperature, power mode, battery/AC, and thermal throttling where observable.

Every benchmark manifest includes hardware identifier, RAM/storage capacity,
macOS version, power state, model/artifact hash, dependency commits, prompt/output
length, sampler, context cache state, and whether filesystem caches were warm.

## 10. Staged implementation plan and go/no-go gates

### Stage 0 — measurement harness

Build a normal MLX-LM reference on a machine that fits the chosen 4-bit model.
Capture per-layer routes, states, tokens, memory, and baseline performance. On
the 8 GB M2, measure aligned random/coalesced reads at the actual expert record
size plus MLX memory overhead for trunk and context.

Run the conventional oversubscription and synchronous-demand baselines defined
in section 1.1. Before implementing cache/prefetch tuning, freeze the sustained
decode target and the minimum numeric improvement required over those baselines;
record hardware, prompt, context, output length, thermal protocol, repetitions,
and confidence intervals in the benchmark manifest.

Go only if the measured resident trunk, minimum state, temporaries, and safety
margin leave a nontrivial expert pool, and the zero-hit I/O ceiling is not below
the project's minimum useful throughput.

### Stage 1 — streaming artifact builder

Using only the pinned MLX 4-bit artifact, produce the text core, expert
pack/index, manifest, and complete hash verifier without exceeding bounded
conversion memory. Reconstruct random and full layers and compare them
byte-for-byte to source tensors.

### Stage 2 — correctness-first Python cache

Patch the MoE seam, use a small synchronous cache, disable prediction, and force
misses. Pass all layerwise and greedy-token comparisons. Speed is irrelevant.

### Stage 3 — stable native slots and fused Metal expert op

Remove per-layer tensor reconstruction and allocator churn. Validate forced
slot reuse, delayed I/O, generation counters, GPU completion pinning, and memory
ceilings. Match the Stage 2 outputs. **In progress:** stable native slots and
direct reads are implemented and exact; the current execution path still uses
MLX `gather_qmm`, and asynchronous ownership plus any custom fused Metal kernel
remain future work justified only by profiling.

### Stage 4 — asynchronous demand, trace replay, and prefetch

Add bounded native I/O and evaluate policies offline before live integration.
Prefetch must increase throughput or reduce p95 stalls without increasing
memory above budget or changing outputs. Keep a no-prefetch baseline in CI.
**In progress:** the offline evaluator now separates exposed demand misses,
useful speculative reads, eviction pollution, and total physical reads. A
fixed prefill-only table failed to generalize and was rejected. Its causal
online-adaptive successor cleared a six-trace gate and now has a bounded,
exact-demand-only live staging implementation. A two-record budget cleared the
absolute 8K throughput threshold in two output-exact runs and passed the
five-domain exact corpus. It remains opt-in because swap occupancy and
throughput varied across host-state samples. A route-width-normalized
eight-observation confidence gate now limits noisy second predictions and
clears the six-trace replay gate plus the live five-domain exact corpus;
broader live and pressure cases remain.

### Stage 5 — bounded prefill and long contexts

Implement expert-major chunked prefill, state reservation, context admission,
and multi-turn correctness. Publish separate TTFT and decode profiles at 4K,
8K, 16K, 32K, and the largest safely admitted context. **In progress:** exact
working-set splitting, phase-specific cache policies, and allocator cleanup are
implemented. One synthetic 8K-total/256-output workload cleared the absolute
throughput and no-swap gate in three repetitions, and a separate 7,936-token
needle-retrieval case passed. Exact Qwen state accounting and measured-profile
admission are implemented. The first multi-turn split-state reference and a
five-domain deterministic corpus pass exactly. Same-context baseline
performance clears both frozen 10% improvement thresholds. Broader cases such
as tool calls, schemas, adversarial churn, corruption/failure injection, and
stochastic sampling remain.

A 256-token prefill-chunk experiment is explicitly rejected even though it
reduced peak MLX memory from 2.903 GB to 2.553 GB and decoded at 2.471 tok/s.
Its greedy output first diverged from the accepted 512-chunk token sequence at
output index 72. A prefetch-disabled 73-token isolation reproduced the same
candidate prefix and divergence, proving the change follows chunked model
execution rather than speculative expert publication. Prefill chunk size is
therefore part of the qualified Qwen execution configuration until the
chunk-boundary state/numerics issue is fixed and exactness is re-established.
The reusable `prefill-chunk-verify` harness reproduces the underlying numerical
non-invariance with only 512 prompt tokens: final logits differ by up to 1.1875
(mean absolute difference 0.1823). Router rank first differs in layer 0 at
token 4, and top-8 membership first differs in layer 0 at token 6 (expert 68
versus 214), before any full-attention layer. The first 16 greedy tokens still
match, demonstrating that token-only smoke checks can miss a router/math
divergence that becomes user-visible later.
The generation entry point now rejects a non-512 multi-chunk Qwen prefill by
default. `--allow-unqualified-prefill-chunk-size` exists only for explicit
correctness experiments such as the verifier; it is not a performance mode.

A phase-separated profile of the selected 320-slot, confidence-gated
two-record prefetch policy then attributed the 8K decode ceiling. Over 255
decode calls it read 99,315,154,944 demand bytes plus 18,712,166,400 prefetch
bytes. Demand `preadv` occupied 69.434 seconds (272.3 ms/call), equivalent to
1.430 GB/s and therefore consistent with the independently measured 1.451 GB/s
internal-SSD random-record result. The router/previous-graph synchronization
path occupied another 40.138 seconds (157.4 ms/call). Expert execution totaled
111.475 seconds inside 116.772 seconds of decode wall time; the run sustained
2.184 tok/s and preserved the canonical token hash. Storage is the largest
exposed decode cost, with serialized router/graph synchronization second.

Phase-specific pool replacement was implemented to test whether decode could
spend memory released after bounded prefill. The first implementation was
rejected before an 8K run: `owned_zeros` intentionally retained every native
allocator buffer for process lifetime, so resizing 320 to 400 slots accumulated
1,274,019,840 active bytes instead of replacing the 566,231,040-byte pool. The
native arrays now own their buffers through MLX's allocator deleter; the same
isolation leaves exactly the 707,788,800-byte 400-slot replacement and returns
to zero after close. Tests cover allocator release and bit-exact expert output
across resize.

The repaired mechanism preserved the accepted 256-token hash at 480 and 560
decode slots, but neither capacity is selected as the 8 GB default. One
560-slot run reduced decode physical bytes 9.43% and sustained 2.258 tok/s, but
grew system-wide swap occupancy by 482,541,568 bytes. Two 480-slot runs reduced
physical bytes by 6.66% and misses by 5.91%; they sustained 2.224 and 2.199
tok/s (mean 2.211), only 1.27% above the phase-profiled 320-slot run. Their swap
deltas were -41,943,040 and +8,323,072 bytes. The extra 283 MB therefore saved
only 2.35% of demand-read time and 1.17% of expert execution time on average;
it does not justify extra pressure or pass the frozen no-swap burden. The
admission-gated resize remains available for machines with more headroom and
future policies, while this 8 GB configuration continues to use 320 slots in
both phases. Full evidence, including the rejected native lifetime design, is
in `experiments/runtime/stage5-phase-cache-resize-2026-09-01.json`.

### Stage 6 — quantization experiments

Only now notify the user that the official BF16 checkpoint is required and ask
before downloading it. Then explore asymmetric precision or perform proper
BF16-versus-4-bit quality evaluation. Qualify each artifact against BF16 and
keep runtime comparisons fixed to the artifact.

### Stage 7 — Core ML/ANE experiments

Profile fixed/enumerated-shape subgraphs, inspect actual placement, and retain
only end-to-end wins that fit memory and pass correctness. The product remains
functional if no ANE candidate wins.

## 11. Main risks and falsification tests

| Risk                                             | Early falsification test                                          | Consequence                                                           |
| ------------------------------------------------ | ----------------------------------------------------------------- | --------------------------------------------------------------------- |
| Non-routed trunk/temporaries leave too few slots | Measure peak with core-only forward and reserved state            | Stronger trunk quantization, shorter context, or target is infeasible |
| Routing has poor cacheability                    | Record diverse Qwen3.5 traces; replay capacities/policies         | Throughput becomes SSD-bound; prediction cannot manufacture locality  |
| Internal storage cannot sustain required small reads | Measure cold/warm expert-sized, coalesced, sustained, and end-to-end stall behavior | Repack/coalesce, revise cache policy, or identify the internal-storage hardware ceiling |
| Router synchronization serializes pipeline       | Instruments per-layer eval/sync and native fused route submission | Integrate routing/demand notification deeper into native MLX seam     |
| MLX allocator duplicates/retains slot data       | Compare slot bytes, active/peak memory, RSS over long decode      | Native stable buffers and strict cache-limit management required      |
| Prefill touches most experts                     | Measure union by chunk/prompt and TTFT bytes                      | Dedicated expert-major prefill is mandatory                           |
| Core ML duplicates weights/state or falls back   | Compute plan + peak memory + end-to-end benchmark                 | Drop ANE path; it is optional                                         |
| Quantization changes routing/quality             | BF16-vs-candidate route and quality suite                         | Keep higher precision for sensitive classes or reject artifact        |
| MacBook Air thermally throttles                  | Sustained 10–30 minute tests on AC/battery                        | Optimize energy/traffic, report sustained rather than burst speed     |
| SSD endurance becomes material                   | Track physical bytes/token and session write volume               | Reads dominate here, but avoid temporary rewrites and swap pressure   |

The project should be willing to conclude that a particular speed/context/quality
combination is infeasible on 8 GB. A measured negative result is better than
silently weakening the model.

## 12. Related systems and lessons to validate

- [MoE-Infinity](https://arxiv.org/abs/2401.14361): request-level activation
  tracing, expert caching, and prefetching across memory tiers.
- [HOBBIT](https://arxiv.org/abs/2411.01433): multi-timescale caching and mixed
  precision. Its lower-precision miss substitution conflicts with vference's
  default quality invariant, but its scheduling measurements remain relevant.
- [Apple SpecMD](https://machinelearning.apple.com/research/specmd-expert-prefetching):
  controlled cache-policy study; important warning against assuming LRU/LFU.
- [Vates](https://github.com/AMOS144/Vates): MLX out-of-core Qwen3-Next system
  using native slots, per-expert blobs, `pread`, fused computation, bounded
  prefill, and route-driven prefetch. Audit its implementation and tests before
  borrowing ideas; do not treat project claims as results for this hardware.
- [Flash-MoE](https://github.com/tayoun/flash-moe): specialized C/Metal Qwen3.5
  expert streaming and runtime-specific repacking. Its published headline uses
  reduced routing K on different hardware, so it is not a quality-equivalent
  baseline for this project, but its packing and kernel work is directly relevant.

These systems demonstrate that the design space is active; they do not prove
Qwen3.5 at exact K=8, chosen quantization quality, useful context, and useful
sustained speed on an 8 GB M2 Air. That is the project’s measurement burden.

## 13. Source and update notes

Primary sources last checked for this revision:

- [Official Qwen checkpoint/config](https://huggingface.co/Qwen/Qwen3.5-35B-A3B)
  at revision `59d61f3ce65a6d9863b86d2e96597125219dc754`.
- [MLX-LM](https://github.com/ml-explore/mlx-lm) at
  `77c33b14373ac70d7abd6f82af15962852adadbb` (reported package version 0.32.0
  in the checked tree).
- [MLX](https://github.com/ml-explore/mlx) at
  `37c26e5755da637255d57ea34b4879196a485301` (documentation reported 0.32.2).
- [Apple M2 MacBook Air specifications](https://support.apple.com/en-us/111867).
- Apple documentation for
  [`os_proc_available_memory`](https://developer.apple.com/documentation/os/os_proc_available_memory),
  [`mmap`](https://developer.apple.com/library/archive/documentation/System/Conceptual/ManPages_iPhoneOS/man2/mmap.2.html),
  [`madvise`](https://developer.apple.com/library/archive/documentation/System/Conceptual/ManPages_iPhoneOS/man2/madvise.2.html), and
  [filesystem caching](https://developer.apple.com/library/archive/documentation/Performance/Conceptual/FileSystem/Articles/FilePerformance.html).

When updating this document:

1. Record the new verification date and exact upstream revisions.
2. Re-run safetensors header accounting if the model revision changes.
3. Label all performance values with machine and benchmark manifests.
4. Keep proposals and published/measured facts visibly distinct.
5. Do not weaken the quality invariant to make a benchmark look better.
