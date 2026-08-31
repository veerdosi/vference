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

Stage 0 must measure storage and baseline limits, then confirm or revise the
2.0 tok/s threshold **before** cache, prefetch, or heterogeneous-compute tuning.
Any revision requires a dated rationale in this document and a fixed benchmark
manifest. “Material improvement” must also receive a numeric threshold at that
point, with confidence intervals; until then it means a repeatable improvement
outside run-to-run noise in both decode throughput and exposed expert-stall
time. A system that merely completes inference but does not meet the frozen
threshold is a feasibility demonstration, not a successful runtime.

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
for random access. The best choice must be measured on the actual internal SSD,
including cold cache, warm cache, random record reads, coalesced reads, and
sustained thermal behavior. Do not use advertised sequential bandwidth as the
runtime model.

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
ceilings. Match the Stage 2 outputs.

### Stage 4 — asynchronous demand, trace replay, and prefetch

Add bounded native I/O and evaluate policies offline before live integration.
Prefetch must increase throughput or reduce p95 stalls without increasing
memory above budget or changing outputs. Keep a no-prefetch baseline in CI.

### Stage 5 — bounded prefill and long contexts

Implement expert-major chunked prefill, state reservation, context admission,
and multi-turn correctness. Publish separate TTFT and decode profiles at 4K,
8K, 16K, 32K, and the largest safely admitted context.

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
| SSD cannot sustain required small reads          | Cold/warm aligned record benchmark over minutes                   | Repack/coalesce or accept a hard throughput ceiling                   |
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
