# vference

`vference` is an experimental, general local inference runtime for running
sparse Mixture-of-Experts models larger than unified memory on Apple Silicon.
Qwen3.5-35B-A3B on an 8 GB M2 MacBook Air is the first architecture adapter and
validation target, not the intended permanent scope.

The generic artifact path is also exercised with a synthetic Qwen3 MoE adapter:
its different tensor prefix and sparse-layer map pass the same lossless pack and
verification machinery. This is a boundary test, not a claim that a second real
checkpoint is already qualified for inference; each architecture still needs a
loader, memory profile, and exact-output suite before being listed as supported.

The project has completed the lossless artifact build, exact streamed-expert
runtime, asynchronous demand path, and current-version long-context
qualification. The runtime loads the 1.38 GB resident core and streams exact
experts into a bounded cache. Both the Python reference cache and the native
stable-slot cache generate with Qwen. Start with
[the technical reference](docs/technical-reference.md). It records the model
facts, memory and I/O arithmetic, proposed runtime architecture, quality
invariants, validation gates, risks, and staged implementation plan.

The non-negotiable rule is that storage and scheduling optimizations must not
change model behavior. A missing expert causes a stall, never expert
substitution, skipping, remapping, or a reduced top-K.

The provisional usability target is at least 2.0 sustained decode tokens/second
at an admitted 8K context, while also materially outperforming both OS swapping
and naïve synchronous expert streaming on the same hardware and model artifact.
The measured internal-SSD baseline is the storage ceiling used to evaluate that
target.

The external `VEER` SSD holds the downloaded source checkpoint and inactive
artifacts. Active repacked experts and other latency-critical runtime files
preferentially live on the internal SSD, subject to a system free-space reserve.

## Install and build

Install the pinned environment. The project build now compiles and installs the
stable-slot extension automatically:

```sh
uv sync --dev
```

The distributable wheel is platform-specific; it is not a pure-Python wheel.
`uv build` must compile and include `vference/native/_vference_native*.so`.
For native-only iteration in an existing checkout, `make -C native build`
remains available.

The reference and native paths can then be selected explicitly with
`vference stream-generate ... --store python` and `--store stable`. The current
best qualified short-context decode configuration is:

```sh
uv run vference stream-generate artifacts/qwen3.5-35b-a3b-4bit-runtime \
  --prompt 'Hello' --max-tokens 64 --store stable \
  --cache-capacity 640 --cache-policy layer --prefill-chunk-size 32 \
  --demand-workers 8 --nocache
```

It remains experimental while heterogeneous-compute and release-reliability
work continues; the current 8K throughput and no-swap gate passes.

For the current 8 GB long-context feasibility configuration, use 320 slots and
bound allocator caching between prefill chunks:

```sh
uv run vference stream-generate artifacts/qwen3.5-35b-a3b-4bit-runtime \
  --prompt 'Your long prompt' --max-tokens 256 --store stable \
  --cache-capacity 320 --cache-policy global --decode-cache-policy layer \
  --prefill-chunk-size 512 --clear-cache-between-prefill-chunks \
  --demand-workers 8 --prefetch-policy adaptive_cross --prefetch-budget 2 \
  --nocache
```

The serial-demand predecessor averaged 2.278 decode tok/s across three identical
8K-total-token runs without swap growth, and a separate 7,936-token needle
retrieval returned the expected code. Unsafe contexts are rejected against a
measured memory model before prefill. Stable slots also match the Python
exact-expert reference with zero logit error on the first split-state and
five-domain deterministic corpus. Against an exact forced-demand 8K baseline,
it improves throughput 17.6% and reduces exposed expert time 15.7%. It is not
yet a release claim; broader long-context and application-surface cases remain.

The native demand queue reads up to eight router-requested records concurrently
into distinct reserved slots, then publishes only the exact requested experts.
On the qualified 8K workload it preserved the canonical 256-token output,
decoded at 2.523 tok/s, and grew no swap. Full-logit application and forced-
churn comparisons also remained exact. Eight workers reached about 1.65 GB/s,
the measured queue-depth-eight ceiling of the internal artifact path.
The CLI therefore defaults the stable store to eight demand workers; use
`--demand-workers 1` only when deliberately reproducing the serial baseline.

Stage 5 is complete for the current-version scope. Context support has three
measured tiers on the 8 GB M2 Air:

- **8K strict:** 2.523 decode tok/s, 2.903 GB peak MLX memory, and zero swap
  growth on the qualified 7,936-prompt/256-output workload.
- **16K practical extended:** a heterogeneous 16,128-token document prompt
  completed in 501.2 seconds of prefill and decoded at 2.520 tok/s. Peak MLX
  memory was 3.076 GB (2.865 GiB); system-wide swap grew 136 MiB.
- **32K rejected for this version:** the exact 512-token-chunk path had not
  completed prefill after about 52 minutes and grew swap by about 1.10 GiB.

The 16K run was successful; its small positive swap delta is why it is an
extended mode rather than the stricter zero-swap tier. A representative
document caused 80.3% more prefill expert reads than repeated text, so long-
context TTFT claims must use realistic content. Use `--prompt-file` with
`--truncate-raw-prompt-to-tokens` and `--raw-prompt-suffix` for reproducible
bounded document workloads.

Runtime artifacts are checked automatically before model load. The first use
hashes every manifest-listed payload and support file; later runs use a
file-identity-bound stamp and rehash automatically if anything changed. To
explicitly rehash the active artifact:

```sh
uv run vference artifact-integrity \
  artifacts/qwen3.5-35b-a3b-4bit-runtime --force
```

The initial failure-injection gate covers corruption, short reads, late and
failed speculative reads, insufficient exact working-set capacity, and unsafe
memory admission. All fail explicitly or fall back to the exact requested
expert; none can silently alter model math.

The application corpus also compares real chat transcripts and Qwen tool
declarations with the exact Python expert reference. Very short, repeated-token,
multi-turn transcript, function-tool, and nested-JSON cases matched every
full-vocabulary logit vector and selected token. The accepted run peaked at
2.362 GB of MLX memory without swap growth.

An additional no-prefetch stress case uses only eight global expert slots. It
incurred a 66.9% miss rate and 43.2 GB of demand reads while constantly
evicting and splitting prefill working sets, yet still matched every reference
logit and output token without swap growth.

Incremental multi-turn cache reuse is deferred to a future version. Stable and
Python expert stores match each other exactly under split-state updates, but
the pinned Qwen/MLX split execution differs numerically from one-pass prefill
(0.7109 maximum initial-logit error in the first boundary probe). Its 64 greedy
tokens still matched. The current safe behavior is stateless transcript
inference: render the complete conversation and prefill it as one request.

`--decode-cache-capacity` can replace the stable expert pool at the synchronized
prefill/decode boundary when a different phase budget is explicitly desired.
It is not enabled automatically on the 8 GB target. Exact 480- and 560-slot 8K
experiments reduced expert traffic, but 560 caused 482.5 MB of swap growth and
480 improved mean throughput only 1.3% while one repetition had positive swap
growth. Keep the 8 GB long-context default at 320 slots; the resize mechanism
is retained for larger-memory machines and future pressure-aware policies.

Generation also supports reproducible categorical sampling with
`--temperature`, `--top-p`, `--top-k`, and `--seed`; temperature zero preserves
the existing greedy default. A five-domain, five-seed corpus matched the Python
exact-expert reference bit-for-bit at every generated-step logit vector and
produced identical sampled tokens under adaptive prefetch. These checks prove
runtime equivalence for the pinned 4-bit artifact, not 4-bit-versus-BF16 model
quality.

Stage 4 also includes an opt-in one-record adaptive cross-layer prefetcher. It
stages bytes in CPU memory and publishes them only when the exact router later
requests that expert; synchronous exact demand remains the fallback. Enable it
with budget one and explicitly set one observation:
`--prefetch-policy adaptive_cross --prefetch-budget 1 --prefetch-min-observations 1`.
It improved a short code workload by 3.8% and the 8K workload by 4.4% with exact
outputs, but the measured 8K result was 1.944 tok/s and therefore remains below
the 2 tok/s gate. Power source is recorded as metadata, not treated as a
separate runtime qualification.

A two-record budget is the current faster experimental setting:
`--prefetch-policy adaptive_cross --prefetch-budget 2`. The default confidence
gate waits for eight causal route observations before staging a candidate.
Three exact short-code runs averaged 2.690 decode tok/s with 3.13% physical-read
amplification, while the stored 8K route retained the same predictions as the
earlier unrestricted budget-two policy. The five-domain corpus retained exact
tokens with zero initial-logit error. Prefetch remains opt-in because host-state
performance still varies. Keep the qualified prefill chunk at 512: a 256-token
chunk changed the output sequence even with prefetch disabled and is rejected
until chunk-boundary invariance is fixed. Non-512 multi-chunk generation
therefore requires the explicit experimental
`--allow-unqualified-prefill-chunk-size` override.

BF16-versus-4-bit quality evaluation and custom quantization are also deferred
to a future version. The current evidence proves that vference preserves the
pinned 4-bit artifact's computation; it does not claim that the artifact is
quality-equivalent to BF16. Do not download the BF16 checkpoint until that
evaluation is deliberately resumed.

Core ML/ANE remains an explicit differentiating workstream, and the first M2
qualification is complete. Native Core ML and `MLComputePlan` proved real ANE
placement for the shared-expert graph, but its whole-runtime upside was below
2%. A larger compressed linear-attention projection graph was assigned to GPU
or CPU and did not clear the latency/numerical gate. Neither candidate is in the
product path; the reusable harness remains for materially different compiler,
graph-boundary, or hardware candidates. MLX/Metal continues to preserve the
qualified computation.
