# vference

`vference` is an experimental, general local inference runtime for running
sparse Mixture-of-Experts models larger than unified memory on Apple Silicon.
Qwen3.5-35B-A3B on an 8 GB M2 MacBook Air is the first architecture adapter and
validation target, not the intended permanent scope.

The project has completed the lossless Stage 1 artifact build and internal-SSD
storage baseline. The Stage 2 runtime loads the 1.38 GB resident core and
streams exact experts into a bounded cache. Both the Python reference cache and
the native stable-slot cache generate with Qwen; sustained qualification and
further scheduling work remain in progress. Start with
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

## Native development build

Install the pinned environment and build the stable-slot extension:

```sh
uv sync --dev
make -C native build
```

The reference and native paths can then be selected explicitly with
`vference stream-generate ... --store python` and `--store stable`. The current
best qualified short-context decode configuration is:

```sh
uv run vference stream-generate artifacts/qwen3.5-35b-a3b-4bit-runtime \
  --prompt 'Hello' --max-tokens 64 --store stable \
  --cache-capacity 640 --cache-policy layer --prefill-chunk-size 32 --nocache
```

It remains experimental until the 8K context memory and reliability gate
passes.

For the current 8 GB long-context feasibility configuration, use 320 slots and
bound allocator caching between prefill chunks:

```sh
uv run vference stream-generate artifacts/qwen3.5-35b-a3b-4bit-runtime \
  --prompt 'Your long prompt' --max-tokens 256 --store stable \
  --cache-capacity 320 --cache-policy global --decode-cache-policy layer \
  --prefill-chunk-size 512 --clear-cache-between-prefill-chunks --nocache
```

This configuration averaged 2.278 decode tok/s across three identical
8K-total-token runs without swap growth, and a separate 7,936-token needle
retrieval returned the expected code. Unsafe contexts are rejected against a
measured memory model before prefill. Stable slots also match the Python
exact-expert reference with zero logit error on the first split-state and
five-domain deterministic corpus. Against an exact forced-demand 8K baseline,
it improves throughput 17.6% and reduces exposed expert time 15.7%. It is not
yet a release claim; broader long-context and application-surface cases remain.

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
