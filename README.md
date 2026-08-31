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

This configuration has passed one synthetic 8K-total-token run at 2.281 decode
tok/s without swap growth. It is not yet a release claim; repeated and
meaningful long-context regressions remain.
