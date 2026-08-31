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
`vference stream-generate ... --store python` and `--store stable`. The stable
path remains opt-in until its sustained acceptance run is recorded.
