# vference

`vference` is an experimental, general local inference runtime for running
sparse Mixture-of-Experts models larger than unified memory on Apple Silicon.
Qwen3.5-35B-A3B on an 8 GB M2 MacBook Air is the first architecture adapter and
validation target, not the intended permanent scope.

The project is currently in the architecture and feasibility phase. Start with
[the technical reference](docs/technical-reference.md). It records the model
facts, memory and I/O arithmetic, proposed runtime architecture, quality
invariants, validation gates, risks, and staged implementation plan.

The non-negotiable rule is that storage and scheduling optimizations must not
change model behavior. A missing expert causes a stall, never expert
substitution, skipping, remapping, or a reduced top-K.

The provisional usability target is at least 2.0 sustained decode tokens/second
at an admitted 8K context, while also materially outperforming both OS swapping
and naïve synchronous expert streaming on the same hardware and model artifact.
Stage 0 must confirm or revise the absolute threshold from measured M2 storage
limits before cache and prefetch tuning begins.
