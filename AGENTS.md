# Agent guidance

Before designing or implementing runtime work, read
`docs/technical-reference.md` in full.

The correctness/quality invariant in section 2 is authoritative. Storage,
caching, prefetch, scheduling, and heterogeneous-compute changes must not alter
the checkpoint's router decisions or model math. On an expert miss, stall for
the exact expert. Do not substitute, skip, remap, or reduce top-K in the default
runtime.

Keep runtime-correctness comparisons separate from quantization-quality
comparisons as defined in section 8. Update the reference when verified facts,
upstream revisions, architectural decisions, or measured feasibility results
change. Label derived estimates, external claims, and local measurements
clearly, and include reproducible benchmark metadata with every performance
claim.

Artifact acquisition is staged. Use the pinned MLX 4-bit Qwen artifact for
runtime development. Do not download the official BF16 checkpoint until Stage
6 quantization-quality evaluation.
