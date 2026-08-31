# Experiments

Compact, reproducible experiment records live in `results.jsonl`. Large raw
traces and temporary benchmark payloads belong in `experiments/raw/` and are
ignored by Git.

Each JSONL record must include the fields required by
`docs/technical-reference.md`: source commit, timestamp, model revision,
hardware and storage path, runtime parameters, workload shape, cache state,
latency/throughput, memory/swap, relevant I/O/cache metrics, and conclusion.

