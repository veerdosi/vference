from pathlib import Path

import mlx.core as mx

from vference.runtime.admission import AdmissionEstimate
from vference.runtime.generate import generate_greedy
from vference.runtime.model import LoadedStreamingModel


class _Tokenizer:
    eos_token_ids = [999]

    def encode(self, prompt, add_special_tokens=False):
        return [7, 8]

    def decode(self, tokens):
        return "ok"


class _Store:
    capacity = 1
    cache_policy = "global"
    record_size = 1
    closed = False

    def validate_source_unchanged(self):
        return None

    def stats(self):
        return {
            "hits": 0,
            "misses": 0,
            "bytes_read": 0,
            "timing_seconds": {
                "pread": 0.0,
                "materialize": 0.0,
                "router_and_graph_wait": 0.0,
                "execute_total": 0.0,
            },
            "per_layer": [],
        }

    def close(self):
        self.closed = True

    def set_cache_policy(self, policy):
        self.cache_policy = policy


class _Model:
    calls = 0

    def __call__(self, inputs, cache):
        self.calls += 1
        return mx.array([[[0.0, 1.0]]])


class _Adapter:
    name = "test_adapter"
    caches = 0

    def route_shape(self, config):
        return 1, 1

    def validate_prefill_chunk(self, prompt_tokens, chunk_size, allow_unqualified):
        return None

    def estimate_admission(self, config, **kwargs):
        return AdmissionEstimate(
            admitted=True,
            budget_bytes=kwargs["budget_bytes"],
            estimated_peak_bytes=1,
            resident_bytes=0,
            model_state_bytes=0,
            runtime_reserve_bytes=0,
            prefill_transient_reserve_bytes=0,
            total_tokens=kwargs["total_tokens"],
            prefill_chunk_size=kwargs["prefill_chunk_size"],
        )

    def make_cache(self, model):
        self.caches += 1
        return []


def test_loaded_runtime_is_reused_but_request_state_is_fresh(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text("{}")
    store = _Store()
    model = _Model()
    adapter = _Adapter()
    runtime = LoadedStreamingModel(tmp_path.resolve(), model, _Tokenizer(), store, adapter)

    results = [
        generate_greedy(
            tmp_path,
            "hello",
            loaded_runtime=runtime,
            max_tokens=1,
            cache_capacity=1,
            chat_template=False,
            enable_thinking=False,
            prefill_chunk_size=512,
            max_mlx_memory_bytes=1024**3,
            store_kind="stable",
        )
        for _ in range(2)
    ]

    assert [result["output_text"] for result in results] == ["ok", "ok"]
    assert all(result["runtime_reused"] for result in results)
    assert adapter.caches == 2
    assert model.calls == 2
    assert not store.closed
