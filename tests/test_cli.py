import builtins
from copy import deepcopy

import vference.cli as cli
from vference.cli import _resolved_demand_workers, build_parser


def test_stable_cli_defaults_to_qualified_parallel_demand() -> None:
    args = build_parser().parse_args(
        ["stream-generate", "artifact", "--prompt", "hello"]
    )

    assert args.store == "stable"
    assert args.demand_workers is None
    assert _resolved_demand_workers(args.store, args.demand_workers) == 8


def test_python_cli_keeps_serial_demand_default() -> None:
    args = build_parser().parse_args(
        ["stream-generate", "artifact", "--prompt", "hello", "--store", "python"]
    )

    assert _resolved_demand_workers(args.store, args.demand_workers) == 1


def test_explicit_demand_worker_count_wins() -> None:
    assert _resolved_demand_workers("stable", 3) == 3


def test_prompt_file_and_literal_prompt_are_exclusive() -> None:
    parser = build_parser()
    args = parser.parse_args(
        ["stream-generate", "artifact", "--prompt-file", "document.md"]
    )

    assert str(args.prompt_file) == "document.md"
    assert args.prompt is None


def test_chat_defaults_to_qualified_stateless_configuration() -> None:
    args = build_parser().parse_args(["chat", "artifact"])

    assert args.cache_capacity == 320
    assert args.max_tokens == 256
    assert args.nocache
    assert args.session_log is None


def test_doctor_defaults_to_cached_integrity_check() -> None:
    args = build_parser().parse_args(["doctor", "artifact"])

    assert str(args.artifact) == "artifact"
    assert not args.force_integrity


def test_chat_rerenders_complete_transcript_each_turn(monkeypatch) -> None:
    prompts = iter(["first", "second", "/quit"])
    calls = []
    outputs = iter(["answer one", "answer two"])

    class FakeRuntime:
        closed = False

        def close(self):
            self.closed = True

    runtime = FakeRuntime()

    monkeypatch.setattr(builtins, "input", lambda _: next(prompts))
    monkeypatch.setattr(cli, "load_streaming_model", lambda *args, **kwargs: runtime)

    def fake_generate(*args, **kwargs):
        assert kwargs["loaded_runtime"] is runtime
        calls.append(deepcopy(kwargs["messages"]))
        return {
            "output_text": next(outputs),
            "prompt_tokens": 10,
            "output_tokens": [1],
            "decode_tokens_per_second": 2.5,
            "mlx_peak_bytes": 2 * 1024**3,
            "system_swap_bytes": {"delta": 0},
        }

    monkeypatch.setattr(cli, "generate_greedy", fake_generate)
    args = build_parser().parse_args(["chat", "artifact"])
    args.func(args)

    assert calls == [
        [{"role": "user", "content": "first"}],
        [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "answer one"},
            {"role": "user", "content": "second"},
        ],
    ]
    assert runtime.closed
