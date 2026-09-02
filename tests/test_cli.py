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
