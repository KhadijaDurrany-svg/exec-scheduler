"""Command-line interface for slotfit.

Subcommands
-----------
``bench``   run the full benchmark; writes ``benchmarks/results.json`` and
            ``benchmarks/manifest.json`` unless ``--no-write``.
``plan``    load a tasks file (an Instance as JSON) and print a schedule.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from slotfit.benchmark import build_manifest, results_to_json, run_benchmark
from slotfit.metrics import deadline_hit_rate, utility_fraction
from slotfit.models import Instance
from slotfit.schedulers import SCHEDULERS, plan


def _fmt_ci(ci: list[float] | tuple[float, float]) -> str:
    return f"[{ci[0]:.3f}, {ci[1]:.3f}]"


def _cmd_bench(args: argparse.Namespace) -> int:
    results = run_benchmark(n_instances=args.instances, seed=args.seed, iters=args.iters)
    methods = results["methods"]
    comp = results["comparison"]

    cfg = results["config"]
    print(f"instances={cfg['n_instances']}  seed={cfg['seed']}  iters={cfg['iters']}")
    print(f"{'method':<10} {'utility':>8} {'util 95% CI':>18} {'hit-rate':>9} {'hit 95% CI':>18}")
    for method in ("slotfit", "fifo", "edf", "priority", "random"):
        r = methods[method]
        print(
            f"{method:<10} {r['utility_mean']:>8.3f} {_fmt_ci(r['utility_ci']):>18} "
            f"{r['hit_rate_mean']:>9.3f} {_fmt_ci(r['hit_rate_ci']):>18}"
        )
    print(
        f"\nslotfit vs {comp['slotfit_vs']}: utility delta={comp['utility_delta']:+.3f}, "
        f"paired permutation p={comp['paired_permutation_p']:.4g}"
    )

    if args.no_write:
        return 0

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(results_to_json(results) + "\n")
    (out_dir / "manifest.json").write_text(json.dumps(build_manifest(results), indent=2, sort_keys=True) + "\n")
    print(f"\nwrote {out_dir / 'results.json'} and {out_dir / 'manifest.json'}")
    return 0


def _cmd_plan(args: argparse.Namespace) -> int:
    raw = json.loads(Path(args.file).read_text())
    instance = Instance.model_validate(raw)
    schedule = plan(instance, method=args.method, seed=args.seed)

    placed = sorted(schedule.placed, key=lambda p: p.start)
    print(f"method={args.method}  placed={len(placed)}  unplaced={len(schedule.unplaced)}")
    print(f"utility={utility_fraction(instance, schedule):.3f}  hit-rate={deadline_hit_rate(instance, schedule):.3f}")
    deadlines = {t.id: t.deadline for t in instance.tasks}
    print(f"{'task':<10} {'start':>7} {'finish':>7} {'deadline':>9} {'on-time':>8}")
    for p in placed:
        on_time = "yes" if p.finish <= deadlines[p.task_id] else "LATE"
        print(f"{p.task_id:<10} {p.start:>7} {p.finish:>7} {deadlines[p.task_id]:>9} {on_time:>8}")
    if schedule.unplaced:
        print("unplaced:", ", ".join(sorted(schedule.unplaced)))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="slotfit", description="offline constraint-based day planner")
    sub = parser.add_subparsers(dest="command", required=True)

    b = sub.add_parser("bench", help="run benchmark vs baselines")
    b.add_argument("--instances", type=int, default=200)
    b.add_argument("--seed", type=int, default=7)
    b.add_argument("--iters", type=int, default=400, help="local-search iterations for slotfit")
    b.add_argument("--out", default="benchmarks", help="output directory for results/manifest")
    b.add_argument("--no-write", action="store_true", help="print only; do not write files")
    b.set_defaults(func=_cmd_bench)

    p = sub.add_parser("plan", help="schedule a tasks file")
    p.add_argument("file", help="path to an Instance JSON file")
    p.add_argument("--method", default="slotfit", choices=sorted(SCHEDULERS))
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=_cmd_plan)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
