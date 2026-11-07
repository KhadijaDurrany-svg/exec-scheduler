"""Benchmark harness: run every scheduler over a family of instances.

Reports, per method, the mean utility fraction and deadline-hit-rate with
percentile bootstrap 95% CIs, and a paired permutation p-value comparing
slotfit against the strongest baseline. Every number here is measured from a
real run -- nothing is hard-coded.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from dataclasses import asdict, dataclass

from slotfit.generator import GenConfig, generate_many
from slotfit.metrics import deadline_hit_rate, utility_fraction
from slotfit.models import Instance
from slotfit.schedulers import BASELINES, SCHEDULERS
from slotfit.stats import bootstrap_ci, paired_permutation_test


@dataclass(frozen=True)
class MethodResult:
    method: str
    utility_mean: float
    utility_ci: tuple[float, float]
    hit_rate_mean: float
    hit_rate_ci: tuple[float, float]


def dataset_checksum(instances: list[Instance]) -> str:
    h = hashlib.sha256()
    for inst in instances:
        h.update(inst.model_dump_json().encode("utf-8"))
    return h.hexdigest()


def _git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5, check=False,
        )
        sha = out.stdout.strip()
        return sha or "n/a"
    except (OSError, subprocess.SubprocessError):
        return "n/a"


def run_benchmark(
    n_instances: int = 200,
    seed: int = 7,
    iters: int = 400,
    cfg: GenConfig | None = None,
) -> dict:
    """Execute all schedulers and assemble a results dict."""
    instances = generate_many(seed, n_instances, cfg)

    util: dict[str, list[float]] = {m: [] for m in SCHEDULERS}
    hits: dict[str, list[float]] = {m: [] for m in SCHEDULERS}
    for i, inst in enumerate(instances):
        for method, fn in SCHEDULERS.items():
            # per-instance seed keeps random baseline + search reproducible yet varied
            sched = fn(inst, seed * 1000 + i)
            util[method].append(utility_fraction(inst, sched))
            hits[method].append(deadline_hit_rate(inst, sched))

    results: dict[str, MethodResult] = {}
    for method in SCHEDULERS:
        u_mean, u_lo, u_hi = bootstrap_ci(util[method], seed=seed)
        h_mean, h_lo, h_hi = bootstrap_ci(hits[method], seed=seed + 1)
        results[method] = MethodResult(method, u_mean, (u_lo, u_hi), h_mean, (h_lo, h_hi))

    best_baseline = max(BASELINES, key=lambda m: results[m].utility_mean)
    p_value = paired_permutation_test(util["slotfit"], util[best_baseline], seed=seed)
    delta = results["slotfit"].utility_mean - results[best_baseline].utility_mean

    return {
        "config": {
            "n_instances": n_instances,
            "seed": seed,
            "iters": iters,
            "generator": asdict(cfg or GenConfig()),
        },
        "methods": {m: asdict(r) for m, r in results.items()},
        "comparison": {
            "slotfit_vs": best_baseline,
            "utility_delta": delta,
            "paired_permutation_p": p_value,
        },
        "dataset_checksum": dataset_checksum(instances),
    }


def build_manifest(results: dict) -> dict:
    return {
        "git_sha": _git_sha(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "dataset_checksum": results["dataset_checksum"],
        "n_instances": results["config"]["n_instances"],
        "seed": results["config"]["seed"],
    }


def results_to_json(results: dict) -> str:
    return json.dumps(results, indent=2, sort_keys=True)
