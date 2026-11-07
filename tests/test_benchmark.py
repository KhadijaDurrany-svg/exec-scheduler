from __future__ import annotations

import json

from slotfit.benchmark import build_manifest, dataset_checksum, results_to_json, run_benchmark
from slotfit.generator import generate_many


def test_run_benchmark_structure():
    res = run_benchmark(n_instances=12, seed=7, iters=60)
    assert set(res["methods"]) == {"slotfit", "fifo", "edf", "priority", "random"}
    for m in res["methods"].values():
        assert 0.0 <= m["utility_mean"] <= 1.0
        assert 0.0 <= m["hit_rate_mean"] <= 1.0
    assert "paired_permutation_p" in res["comparison"]
    assert 0.0 < res["comparison"]["paired_permutation_p"] <= 1.0


def test_slotfit_at_least_matches_best_baseline():
    res = run_benchmark(n_instances=20, seed=7, iters=120)
    methods = res["methods"]
    best_baseline = max(("fifo", "edf", "priority", "random"), key=lambda m: methods[m]["utility_mean"])
    assert methods["slotfit"]["utility_mean"] >= methods[best_baseline]["utility_mean"]


def test_dataset_checksum_deterministic():
    a = generate_many(7, 8)
    b = generate_many(7, 8)
    assert dataset_checksum(a) == dataset_checksum(b)
    assert len(dataset_checksum(a)) == 64


def test_manifest_fields():
    res = run_benchmark(n_instances=6, seed=1, iters=30)
    man = build_manifest(res)
    assert set(man) >= {"git_sha", "python", "dataset_checksum", "n_instances", "seed"}
    assert man["dataset_checksum"] == res["dataset_checksum"]


def test_results_json_roundtrips():
    res = run_benchmark(n_instances=6, seed=2, iters=30)
    parsed = json.loads(results_to_json(res))
    assert parsed["config"]["n_instances"] == 6
