from __future__ import annotations

import json
from pathlib import Path

from slotfit.cli import main
from slotfit.generator import GenConfig, generate


def test_bench_no_write_runs(capsys):
    rc = main(["bench", "--instances", "8", "--seed", "7", "--iters", "40", "--no-write"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "slotfit" in out and "paired permutation p" in out


def test_bench_writes_files(tmp_path, capsys):
    out_dir = tmp_path / "bench"
    rc = main(["bench", "--instances", "6", "--seed", "1", "--iters", "30", "--out", str(out_dir)])
    assert rc == 0
    results = json.loads((out_dir / "results.json").read_text())
    manifest = json.loads((out_dir / "manifest.json").read_text())
    assert results["config"]["n_instances"] == 6
    assert manifest["dataset_checksum"] == results["dataset_checksum"]


def test_plan_command(tmp_path, capsys):
    inst = generate(42, GenConfig(n_tasks=10, n_days=2))
    f = tmp_path / "inst.json"
    f.write_text(inst.model_dump_json())
    rc = main(["plan", str(f), "--method", "slotfit"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "utility=" in out and "hit-rate=" in out


def test_plan_with_baseline_method(tmp_path):
    inst = generate(3, GenConfig(n_tasks=8, n_days=2))
    f = tmp_path / "inst.json"
    f.write_text(inst.model_dump_json())
    assert main(["plan", str(f), "--method", "edf"]) == 0


def test_sample_instance_file_is_valid():
    path = Path(__file__).resolve().parents[1] / "examples" / "sample_tasks.json"
    data = json.loads(path.read_text())
    assert "tasks" in data and "work_windows" in data
