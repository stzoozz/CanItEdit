#!/usr/bin/env python3
"""Collect CanItEdit official evaluator results for Harbor parity runs."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return json.load(f)
    return json.loads(path.read_text(encoding="utf-8"))


def is_success(result: dict[str, Any]) -> bool:
    return result.get("status") == "OK" and result.get("exit_code") == 0


def collect(results_dir: Path) -> dict[str, Any]:
    paths = sorted(results_dir.glob("*.results.json")) + sorted(
        results_dir.glob("*.results.json.gz")
    )
    if not paths:
        raise FileNotFoundError(
            f"No official evaluator result files found in {results_dir}. "
            "Run ghcr.io/nuprl/canitedit first."
        )

    tasks: list[dict[str, Any]] = []
    for path in paths:
        payload = read_json(path)
        completions = payload.get("results") or []
        passed = any(is_success(result) for result in completions if isinstance(result, dict))
        first = completions[0] if completions and isinstance(completions[0], dict) else {}
        tasks.append(
            {
                "file": path.name,
                "id": payload.get("id"),
                "full_name": payload.get("full_name"),
                "instr_kind": payload.get("instr_kind"),
                "passed": passed,
                "num_completions": len(completions),
                "status": first.get("status"),
                "exit_code": first.get("exit_code"),
                "stderr": first.get("stderr"),
            }
        )

    num_tasks = len(tasks)
    num_resolved = sum(1 for task in tasks if task["passed"])
    accuracy = num_resolved / num_tasks if num_tasks else 0.0
    return {
        "results_dir": str(results_dir),
        "num_tasks": num_tasks,
        "num_resolved": num_resolved,
        "accuracy": accuracy,
        "accuracy_percent": accuracy * 100,
        "tasks": tasks,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    summary = collect(args.results_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        f"accuracy={summary['accuracy_percent']:.2f}% "
        f"({summary['num_resolved']}/{summary['num_tasks']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
