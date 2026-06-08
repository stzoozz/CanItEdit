#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "datasets==4.0.*",
#     "litellm",
#     "tqdm",
# ]
# ///
"""Run Harbor-style Codex parity completions for CanItEdit.

This script is intentionally limited to the original-side parity use case:
it runs Codex in a clean Docker workspace where only the before-code and edit
instruction are visible, then writes CanItEdit official-evaluator-compatible
completion JSON files.
"""

from __future__ import annotations

import argparse
import gzip
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any, Iterable, Literal

from datasets import load_dataset

DEFAULT_DATASET = "nuprl/CanItEdit"
DEFAULT_DATASET_REVISION = "3c07f38b1f9385f3214fcea94d4664c79df0d36a"
DEFAULT_SPLIT = "test"
DEFAULT_AGENT = "codex"
DEFAULT_CODEX_VERSION = "0.118.0"
DEFAULT_MODEL = "gpt-5-mini"
DEFAULT_REASONING_EFFORT = "low"
DEFAULT_REASONING_SUMMARY = "none"
DEFAULT_AGENT_IMAGE = f"canitedit-codex-agent:{DEFAULT_CODEX_VERSION}"
DEFAULT_TIMEOUT_SEC = 600

InstructionKind = Literal["instruction_descriptive", "instruction_lazy"]


CODEX_WRITEBACK_TEMPLATE = """The following is the original direct-edit prompt. In the original benchmark, the model returns the edited code after `## Code After:`.

For this Codex parity run, apply the same edit by writing the final edited Python code to `/workspace/solution.py` instead of returning it in chat. Do not create a different answer file.

{official_prompt}"""


@dataclass(frozen=True)
class TaskItem:
    example: dict[str, Any]
    instr_kind: InstructionKind

    @property
    def output_name(self) -> str:
        return f"{self.example['full_name']}_{self.instr_kind}.json.gz"

    @property
    def safe_name(self) -> str:
        raw = f"{self.example['full_name']}_{self.instr_kind}"
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", raw)

    @property
    def instruction(self) -> str:
        value = self.example.get(self.instr_kind)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"Missing instruction text for {self.example.get('full_name')} {self.instr_kind}"
            )
        return value

    @property
    def before(self) -> str:
        value = self.example.get("before")
        if not isinstance(value, str):
            raise ValueError(f"Missing before code for {self.example.get('full_name')}")
        return value


def write_json_gz(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)


def parse_instruction_kinds(value: str) -> list[InstructionKind]:
    normalized = value.strip().lower()
    if normalized == "both":
        return ["instruction_descriptive", "instruction_lazy"]
    if normalized in {"descriptive", "instruction_descriptive"}:
        return ["instruction_descriptive"]
    if normalized in {"lazy", "instruction_lazy"}:
        return ["instruction_lazy"]
    raise ValueError(
        "--instruction-kinds must be one of: both, descriptive, lazy, "
        "instruction_descriptive, instruction_lazy"
    )


def load_task_items(args: argparse.Namespace) -> list[TaskItem]:
    dataset = load_dataset(
        args.dataset,
        args.subset,
        split=args.split,
        revision=args.dataset_revision,
    )
    instruction_kinds = parse_instruction_kinds(args.instruction_kinds)

    allow_ids: set[str] | None = None
    if args.task_ids:
        allow_ids = {part.strip() for part in args.task_ids.split(",") if part.strip()}

    items: list[TaskItem] = []
    for row in dataset:
        ex = dict(row)
        if allow_ids is not None:
            identifiers = {str(ex.get("id")), str(ex.get("name")), str(ex.get("full_name"))}
            if identifiers.isdisjoint(allow_ids):
                continue
        for instr_kind in instruction_kinds:
            items.append(TaskItem(example=ex, instr_kind=instr_kind))

    if args.limit is not None:
        items = items[: args.limit]
    return items


@cache
def load_official_direct_model_class() -> type:
    official_path = Path(__file__).resolve().parents[1] / "generate_completions.py"
    spec = importlib.util.spec_from_file_location("canitedit_generate_completions", official_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load official CanItEdit generator from {official_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.DirectEditModel


def build_official_prompt(item: TaskItem) -> str:
    # Intentionally delegates to the official CanItEdit prompt builder instead of
    # reimplementing its template. This preserves upstream prompt/config semantics.
    direct_model_cls = load_official_direct_model_class()
    return direct_model_cls(model_name="codex-parity-placeholder", one_shot=False).format_prompt(
        item.before,
        item.instruction,
    )


def render_writeback_template(template: str, replacements: dict[str, str]) -> str:
    def replace_match(match: re.Match[str]) -> str:
        key = match.group(1)
        return replacements.get(key, match.group(0))

    return re.sub(r"\{([A-Za-z_][A-Za-z0-9_]*)\}", replace_match, template)


def build_codex_instruction(item: TaskItem) -> str:
    return render_writeback_template(
        CODEX_WRITEBACK_TEMPLATE,
        {"official_prompt": build_official_prompt(item)},
    )


def run_command(
    command: list[str], *, timeout: int | None = None, cwd: Path | None = None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=str(cwd) if cwd else None,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
    )


def get_image_codex_version(args: argparse.Namespace) -> str:
    command = [
        args.docker_binary,
        "run",
        "--rm",
        args.agent_image,
        "bash",
        "-lc",
        "codex --version",
    ]
    result = run_command(command, timeout=60)
    if result.returncode != 0:
        raise RuntimeError(
            "Unable to inspect Codex version in agent image. "
            f"Command stderr:\n{result.stderr.strip()}"
        )
    output = result.stdout.strip().splitlines()
    if not output:
        raise RuntimeError("codex --version returned no output in agent image")
    version_line = output[-1].strip()
    return version_line.removeprefix("codex-cli").strip()


def ensure_codex_version(args: argparse.Namespace) -> str:
    observed = get_image_codex_version(args)
    if observed != args.codex_version and not args.allow_version_mismatch:
        raise RuntimeError(
            f"Agent image has codex version {observed!r}; expected {args.codex_version!r}. "
            "Rebuild Dockerfile.agent or pass --allow-version-mismatch only for debugging."
        )
    return observed


def docker_env_flags() -> list[str]:
    flags = ["-e", "OPENAI_API_KEY"]
    if os.environ.get("OPENAI_BASE_URL"):
        flags.extend(["-e", "OPENAI_BASE_URL"])
    return flags


def codex_shell_script(args: argparse.Namespace) -> str:
    # Keep secrets out of docker command arguments. The key is copied from the
    # container environment into an ephemeral CODEX_HOME auth.json, then removed.
    return f"""
set -euo pipefail
if [ -z "${{OPENAI_API_KEY:-}}" ]; then
  echo "OPENAI_API_KEY is required for Codex parity runs" >&2
  exit 2
fi
export CODEX_HOME=/tmp/codex-home
mkdir -p "$CODEX_HOME" /logs
python3 - <<'PY_SETUP'
import json
import os
from pathlib import Path
home = Path(os.environ["CODEX_HOME"])
home.mkdir(parents=True, exist_ok=True)
(home / "auth.json").write_text(json.dumps({{"OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY", "")}}), encoding="utf-8")
base_url = os.environ.get("OPENAI_BASE_URL")
if base_url:
    (home / "config.toml").write_text(f"openai_base_url = {{json.dumps(base_url)}}\\n", encoding="utf-8")
PY_SETUP
prompt="$1"
set +e
codex exec \
  --dangerously-bypass-approvals-and-sandbox \
  --skip-git-repo-check \
  --model {args.model} \
  --json \
  --enable unified_exec \
  -c model_reasoning_effort={args.reasoning_effort} \
  -c model_reasoning_summary={args.reasoning_summary} \
  -- "$prompt" 2>&1 </dev/null | tee /logs/codex.jsonl
status=${{PIPESTATUS[0]}}
set -e
rm -f "$CODEX_HOME/auth.json" "$CODEX_HOME/config.toml"
if [ -d "$CODEX_HOME/sessions" ]; then
  rm -rf /logs/sessions
  cp -R "$CODEX_HOME/sessions" /logs/sessions
fi
rm -rf "$CODEX_HOME"
exit "$status"
""".strip()


def run_codex_for_item(
    item: TaskItem,
    args: argparse.Namespace,
    output_dir: Path,
    work_root: Path,
    codex_version_observed: str,
) -> dict[str, Any]:
    completion_path = output_dir / item.output_name
    if completion_path.exists() and not args.overwrite:
        return {"skipped": True, "path": str(completion_path)}

    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required unless --dry-run is used")

    task_work_dir = work_root / item.safe_name
    task_logs_dir = output_dir / "logs" / item.safe_name
    if task_work_dir.exists():
        shutil.rmtree(task_work_dir)
    if task_logs_dir.exists() and args.overwrite:
        shutil.rmtree(task_logs_dir)
    task_work_dir.mkdir(parents=True, exist_ok=True)
    task_logs_dir.mkdir(parents=True, exist_ok=True)

    (task_work_dir / "solution.py").write_text(item.before, encoding="utf-8")
    codex_instruction = build_codex_instruction(item)

    container_name = f"canitedit-parity-{item.safe_name[:48]}-{uuid.uuid4().hex[:8]}"
    command = [
        args.docker_binary,
        "run",
        "--rm",
        "--name",
        container_name,
        *docker_env_flags(),
        "-v",
        f"{task_work_dir.resolve()}:/workspace:rw",
        "-v",
        f"{task_logs_dir.resolve()}:/logs:rw",
        "-w",
        "/workspace",
        args.agent_image,
        "bash",
        "-lc",
        codex_shell_script(args),
        "canitedit-codex",
        codex_instruction,
    ]

    started = time.time()
    timed_out = False
    try:
        result = run_command(command, timeout=args.timeout_sec)
    except subprocess.TimeoutExpired:
        timed_out = True
        subprocess.run(
            [args.docker_binary, "rm", "-f", container_name],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        result = subprocess.CompletedProcess(command, returncode=124, stdout="", stderr="timeout")
    duration_sec = time.time() - started

    solution_path = task_work_dir / "solution.py"
    completion = solution_path.read_text(encoding="utf-8") if solution_path.exists() else ""

    (task_logs_dir / "docker_stdout.txt").write_text(result.stdout or "", encoding="utf-8")
    (task_logs_dir / "docker_stderr.txt").write_text(result.stderr or "", encoding="utf-8")

    if result.returncode != 0 and not args.allow_agent_failures:
        raise RuntimeError(
            f"Codex exited with code {result.returncode} for {item.safe_name}. "
            f"Logs are in {task_logs_dir}. Pass --allow-agent-failures only for debugging."
        )

    payload = dict(item.example)
    payload.update(
        {
            "instr_kind": item.instr_kind,
            "prompt": "",
            "completions": [completion],
            "language": "py",
            "script_args": sanitized_script_args(args),
            "harbor_parity_metadata": {
                "agent": DEFAULT_AGENT,
                "agent_image": args.agent_image,
                "codex_version_expected": args.codex_version,
                "codex_version_observed": codex_version_observed,
                "model": args.model,
                "reasoning_effort": args.reasoning_effort,
                "reasoning_summary": args.reasoning_summary,
                "timeout_sec": args.timeout_sec,
                "duration_sec": round(duration_sec, 3),
                "exit_code": result.returncode,
                "timed_out": timed_out,
                "trace_path": str((Path("logs") / item.safe_name).as_posix()),
                "openai_base_url_configured": bool(os.environ.get("OPENAI_BASE_URL")),
            },
        }
    )
    write_json_gz(completion_path, payload)

    if not args.keep_workspaces:
        shutil.rmtree(task_work_dir, ignore_errors=True)

    return {
        "skipped": False,
        "path": str(completion_path),
        "exit_code": result.returncode,
        "duration_sec": duration_sec,
    }


def sanitized_script_args(args: argparse.Namespace) -> dict[str, Any]:
    hidden = {"dry_run"}
    result: dict[str, Any] = {}
    for key, value in vars(args).items():
        if key in hidden:
            continue
        if isinstance(value, Path):
            result[key] = str(value)
        else:
            result[key] = value
    return result


def summarize_selection(items: Iterable[TaskItem]) -> list[dict[str, Any]]:
    return [
        {
            "id": item.example.get("id"),
            "full_name": item.example.get("full_name"),
            "instr_kind": item.instr_kind,
            "output_name": item.output_name,
        }
        for item in items
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--dataset-revision", default=DEFAULT_DATASET_REVISION)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--subset", default=None)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--work-dir", type=Path, default=None)
    parser.add_argument("--agent-image", default=DEFAULT_AGENT_IMAGE)
    parser.add_argument("--docker-binary", default="docker")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--codex-version", default=DEFAULT_CODEX_VERSION)
    parser.add_argument("--reasoning-effort", default=DEFAULT_REASONING_EFFORT)
    parser.add_argument("--reasoning-summary", default=DEFAULT_REASONING_SUMMARY)
    parser.add_argument("--instruction-kinds", default="both")
    parser.add_argument("--task-ids", default=None, help="Comma-separated id/name/full_name filter")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--timeout-sec", type=int, default=DEFAULT_TIMEOUT_SEC)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--keep-workspaces", action="store_true")
    parser.add_argument(
        "--allow-agent-failures",
        action="store_true",
        help="Write completion files even when Codex exits non-zero; use only for debugging.",
    )
    parser.add_argument("--allow-version-mismatch", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="List selected tasks without running Codex")
    args = parser.parse_args(argv)

    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be >= 1")

    items = load_task_items(args)
    if not items:
        raise RuntimeError("No CanItEdit tasks selected")

    if args.dry_run:
        print(json.dumps({"num_items": len(items), "items": summarize_selection(items)}, indent=2))
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    work_root = args.work_dir or Path(tempfile.mkdtemp(prefix="canitedit-parity-work-"))
    work_root.mkdir(parents=True, exist_ok=True)

    codex_version_observed = ensure_codex_version(args)
    print(
        f"Running {len(items)} CanItEdit parity items with codex@{codex_version_observed} + {args.model}",
        flush=True,
    )

    summary: list[dict[str, Any]] = []
    try:
        for index, item in enumerate(items, start=1):
            print(f"[{index}/{len(items)}] {item.example['full_name']} {item.instr_kind}", flush=True)
            result = run_codex_for_item(item, args, args.output_dir, work_root, codex_version_observed)
            summary.append(
                {
                    "id": item.example.get("id"),
                    "full_name": item.example.get("full_name"),
                    "instr_kind": item.instr_kind,
                    **result,
                }
            )
    finally:
        if args.work_dir is None and not args.keep_workspaces:
            shutil.rmtree(work_root, ignore_errors=True)

    (args.output_dir / "run_codex_completions_summary.json").write_text(
        json.dumps(
            {
                "dataset": args.dataset,
                "dataset_revision": args.dataset_revision,
                "split": args.split,
                "num_items": len(items),
                "agent": DEFAULT_AGENT,
                "codex_version": codex_version_observed,
                "model": args.model,
                "reasoning_effort": args.reasoning_effort,
                "reasoning_summary": args.reasoning_summary,
                "items": summary,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
