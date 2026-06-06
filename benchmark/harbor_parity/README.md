# CanItEdit Harbor Parity Runner

This directory adds an original-side Harbor parity runner for CanItEdit. It does **not** change the official CanItEdit dataset or evaluator. It only adds a Codex CLI execution path so the original benchmark can be compared fairly with the Harbor adapter.

## Parity setting

Use the same setting on both the original CanItEdit side and the Harbor side:

- Dataset: `nuprl/CanItEdit`
- Split: `test`
- Dataset revision: `3c07f38b1f9385f3214fcea94d4664c79df0d36a`
- Tasks: 210 total = 105 examples × `instruction_descriptive` and `instruction_lazy`
- Agent: `codex@0.118.0`
- Model: `gpt-5-mini`
- Reasoning effort: `low`
- Reasoning summary: `none`
- Completion count: 1
- Metric: pass@1 / resolved rate / accuracy
- Agent timeout: 600 seconds per task

Codex is run with:

```bash
codex exec \
  --dangerously-bypass-approvals-and-sandbox \
  --skip-git-repo-check \
  --model gpt-5-mini \
  --json \
  --enable unified_exec \
  -c model_reasoning_effort=low \
  -c model_reasoning_summary=none \
  -- "$(cat /workspace/prompt.md)"
```

The prompt text must match the Harbor adapter prompt.

## Isolation model

For each task, `run_codex_completions.py` creates a clean Docker workspace containing only:

- `/workspace/solution.py` — initialized to the CanItEdit `before` code
- `/workspace/prompt.md` — edit instruction plus the same before code

The agent container does **not** receive:

- official tests
- `after` reference code
- CanItEdit repository files
- benchmark outputs from other tasks

The script reads the final `/workspace/solution.py` and writes it as one official CanItEdit completion. Official tests and `after` code are present only in the generated completion JSON used by the official evaluator.

## Build the agent image

```bash
docker build \
  -f benchmark/harbor_parity/Dockerfile.agent \
  --build-arg CODEX_VERSION=0.118.0 \
  -t canitedit-codex-agent:0.118.0 \
  .
```

## Run original-side completions

Set parity API environment variables first. Do not commit them.

```bash
export OPENAI_API_KEY="..."
export OPENAI_BASE_URL="..."  # only if using a non-default OpenAI-compatible endpoint
```

Run one full parity round:

```bash
uv run --script benchmark/harbor_parity/run_codex_completions.py \
  --output-dir benchmark/harbor_parity/outputs/original_run1 \
  --agent-image canitedit-codex-agent:0.118.0 \
  --model gpt-5-mini \
  --codex-version 0.118.0 \
  --reasoning-effort low \
  --reasoning-summary none \
  --dataset-revision 3c07f38b1f9385f3214fcea94d4664c79df0d36a \
  --instruction-kinds both \
  --timeout-sec 600
```

Repeat with `original_run2` and `original_run3`.

For a no-API selection check:

```bash
uv run --script benchmark/harbor_parity/run_codex_completions.py \
  --output-dir /tmp/canitedit-dry-run \
  --limit 2 \
  --dry-run
```

## Evaluate with the official CanItEdit evaluator

The official evaluator remains unchanged:

```bash
podman run --rm --network none \
  --volume benchmark/harbor_parity/outputs/original_run1:/data:rw \
  ghcr.io/nuprl/canitedit \
  --dir /data --output-dir /data
```

`docker run` can be used instead of `podman run` if Docker is the available runtime.

## Collect scores

```bash
python benchmark/harbor_parity/collect_scores.py \
  --results-dir benchmark/harbor_parity/outputs/original_run1 \
  --output benchmark/harbor_parity/outputs/original_run1_summary.json
```

The summary reports `accuracy_percent = resolved / total * 100`. Use the three original-side run scores and the three Harbor-side run scores to report mean ± sample SEM and check run-range overlap.

## Artifact policy

Do not commit generated outputs, logs, traces, workspaces, or API keys. Keep run artifacts locally and upload them separately to the Harbor parity experiments dataset when the Harbor adapter PR is ready.
