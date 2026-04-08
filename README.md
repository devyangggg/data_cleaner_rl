# Data Cleaner RL

Data Cleaner RL is an OpenEnv-compatible RL environment where an agent repairs corrupted pandas pipelines step by step. It is designed for reproducible evaluation: deterministic task generation, fixed reward shaping, explicit command APIs, and portable deployment on Docker/Hugging Face Spaces.

## Why This Project

- Turns data cleaning into a sequential decision process instead of static one-shot scripts.
- Provides a controllable benchmark for agent behavior under realistic ETL bugs.
- Exposes OpenEnv contract endpoints (`/reset`, `/step`, `/state`) for plug-and-play agent evaluation.

## Judging Criteria Mapping

- **Correct OpenEnv API**: Implemented in `src/server.py` with `/reset`, `/step`, `/state`, plus `/metadata` and `/schema` for discoverability.
- **Working graders/reward logic**: Deterministic scoring in `src/graders/grader.py` (schema + values + efficiency + penalties).
- **LLM-driven inference**: `inference.py` calls an OpenAI-compatible endpoint using `API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN`.
- **Programmatic eval scores**: `benchmark.py` runs multi-task evaluations and saves `benchmark_results.json`.
- **Innovation**: Curriculum endpoint, procedural task generator, safe command dispatch, explainable deterministic step rationales.
- **Reproducibility**: Seeded data generation and deterministic task definitions in `src/tasks/easy.py`, `src/tasks/medium.py`, `src/tasks/hard.py`.

## API Endpoints

- `GET /health` -> `{"status": "healthy"}`
- `GET /metadata` -> environment metadata, task list, command signatures
- `GET /schema` -> JSON schema for action, observation, and step result
- `GET /mcp` -> MCP tool manifest
- `POST /reset?task_id=easy|medium|hard` -> start episode and return observation
- `POST /curriculum/reset` -> auto-select difficulty from rolling success history
- `POST /step` -> apply one action and get `observation`, `reward`, `done`, `info`
- `GET /state` -> current episode snapshot
- `GET /web` -> lightweight HTML status page

## Quickstart

```bash
docker build -t data-cleaner-rl .
docker run -p 7860:7860 data-cleaner-rl
```

Open `http://localhost:7860/web`.

## LLM Inference Runner

```bash
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=HuggingFaceH4/zephyr-7b-beta
export HF_TOKEN=hf_your_token_here
export ENV_URL=http://localhost:7860

python inference.py
```

`inference.py` runs all three tasks (`easy`, `medium`, `hard`) and prints per-step traces with deterministic rationale from the environment info payload.

## Benchmarking

Run:

```bash
python benchmark.py --agent heuristic
python benchmark.py --agent llm
```

Outputs:
- Markdown table in stdout
- `benchmark_results.json` with per-run details

Benchmark template (replace with your measured values before final submission):

| Task   | Agent     | Success Rate | Avg Reward | Avg Steps |
|--------|-----------|--------------|------------|-----------|
| easy   | heuristic | 5/5 (100%)   | 0.94       | 3.0       |
| medium | heuristic | 5/5 (100%)   | 0.9867     | 1.0       |
| hard   | heuristic | 5/5 (100%)   | 0.99       | 1.0       |
| easy   | llm       | 0/5 (0%)     | 0.0        | 0.0       |
| medium | llm       | 0/5 (0%)     | 0.0        | 0.0       |
| hard   | llm       | 0/5 (0%)     | 0.0        | 0.0       |

## RL Training Demo (Learning Loop)

This repo includes a lightweight PyTorch policy-gradient training loop to demonstrate real agent learning inside the environment on the `easy` task.

Run training (100 episodes):

```bash
py train_rl_easy.py --episodes 100
```

Evaluate trained policy vs random baseline:

```bash
py eval_rl_agent.py --episodes 20 --checkpoint outputs/easy_policy.pt
```

Plot reward and success curves:

```bash
py plot_training_curve.py --metrics outputs/train_metrics_easy.json --out outputs/reward_curve_easy.png
```

Artifacts:
- `outputs/easy_policy.pt` (trained policy checkpoint)
- `outputs/train_metrics_easy.json` (reward/success curves)
- `outputs/eval_easy.json` (policy vs random comparison)
- `outputs/reward_curve_easy.png` (judge-ready reward curve)

Example run (100 episodes, seed 42):
- training reward improved from `0.5446` (first 10 episodes avg) to `0.6578` (last 10 episodes avg)
- evaluation snapshot (20 episodes): RL avg reward `0.59`, random avg reward `0.5025`

## Available Commands

| Command | Params | Description |
|---|---|---|
| `cast_column` | `column`, `dtype` | Cast a column to a target dtype |
| `fill_nulls` | `column`, `value` | Fill null values in a column |
| `fix_date_format` | `column`, `format` | Parse and normalize date strings |
| `fix_join` | `left_key`, `right_key` | Rebuild join using source task tables |
| `drop_duplicates` | `subset` | Remove duplicate rows |
| `apply_transform` | `column` plus safe transform params | Deterministic transform without `eval` |
| `revert_step` | none | Revert previous dataframe state |
| `rename_column` | `old_name`, `new_name` | Rename a column |
| `drop_column` | `column` | Drop a column |
| `sort_values` | `by`, optional `ascending` | Sort rows by one/multiple columns |

## Reward Function

- Schema score (`0.0` to `0.4`): correctly typed column fraction
- Value score (`0.0` to `0.4`): correctly matching rows (numeric tolerance `1e-2`)
- Efficiency score (`0.0` to `0.2`): remaining-step bonus
- Penalties: `-0.05` invalid action, `-0.02` no-change action

## OpenEnv Manifest

`openenv.yaml`:

```yaml
name: data-cleaner-rl
version: "1.0.0"
description: "RL environment for debugging broken pandas data pipelines"
tags: [data-engineering, pandas, etl, debugging]
tasks:
  - id: easy
    description: "Single dtype mismatch bug"
    max_steps: 10
  - id: medium
    description: "Null values and join errors"
    max_steps: 15
  - id: hard
    description: "Multiple compounding ETL bugs"
    max_steps: 20
action_space:
  type: discrete_parameterized
  commands: [cast_column, fill_nulls, fix_join, rename_column, drop_column, sort_values]
observation_space:
  type: structured
  fields: [current_schema, expected_schema, preview_rows, diff_summary]
```

## Tests

```bash
pytest tests/ -v
```

Current tests validate reset/step behavior, invalid command handling, reward determinism, episode termination, and no-change penalty behavior.

## Demo UI

```bash
python demo.py
```

This launches a Gradio UI for manual interaction with the environment.

For live demos, use **Auto-play heuristic** in the UI to show a full episode trajectory timeline instantly.

## Submission Checklist

- Ensure Docker image boots on port `7860`
- Ensure `/reset`, `/step`, `/state` are reachable
- Run and save benchmark outputs
- Run `inference.py` with a working HF token and include logs/screenshot in submission
- Include `openenv.yaml` and architecture overview in your final deck
