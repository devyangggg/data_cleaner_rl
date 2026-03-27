# Pipeline Debug Environment

An **OpenEnv-compliant** reinforcement learning environment where an AI agent receives a broken pandas dataframe and must issue fix commands one at a time until it matches the expected clean output.

## Tasks

| ID     | Difficulty | Description                                                |
| ------ | ---------- | ---------------------------------------------------------- |
| easy   | Easy       | Fix type casts, null values, and date formats in sales data |
| medium | Medium     | Fix a bad join between orders and customers tables          |
| hard   | Hard       | Trace and fix silent value corruption (double exchange rate) |

## Available Commands

| Command           | Parameters                              | Description                          |
| ----------------- | --------------------------------------- | ------------------------------------ |
| `cast_column`     | `column`, `dtype`                       | Cast a column to a new dtype         |
| `fill_nulls`      | `column`, `value`                       | Fill null values in a column         |
| `fix_date_format` | `column`, `format`                      | Re-parse dates with a given format   |
| `fix_join`        | `left_key`, `right_key`                 | Re-execute a join with correct keys  |
| `drop_duplicates` | `subset`                                | Remove duplicate rows                |
| `apply_transform` | `column`, `expression`                  | Apply a Python expression to a column|
| `revert_step`     | *(none)*                                | Undo the last action                 |

## Reward Function

- **Schema score** (0.0–0.4): fraction of correctly-typed columns
- **Value score** (0.0–0.4): fraction of rows matching expected values (tolerance 1e-2)
- **Efficiency score** (0.0–0.2): bonus for finishing in fewer steps
- **Penalties**: -0.05 for invalid commands, -0.02 for no-change actions

## Quick Start

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Start the server
uvicorn src.server:app --host 0.0.0.0 --port 8000

# Health check
curl http://localhost:8000/health
```

### Docker

```bash
docker build -t pipeline-debug-env .
docker run -p 8000:8000 pipeline-debug-env
```

### API Usage

```bash
# Reset with a task
curl -X POST "http://localhost:8000/reset?task_id=easy"

# Send an action
curl -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{"command": "cast_column", "parameters": {"column": "revenue", "dtype": "float64"}}'

# Check state
curl http://localhost:8000/state
```

### Running Inference

```bash
export API_BASE_URL="https://api-inference.huggingface.co/v1"
export MODEL_NAME="meta-llama/Meta-Llama-3-8B-Instruct"
export HF_TOKEN="hf_..."
export ENV_BASE_URL="http://localhost:8000"

python inference.py
```

## Project Structure

```
pipeline-debug-env/
├── Dockerfile
├── openenv.yaml
├── requirements.txt
├── inference.py
├── README.md
└── src/
    ├── server.py          # FastAPI endpoints
    ├── environment.py     # Core RL environment
    ├── models.py          # Pydantic data models
    ├── tasks/
    │   ├── easy.py        # Sales data bugs
    │   ├── medium.py      # Bad join task
    │   └── hard.py        # Silent corruption task
    ├── graders/
    │   └── grader.py      # Deterministic reward computation
    └── data/
        └── generator.py   # Seeded data generators
```

## Constraints

- Runs on 2 vCPU, 8 GB RAM
- Inference completes in under 20 minutes
- All data generation uses seed 42 for reproducibility
- Grading is fully deterministic
# data_cleaner_rl
