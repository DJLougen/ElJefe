# ElJefe — cross-slice contracts

Single source of truth for interfaces shared between implementation slices.
`src/eljefe/schema.py` is already written — do not change it without updating
this file. All paths below are repo-relative; scripts resolve them from the
repo root regardless of cwd (use `Path(__file__).resolve().parents[1]`).

## Data files (JSONL unless noted)

| File | Row model | Written by | Read by |
|---|---|---|---|
| `data/prompts/tasks.jsonl` | `Task` | 01/02 | 03, 04, 05, 07 |
| `data/generations/local_e4b.jsonl` | `Generation` | 03 | 05, 07 |
| `data/generations/frontier.jsonl` | `Generation` | 04 | 05, 07 |
| `data/scores/scored.jsonl` | `ScoredRow` | 05/06 | 07 |
| `data/router/router_dataset.parquet` | `RouterRow` | 07 | 08–13 |
| `data/router/mac_calibration.jsonl` | `Task` | 07 | 16 |

## Module APIs

### `eljefe.graders`
```python
def grade(task: Task, answer: str) -> GradeResult
class GradeResult(BaseModel):  # score in [0,1]
    score: float
    grader_type: str           # "deterministic" | "llm_judge"
    confidence: float          # 1.0 for deterministic
    detail: dict               # grader-specific evidence
GRADERS: dict[str, Callable[[Task, str], GradeResult]]  # keyed by Task.grader
```
Graders must never raise on malformed answers — return score 0.0 with detail.

### `eljefe.features`
```python
def extract_features(prompt: str, system_prompt: str | None = None,
                     metadata: dict | None = None) -> dict[str, float | int | str]
```
Cheap derived features (plan §4): token/char counts, code-block count, URL
presence, question-word flags, etc. Pure function, no model downloads.

### `eljefe.policy`
```python
def hard_route(inp: RouterInput) -> str | None
    # returns "local" | "frontier" when a hard rule fires, else None
def apply_policy(out: RouterOutput, inp: RouterInput,
                 threshold: float = 0.90) -> str  # "local" | "frontier"
```

### `eljefe.metrics`
```python
def routing_metrics(rows: list[RouterRow], p_local: list[float],
                    threshold: float) -> dict
    # keys: quality_retention, local_rate, frontier_rate, cost_reduction,
    #       false_local_rate, false_frontier_rate, oracle_utility, utility
def threshold_sweep(rows, p_local, thresholds=None) -> pandas.DataFrame
def oracle_route(row: RouterRow, quality_floor: float, epsilon: float) -> str
```

### `eljefe.calibration`
```python
class Calibrator:  # wraps isotonic / platt / temperature
    def fit(self, p: np.ndarray, y: np.ndarray) -> "Calibrator"
    def predict(self, p: np.ndarray) -> np.ndarray
    def save(self, path) -> None
    @classmethod
    def load(cls, path) -> "Calibrator"
```

### `eljefe.router`
```python
class Router:  # uniform wrapper over all router kinds
    def predict_proba(self, prompts: list[str],
                      features: list[dict] | None = None) -> np.ndarray
    def predict_gain(self, prompts, features=None) -> np.ndarray  # E[delta_q]
    def save(self, dir) -> None
    @classmethod
    def load(cls, dir) -> "Router"
def train_tfidf(train: list[RouterRow], ...) -> Router
def train_embedding_router(train, embedder_name, ...) -> Router
def train_minilm_router(train, ...) -> Router  # dual-head fine-tune
```

### `eljefe.datasets`
```python
def fetch_source(name: str, cfg: dict) -> Iterable[Task]   # raw -> Task
def normalize(tasks: Iterable[Task]) -> list[Task]         # dedupe + group_id
def load_tasks(path) -> list[Task]
def split_rows(rows: list[ScoredRow], cfg) -> list[RouterRow]  # grouped splits
```

### `eljefe.local_model`
```python
class LocalGenerator:
    def __init__(self, model_id: str, revision: str | None, gen_params: dict)
    def generate(self, tasks: list[Task]) -> Iterable[Generation]  # batched
```

### `eljefe.frontier`
```python
class FrontierClient:
    def __init__(self, cfg: dict)   # provider, model, base_url, budget caps
    def generate(self, tasks: list[Task],
                 out_path: str) -> Iterator[Generation]  # resumable, cached
    def spent_usd(self) -> float
```

## Configs

- `configs/data_v0.yaml` — sources, per-source caps, split ratios, OOD holdout
  families, `quality_floor`, `epsilon`.
- `configs/local_e4b.yaml` — model id, revision, gen params, batch size.
- `configs/frontier.yaml` — provider `fireworks`, model, `max_examples`,
  `max_total_cost_usd`, temperature, cache path.
- `configs/router_baseline.yaml` — tfidf/embedding hyperparams.
- `configs/router_minilm.yaml` — fine-tune hyperparams.

## Conventions

- Seeds: everything takes `seed` from config, default 42.
- Every script writes/updates `artifacts/reports/manifest.json` entries.
- Scripts are runnable both locally and on Colab (`/content/eljefe`); they take
  `--config` and optional `--max-per-source` / `--max-rows` pilot flags.
- No API keys in files. Frontier reads `FIREWORKS_API_KEY` from env.
