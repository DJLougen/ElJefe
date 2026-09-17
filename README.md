# ElJefe — a tiny model that decides when your big model is worth it

ElJefe ("the boss") is a 22M-parameter router that answers one question:

> *Can the cheap local model handle this request, or is it worth paying for
> the frontier model?*

It never answers the user's question itself. It reads the prompt, and in
~5 ms on CPU it outputs two numbers:

- **`p_local`** — probability the local model's answer will be good enough
- **`delta_q`** — expected quality gain if you escalate to the frontier

```
request ──► ElJefe ──┬──► local   Gemma-4-E4B   (free, private, ~0 ms overhead)
                     └──► frontier API model    (costs money, higher ceiling)
```

- **Model:** [DJLougen/eljefe-v0](https://huggingface.co/DJLougen/eljefe-v0)
- **Data:** [DJLougen/eljefe-router-data](https://huggingface.co/datasets/DJLougen/eljefe-router-data)
- **Design doc:** `PLAN_ElJefe_Colab_CLI.md`

---

## Why this exists

Running every request through a frontier model is wasteful — most prompts
(especially math, code, formatting) are handled fine by a 4B local model.
Running everything locally is worse — reasoning-heavy prompts fail silently.
The hard part is *knowing which is which before you pay for the answer*.

That's a learning problem: given a prompt, predict whether the local model
will succeed. ElJefe is the smallest useful version of that idea — a
fine-tuned MiniLM that beats TF-IDF and frozen-embedding baselines and
recovers most of the theoretical maximum routing value.

## Results (v0, held-out test set n=995)

| Router | Quality retention | Traffic kept local | Cost reduction | Wrongly-local |
|---|---|---|---|---|
| always-local | 83.9% | 100% | 100% | 17.1% |
| heuristic rules | 89.0% | 73.3% | 69.1% | 12.4% |
| TF-IDF + logistic | 99.1% | 24.3% | 19.7% | 1.5% |
| frozen embeddings + GBM | 99.9% | 15.3% | 12.6% | 0.7% |
| **ElJefe v0** | **100.6%** | **31.3%** | **25.1%** | 1.2% |
| oracle (perfect router) | 105.4% | 66.6% | 47.6% | 0% |

At a looser threshold (t=0.6): **62% of traffic stays local at 99.6% quality
retention** — most of the oracle's headroom. ROC-AUC 0.861 vs 0.609 heuristic.

> Retention >100% means the routed mix actually *beat* always-frontier on
> those rows — the frontier model isn't perfect either, and keeping some
> prompts local avoided its failures.

---

## Tutorial: use ElJefe in your app

### 1. Install

```bash
pip install onnxruntime transformers huggingface_hub
```

### 2. Load and route

```python
import onnxruntime as ort
from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download

tok = AutoTokenizer.from_pretrained("DJLougen/eljefe-v0", subfolder="tokenizer")
sess = ort.InferenceSession(
    hf_hub_download("DJLougen/eljefe-v0", "onnx/router_fp32.onnx"))

def route(prompt: str, threshold: float = 0.9) -> str:
    enc = tok([prompt], return_tensors="np", truncation=True, max_length=256)
    p_local, delta_q = sess.run(None, dict(enc))
    return "local" if p_local[0] >= threshold else "frontier"

print(route("What is 12 * 37?"))                    # local
print(route("Prove that sqrt(2) is irrational."))    # frontier
```

**Choosing the threshold** is the whole product decision:

| Threshold | Behavior | Use when |
|---|---|---|
| 0.9 | Conservative — only routes local when very confident | Quality matters more than cost |
| 0.6 | Balanced — 62% local, ~99.6% retention | Default cost-saving |
| 0.3 | Aggressive — most traffic local | Cost-critical, tolerant of occasional misses |

The `delta_q` head gives you a second knob: route frontier when
`delta_q > epsilon` even if `p_local` is middling — useful when the downside
of a bad local answer is high.

### 3. Wire it into a real dispatcher

```python
def answer(prompt: str) -> str:
    if route(prompt) == "local":
        return local_model.generate(prompt)      # your E4B/MLX/llama.cpp
    return frontier_client.chat(prompt)          # your API call
```

ElJefe adds ~5 ms and zero marginal cost per request. The savings come from
the requests that *don't* hit the API.

---

## Tutorial: reproduce the training pipeline

Everything below is what we actually ran. Total cost: ~$3 of Colab L4 time +
~$11 of frontier API calls. A T4 works for the classifier stages; E4B
generation wants ≥16 GB VRAM.

### Step 0 — the idea

You need **counterfactual data**: for each prompt, the local model's answer
*and* the frontier model's answer, both graded. Then the label is just:

```
local_sufficient = (local_score >= QUALITY_FLOOR) and (delta_q <= EPSILON)
```

We used `QUALITY_FLOOR=0.70`, `EPSILON=0.10`. Because our graders are binary
(0/1), this collapses to "local passed" — see *Learnings* for why that
matters and when you'd want graded scores instead.

### Step 1 — tasks that grade themselves

```bash
python scripts/01_fetch_tasks.py     # pulls gsm8k, math500, mmlu_pro, mbpp, ifeval
python scripts/02_normalize_tasks.py # -> data/prompts/tasks.jsonl (12,484 rows)
```

Only use tasks with **deterministic gold answers** — exact match, numeric
equivalence, code that runs against tests, constraint checkers. No LLM judge.
That's what makes the labels trustworthy and the whole experiment
reproducible.

### Step 2 — local answers (needs a GPU)

```bash
# On Colab:
colab new -s eljefe --gpu L4
colab upload -s eljefe . /content/eljefe
colab exec -s eljefe -f scripts/03_generate_e4b.py
```

Or locally with any HF model — edit `configs/local_e4b.yaml`. Generation is
resumable; kill and rerun freely.

### Step 3 — frontier counterfactuals (any OpenAI-compatible API)

```bash
export FIREWORKS_API_KEY=...
python scripts/04_generate_frontier.py --config configs/frontier.yaml --dry-run  # cost estimate
python scripts/04_generate_frontier.py --config configs/frontier.yaml
```

Swap the frontier by pointing `base_url`/`model` at any provider — see
`configs/frontier_fw_*.yaml` (Fireworks) and `configs/frontier_ollama_*.yaml`
(Ollama cloud). Hard budget cap: `max_total_cost_usd`.

### Step 4 — grade + build the router dataset

```bash
python scripts/05_grade_objective.py   # deterministic grading both sides
python scripts/07_build_router_dataset.py  # splits + labels -> parquet
```

Splits are group-aware (near-duplicate prompts stay together) and include a
held-out `test_ood` family (ifeval) so you can see out-of-distribution
behavior honestly.

### Step 5 — train, calibrate, evaluate

```bash
python scripts/08_train_tfidf.py              # baseline 1
python scripts/09_embed_prompts.py            # cache MiniLM embeddings
python scripts/10_train_embedding_router.py   # baseline 2 (frozen emb + GBM)
python scripts/11_train_minilm_router.py      # ElJefe v0 (fine-tuned)
python scripts/12_calibrate.py --router minilm
python scripts/13_evaluate.py                 # metrics, Pareto, error analysis
python scripts/15_export_router.py            # ONNX + parity check
```

**Do the baselines first.** If TF-IDF or frozen embeddings already hit your
target, you don't need the fine-tuned model. (Ours barely beat them — see
Learnings.)

### Step 6 — calibrate to *your* local build

The scores above reflect canonical bf16 E4B on an L4. Your quantized MLX/
GGUF build will be slightly different. `scripts/16_mac_calibration.py` +
the 200-row `mac_calibration.jsonl` set measure the exact model you'll ship
and re-fit the operating threshold.

---

## What we learned (read this before replicating)

1. **Label definition > model choice.** An "oracle" label
   (`local_score >= frontier_score`) matching the oracle's argmax rule made
   every router *worse* — 32% local vs 62% at similar retention. The
   tie-sensitive argmax boundary is noisier and less learnable than a
   conservative floor+margin rule.

2. **Binary grading collapses the label.** With 0/1 scores,
   `local_sufficient` reduces to "local passed" — frontier-independent.
   The frontier model only enters through `delta_q`. If you want the label
   to respond to *which* frontier you'd call, use graded rubric scores.

3. **The boundary generalizes across frontier models.** ElJefe trained on
   deepseek-v4-flash counterfactuals keeps ~95% of oracle utility evaluated
   against glm-5p3-flash or gpt-oss-120b. The rescue sets differ (~50%
   Jaccard) but "which prompts are hard for the local model" is stable.

4. **Reasoning is where frontier earns its keep.** Δq by family:
   reasoning 0.34, math 0.09, code 0.08, instruction-following *negative*
   (E4B beats the frontier on IFEval). "Reasoning → frontier" captures most
   of the value.

5. **Fine-tuning ≈ frozen embeddings here.** ElJefe v0 (0.799 utility) vs
   frozen MiniLM + LightGBM (0.793). Start cheap; fine-tune only if it
   clears the baseline materially.

6. **Deterministic grading is the whole ballgame.** No judge = trustworthy,
   reproducible labels — but it restricts you to objectively-gradable tasks.
   Open-ended quality needs a different label source.

7. **Practical traps** (all fixed in-tree): Colab `exec` runs in a Jupyter
   kernel (no `__file__`, kernel argv) — use `colab console` for real
   processes; sessions drop every ~30–40 min — detached subprocess + log
   survives; LightGBM segfaults on macOS arm64 without `n_jobs=1`; Ollama
   cloud rate-limits to ~0.3 req/s regardless of concurrency — useless for
   bulk generation; `torch.onnx.export` needs `dynamo=False` for correct
   `dynamic_axes` on torch 2.11.

## Repo layout

```
src/eljefe/      schema, datasets, graders, features, router, metrics,
                 calibration, policy, frontier client, local model wrapper
scripts/         numbered pipeline stages 00–16
configs/         data/model/frontier/router configs (YAML)
tests/           schema, graders, metrics, policy unit tests
CONTRACTS.md     cross-module interface contracts
PLAN_ElJefe_Colab_CLI.md  full design document
```

## Roadmap

- **Phase B:** open-ended tasks with graded rubrics or a judge.
- **ElJefe-1:** predict whether a local answer needs *verification*.
- **Unified schema:** predict `expected_quality_gain` for arbitrary actions
  (tool call, retrieval, subagent spawn, extra turn) — ElJefe becomes a
  general inference-time compute allocator.

## License

Apache-2.0.
