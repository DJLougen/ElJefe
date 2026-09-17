# Jeff — a tiny router for local-vs-frontier inference

Jeff is a 22M-parameter model that predicts whether a request can be handled
by a **local model** (Gemma-4-E4B) or is worth escalating to a **frontier
model**. Jeff never answers the user's question — it estimates the *marginal
value of additional compute*.

```
request ──► Jeff ──┬──► local  (Gemma-4-E4B, free, private)
                   └──► frontier (API model, costs money)
```

- **Model:** [DJLougen/jeff-v0](https://huggingface.co/DJLougen/jeff-v0) —
  MiniLM encoder + two heads (`P(local_sufficient)`, `E[delta_q]`),
  PyTorch + ONNX (fp32, 2e-6 parity).
- **Data:** [DJLougen/jeff-router-data](https://huggingface.co/datasets/DJLougen/jeff-router-data) —
  12,475 counterfactual pairs, both sides deterministically graded.
- **Plan:** `PLAN_Jeff_Colab_CLI.md` — the full design doc this repo executes.

## Results (v0, test_iid n=995)

| Router | Quality retention | Kept local | Cost reduction | False-local |
|---|---|---|---|---|
| always-local | 83.9% | 100% | 100% | 17.1% |
| heuristic | 89.0% | 73.3% | 69.1% | 12.4% |
| TF-IDF | 99.1% | 24.3% | 19.7% | 1.5% |
| embedding | 99.9% | 15.3% | 12.6% | 0.7% |
| **Jeff v0** | **100.6%** | **31.3%** | **25.1%** | 1.2% |
| oracle (ceiling) | 105.4% | 66.6% | 47.6% | 0% |

At threshold 0.6: **62% of traffic stays local at 99.6% quality retention.**
ROC-AUC 0.861 vs 0.609 for the heuristic baseline.

## Quickstart — inference

```python
import onnxruntime as ort
from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download

tok = AutoTokenizer.from_pretrained("DJLougen/jeff-v0", subfolder="tokenizer")
sess = ort.InferenceSession(hf_hub_download("DJLougen/jeff-v0", "onnx/router_fp32.onnx"))

enc = tok(["Write a haiku about compilers"], return_tensors="np",
          truncation=True, max_length=256)
p_local, e_delta = sess.run(None, dict(enc))
route = "local" if p_local[0] >= 0.9 else "frontier"
```

## Quickstart — reproduce the whole pipeline

Everything below ran on a Colab L4 (~$3 of GPU time) plus ~$11 of frontier
API calls. A T4 works for the classifier stages; E4B generation wants ≥16GB.

```bash
git clone https://github.com/DJLougen/jeff-router && cd jeff-router
pip install -e . && pytest tests/

# 1. Fetch + normalize objective tasks (gsm8k, math500, mmlu_pro, mbpp, ifeval)
python scripts/01_fetch_tasks.py
python scripts/02_normalize_tasks.py

# 2. Generate local answers — needs a GPU; on Colab:
#    colab new -s jeff --gpu L4 && colab upload ... && colab exec ...
python scripts/03_generate_e4b.py --config configs/local_e4b.yaml

# 3. Generate frontier counterfactuals (any OpenAI-compatible API;
#    set FIREWORKS_API_KEY or point base_url elsewhere)
python scripts/04_generate_frontier.py --config configs/frontier.yaml --dry-run
python scripts/04_generate_frontier.py --config configs/frontier.yaml

# 4. Grade both sides deterministically, build router dataset
python scripts/05_grade_objective.py
python scripts/07_build_router_dataset.py

# 5. Train baselines + Jeff, calibrate, evaluate
python scripts/08_train_tfidf.py
python scripts/09_embed_prompts.py
python scripts/10_train_embedding_router.py
python scripts/11_train_minilm_router.py
python scripts/12_calibrate.py --router minilm
python scripts/13_evaluate.py
python scripts/15_export_router.py   # ONNX export + parity check
```

Swap the frontier model by pointing a config at any OpenAI-compatible
endpoint — see `configs/frontier_fw_*.yaml` (Fireworks) and
`configs/frontier_ollama_*.yaml` (Ollama cloud) for examples. Generation is
resumable and budget-capped (`max_total_cost_usd`).

## Repo layout

```
src/jeff/        schema, datasets, graders, features, router, metrics,
                 calibration, policy, frontier client, local model wrapper
scripts/         numbered pipeline stages 00–16
configs/         data/model/frontier/router configs (YAML)
tests/           schema, graders, metrics, policy unit tests
CONTRACTS.md     cross-module interface contracts
PLAN_Jeff_Colab_CLI.md  full design document
```

## What we learned (the honest version)

1. **The label definition matters more than the model.** We tried an
   "oracle" label (`local_score >= frontier_score`) matching the oracle
   router's argmax rule — every router got *worse* (32% local vs 62% at
   similar retention). The strict floor+margin label creates a cleaner,
   more learnable boundary. Keep the conservative label.

2. **Binary grading collapses the label.** With 0/1 scores,
   `local_sufficient = local_score>=0.7 AND delta_q<=0.1` reduces to "local
   passed" — frontier-independent. The frontier model only matters through
   `delta_q` / `frontier_helpful`. If you want the label itself to respond
   to frontier quality, you need graded (non-binary) scoring.

3. **The boundary generalizes across frontier models.** Jeff trained on
   deepseek-v4-flash counterfactuals retains ~95% of oracle utility when
   evaluated against glm-5p3-flash or gpt-oss-120b outcomes. The *rescue
   sets* differ (~50% Jaccard) but the learnable signal — which prompts are
   hard for the local model — is stable.

4. **Reasoning is where frontier earns its keep.** Δq by family:
   reasoning 0.34, math 0.09, code 0.08, instruction-following *negative*
   (E4B beats the frontier model on IFEval). A router that learns
   "reasoning → frontier" captures most of the value.

5. **Fine-tuning isn't obviously better than frozen embeddings.** Jeff v0
   (fine-tuned MiniLM) ≈ frozen-MiniLM-embeddings + LightGBM on this split
   (0.799 vs 0.793 utility). Start with the cheap baseline; only fine-tune
   if it clears it materially.

6. **Deterministic grading is the whole ballgame.** No LLM judge means the
   counterfactual labels are trustworthy and reproducible — but it also
   restricts you to objectively-gradable tasks. Open-ended quality needs a
   different label source (Phase B in the plan).

7. **Practical traps that cost us time** (all fixed in-tree):
   - Colab `exec` runs in a Jupyter kernel — no `__file__`, argv is the
     kernel's. Run real processes via `colab console` or write scripts that
     tolerate it (`jeff.cli` handles this).
   - Colab sessions drop every ~30–40 min; detached subprocess + log file
     survives; reattach via `SessionState` + `spawn_keep_alive`.
   - LightGBM segfaults on macOS arm64 with default threading —
     `n_jobs=1` fixes it.
   - Ollama cloud rate-limits to ~0.3 req/s regardless of concurrency —
     useless for bulk counterfactual generation; Fireworks did 12.5k rows/hr.
   - `torch.onnx.export` needs `dynamo=False` for correct `dynamic_axes`
     on torch 2.11.

## Roadmap (from the plan)

- **Phase B:** open-ended tasks with a graded rubric or judge.
- **Jeff-1:** predict whether a local answer needs *verification*.
- **Unified schema:** predict `expected_quality_gain` for arbitrary
  candidate actions (tool call, retrieval, subagent spawn, extra turn) —
  Jeff becomes a general inference-time compute allocator.
- **Mac calibration:** `scripts/16_mac_calibration.py` + the 200-row
  `mac_calibration.jsonl` set measure the *exact* local build you'll ship.

## License

Apache-2.0. Dataset card and model card on Hugging Face carry the details.
