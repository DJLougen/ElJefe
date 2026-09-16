# Jeff

A very small probabilistic model that predicts whether a request can be
handled by **local Gemma 4 E4B** or is worth escalating to a **frontier
model**. Jeff does not answer the user's question — it estimates the
marginal value of additional compute. See `PLAN_Jeff_Colab_CLI.md` for the
full design and `CONTRACTS.md` for internal interfaces.

## Quickstart

```bash
pip install -e .          # or: uv venv && uv pip install -e .
pytest tests/

# On Colab (see plan §13):
colab new -s jeff --gpu L4
colab exec -s jeff -f scripts/00_smoke_test.py
```

Pipeline order: `scripts/01` → `02` → `03` → `04` → `05` → `07` → `08` →
`09` → `10` → `13`. Fine-tuned Jeff (11/12/14/15) only if baselines justify.
