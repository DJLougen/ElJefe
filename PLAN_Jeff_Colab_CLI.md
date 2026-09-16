# PLAN.md — Jeff: Hyper-Cheap Local-vs-Frontier Router, Then General Compute Allocator

## 0. Mission

Build **Jeff**, a very small probabilistic model whose first job is:

> Given a user request and the capabilities of the local runtime, estimate whether **Gemma 4 E4B running locally** is sufficient or whether the request is worth escalating to a **frontier model**.

Jeff is **not** a chat model. Jeff does not answer the user's question. Jeff estimates the **marginal value of additional compute**.

The initial product contract is deliberately narrow:

```text
request
   |
   v
 Jeff-0
   |
   +--------------------+
   |                    |
 LOCAL                FRONTIER
   |                    |
Gemma 4 E4B          API model
```

If this works, extend the exact same learning problem to:

- whether to verify a local answer;
- whether to call a tool;
- whether to browse/retrieve;
- whether to spawn a research/coding/critic subagent;
- whether a subagent deserves another turn;
- whether additional parallel subagents are worth their marginal cost.

The long-term abstraction is:

> **Given state `s` and candidate computation `a`, predict the expected marginal utility of performing `a`.**

---

# 1. Core design principles

1. **Local is the default prior.**
   Jeff is not choosing symmetrically between models. It should keep work local unless there is evidence that escalation is worthwhile.

2. **Predict; do not decide policy inside the model.**
   Jeff outputs calibrated probabilities / expected gains. A deterministic harness applies the user's cost, quality, privacy, and latency policy.

3. **Start with supervised learning, not RL.**
   This is initially a selective-prediction / cost-sensitive classification problem. RL is unnecessary until Jeff controls sequential compute allocation.

4. **Train on counterfactual outcomes.**
   For a training prompt, obtain both:
   - the local Gemma 4 E4B result;
   - the frontier result.

   Score both and teach Jeff the difference.

5. **Measure economic performance, not just classifier accuracy.**
   A router can be "accurate" while still making expensive mistakes. The primary outputs are:
   - quality retained vs always-frontier;
   - fraction kept local;
   - frontier calls avoided;
   - estimated API cost avoided;
   - false-local rate;
   - routing regret vs an oracle.

6. **Use deterministic verification whenever possible.**
   Unit tests, exact answers, schema validation, math checks, tool-call validation, and reference answers should be preferred to LLM judges.

7. **Treat privacy as a hard rule, not a learned preference.**
   Requests or files marked `local_only` must never be uploaded to a frontier service regardless of Jeff's probability.

---

# 2. Initial hypothesis

There is a learnable boundary between requests where Gemma 4 E4B is sufficient and requests where a frontier model provides meaningful additional quality.

The useful question is not:

> "Which model is smarter?"

It is:

> "For this request, how much quality do we expect to gain by paying for frontier inference?"

Define:

```text
Q_local     = measured quality of Gemma 4 E4B
Q_frontier  = measured quality of frontier model
delta_q     = Q_frontier - Q_local
```

Jeff should learn at least two targets:

```text
P(local_sufficient | request, runtime)
E[delta_q | request, runtime]
```

The harness then determines the route.

Example output:

```json
{
  "p_local_sufficient": 0.963,
  "expected_frontier_gain": 0.021,
  "uncertainty": 0.034
}
```

The harness can implement policies such as:

```text
LOCAL-FIRST:
    use frontier only if P(local_sufficient) < 0.70

BALANCED:
    use frontier only if P(local_sufficient) < 0.90

QUALITY-FIRST:
    use frontier only if P(local_sufficient) < 0.98
```

Do not bake those thresholds into Jeff.

---

# 3. Model under test

## Local model

Use Google's instruction-tuned checkpoint as the reference local capability:

```text
google/gemma-4-E4B-it
```

The production target is the local Mac deployment, likely an MLX/quantized variant.

Important distinction:

- **quality routing** is primarily model/checkpoint dependent;
- **latency and energy routing** are hardware/runtime dependent.

For v0, generate quality labels using the canonical E4B checkpoint on Colab. Then perform a **Mac calibration set** using the exact MLX/quantized model that will ship locally.

Do not assume the quantized Mac model behaves identically to the BF16/FP16 checkpoint.

## Frontier model

Make the frontier model configurable.

Example configuration:

```yaml
frontier:
  provider: configurable
  model: configurable
  temperature: 0.0

local:
  model: google/gemma-4-E4B-it
  temperature: 0.0
```

For the first experiment, use **one** frontier model consistently. Do not mix several frontier models into one target until the router is working.

Later Jeff can condition on the frontier candidate as another feature.

---

# 4. What Jeff sees

## Jeff-0: pre-inference router

Jeff-0 should only need information available before invoking E4B:

```json
{
  "prompt": "...",
  "system_prompt": "...",
  "input_tokens": 1384,
  "modalities": ["text"],
  "tools_available": ["web", "python"],
  "local_model": "gemma-4-E4B-it",
  "local_context_limit": 131072,
  "privacy": "normal"
}
```

Do not over-engineer metadata initially. The essential input is the prompt.

Useful derived features for cheap baselines:

- input token count;
- code-block count;
- number of files/images;
- modality flags;
- presence of URLs;
- requested output format;
- tool requirement;
- number of conversation turns;
- context length;
- lexical complexity;
- question/task family prediction.

## Jeff-1: post-local verifier

Only build this after Jeff-0.

Jeff-1 sees:

```json
{
  "prompt": "...",
  "local_answer": "...",
  "local_generation_metadata": {
    "tokens": 812,
    "finish_reason": "stop"
  }
}
```

If available from the runtime, optionally include:

- token entropy / logprob statistics;
- repetition indicators;
- malformed structured-output flags;
- verifier outcomes;
- tool-call parse success;
- whether the model expressed uncertainty.

Jeff-1 predicts whether the **actual E4B answer** is sufficient.

This permits a three-way flow:

```text
              Jeff-0
          /      |       \
      LOCAL    UNSURE   FRONTIER
        |        |         |
       E4B      E4B      frontier
                 |
               Jeff-1
              /      \
           ACCEPT   ESCALATE
```

---

# 5. Training target

Do not start with a single hard binary label.

Store the raw outcome information so different routing policies can be trained later.

Canonical row:

```json
{
  "id": "task_000001",
  "source": "math",
  "task_family": "reasoning",
  "prompt": "...",

  "local_model": "google/gemma-4-E4B-it",
  "frontier_model": "FRONTIER_MODEL_ID",

  "local_answer": "...",
  "frontier_answer": "...",

  "local_score": 0.73,
  "frontier_score": 0.96,

  "delta_q": 0.23,

  "local_input_tokens": 540,
  "local_output_tokens": 220,
  "frontier_input_tokens": 540,
  "frontier_output_tokens": 181,

  "frontier_cost": null,
  "local_latency_ms": null,
  "frontier_latency_ms": null,

  "grader_type": "deterministic",
  "grader_confidence": 1.0
}
```

Derived labels:

```text
local_sufficient = local_score >= QUALITY_FLOOR
                   AND delta_q <= EPSILON

frontier_helpful = delta_q > EPSILON
```

Use configurable `QUALITY_FLOOR` and `EPSILON`.

Never discard `local_score`, `frontier_score`, or `delta_q` after deriving labels.

---

# 6. Dataset strategy

## Phase A — objective tasks first

Start with tasks that can be graded without another model.

Useful categories:

### Mathematics / reasoning
- GSM-style arithmetic/reasoning;
- MATH/MATH-500 style problems;
- MMLU-Pro-like multiple-choice reasoning;
- symbolic/logic tasks with known answers.

### Code
- HumanEval+/MBPP+-style tasks;
- small bug fixing tasks with tests;
- function completion;
- deterministic code-output problems.

### Instruction following / structured output
- IFEval-style constraints;
- JSON-schema generation;
- extraction with known fields;
- formatting compliance;
- deterministic transformations.

### Tool selection / function calling
- BFCL-style function-selection cases;
- schema-valid tool calls;
- exact argument extraction.

### Factual QA with references
Use reference-answer datasets where correctness can be computed or verified cheaply.

Goal for the first useful dataset:

```text
~10k–20k examples total
```

This is enough to test whether the routing boundary is learnable. Do not build a million-row dataset before proving that.

## Phase B — open-ended real-user-like prompts

Add a smaller set covering:

- rewriting;
- summarization;
- brainstorming;
- email/message generation;
- ordinary coding questions;
- research questions;
- ambiguous requests;
- long-context synthesis;
- multimodal requests.

For open-ended tasks, use blind pairwise judging.

Randomize answer order:

```text
A = local, B = frontier
```

then separately:

```text
A = frontier, B = local
```

If the two judgments disagree, mark the row low-confidence or send it to a stronger adjudicator.

## Phase C — hard boundary mining

Once Jeff v0 exists, run it over a large unlabeled prompt pool.

Prefer labeling prompts where:

```text
P(local_sufficient) ~= decision boundary
```

or where ensemble members disagree.

This is active learning.

Do **not** keep paying frontier-model costs for examples Jeff already sees as trivial.

---

# 7. Data splits

Avoid a naive random split.

Create:

```text
train
validation
IID test
OOD task-family test
Mac calibration test
```

## Required anti-leakage rules

- deduplicate prompts;
- remove near-duplicates;
- group related benchmark questions before splitting;
- hold out entire sources/task families for an OOD test;
- do not let identical templates with only numbers/names changed cross splits.

The OOD split is important because the product will receive tasks Jeff did not see during training.

---

# 8. Grading hierarchy

Use the cheapest valid grader first.

```text
1. exact answer / exact match
2. unit test
3. schema validation
4. constraint checker
5. reference-based metric
6. cheap LLM judge
7. frontier judge / adjudicator
```

Examples:

```text
code          -> execute tests
JSON          -> parse + validate schema
math          -> normalize answer + exact/semantic numeric match
multiple choice -> exact option
tool call     -> function + arguments match
format task   -> deterministic checker
```

Only open-ended cases should routinely require an LLM judge.

Store both the score and grader provenance.

---

# 9. First models to train

The purpose of the first experiments is to establish the cheapest model that captures most of the routing value.

Train in this order.

## Baseline 0 — no model

Evaluate:

```text
always_local
always_frontier
```

These define the endpoints of the cost-quality curve.

## Baseline 1 — heuristics

Example rules:

- prompt length;
- task family;
- code/non-code;
- context length;
- tool requirement.

This establishes whether trivial rules already solve much of the problem.

## Baseline 2 — TF-IDF + logistic regression

Input:

```text
prompt text
```

Targets:

```text
local_sufficient
```

This should be nearly free to train.

Apply probability calibration afterward.

## Baseline 3 — frozen semantic embeddings + LightGBM/logistic regression

Suggested starting encoder:

```text
sentence-transformers/all-MiniLM-L6-v2
```

Cache one embedding per prompt.

Train:

```text
embedding + cheap metadata -> P(local_sufficient)
```

Also train a regression model:

```text
embedding + metadata -> expected delta_q
```

This is likely the highest-value baseline.

## Jeff v0 — fine-tuned tiny encoder

Only proceed if the frozen embedding model leaves meaningful routing performance on the table.

Initial architecture:

```text
MiniLM-sized encoder
        |
 pooled prompt representation
        |
  +-----+------+
  |            |
sigmoid      regression
  |            |
P(local)    E[delta_q]
```

Loss:

```text
L = BCE(local_sufficient)
    + lambda * Huber(predicted_delta_q, actual_delta_q)
```

Optionally add a third uncertainty head later.

Do not begin with a generative LM.

---

# 10. Calibration

Jeff's probabilities must mean something.

A router that outputs `0.95` but is only correct 70% of the time is dangerous.

Evaluate:

- Brier score;
- expected calibration error;
- reliability curve;
- false-local rate by confidence bucket.

Try:

```text
temperature scaling
isotonic regression
Platt scaling
```

Fit calibration **only on the validation set**.

Freeze it before evaluating the test set.

---

# 11. Primary evaluation

Do not optimize for plain accuracy.

For each routing threshold, simulate the whole system.

## Metrics

### Quality retention

```text
quality_routed / quality_always_frontier
```

### Local rate

```text
local_requests / total_requests
```

### Frontier rate

```text
frontier_requests / total_requests
```

### Estimated monetary cost

Use the actual provider's token pricing at evaluation time.

### Cost reduction

```text
1 - routed_cost / always_frontier_cost
```

### False-local rate

Cases where Jeff selected E4B but frontier would have produced a materially better answer.

This is the critical error class.

### False-frontier rate

Cases where Jeff paid for frontier but E4B would have been sufficient.

This wastes money but usually does not hurt answer quality.

### Oracle regret

Construct an oracle using the measured counterfactual outcomes.

The oracle knows the actual `delta_q`.

Compare Jeff's utility to that oracle.

This tells us how much routing value remains unexploited.

---

# 12. Core graph to produce

The most important figure is a Pareto curve:

```text
Y-axis: quality retained vs always-frontier
X-axis: frontier-call rate OR cost
```

Plot:

- always local;
- always frontier;
- heuristic router;
- TF-IDF baseline;
- frozen embedding router;
- fine-tuned Jeff;
- oracle.

The product is interesting if Jeff moves substantially toward the oracle frontier.

A headline metric should look like:

```text
"At >= X% of frontier quality, Jeff keeps Y% of requests local."
```

Do not pick X/Y in advance. Measure them.

---

# 13. Colab CLI setup

The official Google Colab CLI supports provisioning GPU runtimes, executing local Python files/notebooks, uploading/downloading files, mounting Drive, and tearing the session down.

Install locally on macOS:

```bash
uv tool install google-colab-cli
```

or:

```bash
pip install google-colab-cli
```

Check:

```bash
colab version
```

Provision a persistent development session.

For Gemma 4 E4B generation, prefer an L4 or better because the canonical checkpoint is considerably larger than Jeff itself:

```bash
colab new -s jeff --gpu L4 --high-mem
colab status -s jeff
```

For Jeff-only classifier training, a T4-class runtime should normally be sufficient.

Install dependencies:

```bash
colab install -s jeff -r requirements.txt
```

Execute a local Python file remotely:

```bash
colab exec -s jeff -f scripts/00_smoke_test.py
```

Open a raw terminal if needed:

```bash
colab console -s jeff
```

Upload files:

```bash
colab upload -s jeff configs/v0.yaml /content/jeff/configs/v0.yaml
```

Download artifacts:

```bash
colab download -s jeff /content/jeff/artifacts/jeff-v0.tar.gz ./artifacts/jeff-v0.tar.gz
```

Stop the VM when finished:

```bash
colab stop -s jeff
```

For reproducible one-shot jobs, use:

```bash
colab run --gpu L4 --high-mem scripts/train_router.py -- \
  --config configs/v0.yaml
```

Do not put API keys into committed files or commands that will be archived in logs.

---

# 14. Repository layout

Create:

```text
jeff/
├── PLAN.md
├── README.md
├── requirements.txt
├── pyproject.toml
├── configs/
│   ├── data_v0.yaml
│   ├── local_e4b.yaml
│   ├── frontier.yaml
│   ├── router_baseline.yaml
│   └── router_minilm.yaml
├── data/
│   ├── raw/
│   ├── prompts/
│   ├── generations/
│   ├── scores/
│   └── router/
├── artifacts/
│   ├── baselines/
│   ├── models/
│   ├── calibration/
│   └── reports/
├── scripts/
│   ├── 00_smoke_test.py
│   ├── 01_fetch_tasks.py
│   ├── 02_normalize_tasks.py
│   ├── 03_generate_e4b.py
│   ├── 04_generate_frontier.py
│   ├── 05_grade_objective.py
│   ├── 06_grade_open.py
│   ├── 07_build_router_dataset.py
│   ├── 08_train_tfidf.py
│   ├── 09_embed_prompts.py
│   ├── 10_train_embedding_router.py
│   ├── 11_train_minilm_router.py
│   ├── 12_calibrate.py
│   ├── 13_evaluate.py
│   ├── 14_active_learning.py
│   ├── 15_export_router.py
│   └── 16_mac_calibration.py
├── src/
│   └── jeff/
│       ├── schema.py
│       ├── datasets.py
│       ├── local_model.py
│       ├── frontier.py
│       ├── graders.py
│       ├── features.py
│       ├── router.py
│       ├── calibration.py
│       ├── policy.py
│       └── metrics.py
└── tests/
    ├── test_schema.py
    ├── test_graders.py
    ├── test_policy.py
    └── test_metrics.py
```

---

# 15. Suggested dependencies

Keep this light.

```text
torch
transformers
accelerate
datasets
sentence-transformers
scikit-learn
lightgbm
pandas
numpy
pyarrow
pydantic
pyyaml
tqdm
matplotlib
```

Optional:

```text
bitsandbytes
trl
optimum
onnx
onnxruntime
```

Only add packages when actually required.

---

# 16. Stage-by-stage execution

## Stage 0 — smoke test

`00_smoke_test.py` must:

1. print Python/PyTorch/CUDA versions;
2. print GPU model and memory;
3. load a tiny HF model or tokenizer;
4. verify dataset download;
5. verify artifact writing.

Acceptance:

```text
SMOKE_TEST_OK
```

---

## Stage 1 — acquire and normalize tasks

Run:

```bash
colab exec -s jeff -f scripts/01_fetch_tasks.py
colab exec -s jeff -f scripts/02_normalize_tasks.py
```

Normalize all sources to:

```json
{
  "id": "...",
  "source": "...",
  "task_family": "...",
  "prompt": "...",
  "reference": null,
  "grader": "...",
  "metadata": {}
}
```

Do not allow dataset-specific logic to leak into later stages.

---

## Stage 2 — generate E4B answers

Run:

```bash
colab exec -s jeff -f scripts/03_generate_e4b.py
```

Requirements:

- deterministic/low-temperature decoding;
- batch inference;
- resume from partial outputs;
- write JSONL/Parquet incrementally;
- never regenerate completed IDs unless requested;
- store token counts;
- store generation parameters;
- store checkpoint revision.

Checkpoint frequently.

---

## Stage 3 — generate frontier counterfactuals

Run:

```bash
colab exec -s jeff -f scripts/04_generate_frontier.py
```

Requirements:

- asynchronous/batched API requests where provider permits;
- exponential backoff;
- request IDs;
- cost/token logging;
- restartable cache;
- no duplicate paid calls;
- hard maximum budget configuration.

Example:

```yaml
frontier_generation:
  max_examples: 15000
  max_total_cost_usd: SET_EXPLICITLY
  temperature: 0.0
  cache: true
```

If the budget limit is reached, stop cleanly and retain all completed rows.

---

## Stage 4 — grading

Objective:

```bash
colab exec -s jeff -f scripts/05_grade_objective.py
```

Open-ended:

```bash
colab exec -s jeff -f scripts/06_grade_open.py
```

Each scored example must contain:

```text
local_score
frontier_score
delta_q
grader_type
grader_confidence
```

Reject or quarantine rows with broken graders.

---

## Stage 5 — build router dataset

Run:

```bash
colab exec -s jeff -f scripts/07_build_router_dataset.py
```

Tasks:

- deduplicate;
- group split;
- create IID and OOD tests;
- derive labels;
- calculate class balance;
- emit dataset statistics.

Do not oversample until after the natural distribution has been recorded.

---

## Stage 6 — cheapest baselines

Train TF-IDF:

```bash
colab exec -s jeff -f scripts/08_train_tfidf.py
```

Generate semantic embeddings:

```bash
colab exec -s jeff -f scripts/09_embed_prompts.py
```

Train embedding router:

```bash
colab exec -s jeff -f scripts/10_train_embedding_router.py
```

At this checkpoint answer:

> Is there enough signal in the prompt to predict when E4B needs help?

If the embedding router approaches oracle utility closely, do **not** train a larger router merely for novelty.

---

## Stage 7 — train Jeff v0

Run:

```bash
colab exec -s jeff -f scripts/11_train_minilm_router.py
```

Outputs:

```text
router weights
training config
validation metrics
best checkpoint
tokenizer
feature schema
```

Use early stopping based on routing utility / validation loss, not raw training loss alone.

---

## Stage 8 — calibration

Run:

```bash
colab exec -s jeff -f scripts/12_calibrate.py
```

Fit the chosen calibrator on validation only.

Persist it as a separate artifact.

---

## Stage 9 — full evaluation

Run:

```bash
colab exec -s jeff -f scripts/13_evaluate.py
```

Produce:

```text
metrics.json
threshold_sweep.csv
pareto_curve.png
calibration_curve.png
error_analysis.csv
report.md
```

The report must include failures where Jeff confidently chose local and was wrong.

These are more informative than aggregate accuracy.

---

# 17. Active learning

If v0 demonstrates useful routing signal, do not simply enlarge the dataset uniformly.

Run:

```bash
colab exec -s jeff -f scripts/14_active_learning.py
```

Select examples by:

1. probability near routing threshold;
2. high predictive uncertainty;
3. disagreement between baseline/router models;
4. underrepresented task families;
5. confident errors from the validation/error-analysis set.

Generate frontier counterfactuals only for those selected rows.

Retrain.

Repeat until marginal improvement becomes small.

---

# 18. Mac calibration

The production question is not merely:

> "Can canonical Gemma 4 E4B solve this?"

It is:

> "Can the exact E4B build on this Mac solve this adequately?"

After Colab training, create a fixed calibration set.

Download it:

```bash
colab download -s jeff /content/jeff/data/router/mac_calibration.jsonl ./data/mac_calibration.jsonl
```

Run the exact local production model on the Mac.

Record:

```text
answer quality
time to first token
tokens/sec
wall-clock latency
memory pressure
context failures
```

Use these results to recalibrate Jeff's policy.

Do not necessarily retrain the whole encoder. First try adjusting:

- calibration;
- threshold;
- hardware metadata;
- local-model identifier.

---

# 19. Export

If MiniLM/encoder Jeff wins:

```bash
colab exec -s jeff -f scripts/15_export_router.py
```

Export at least one portable representation:

```text
ONNX
```

Then test:

- FP32;
- FP16 if supported;
- INT8/dynamic quantization.

The winning deployment is the **smallest representation whose routing curve is materially unchanged**.

Jeff should be cheap enough that routing overhead is negligible compared with invoking E4B.

---

# 20. Jeff-1: post-E4B verifier

Only start after Jeff-0 has been evaluated.

Dataset:

```text
prompt
+ E4B answer
+ verifier metadata
-> whether frontier materially improves the result
```

Reuse the same counterfactual labels.

Possible architecture:

```text
tiny encoder over:
"[PROMPT] ... [LOCAL_ANSWER] ..."
```

Targets:

```text
P(local_answer_sufficient)
E[frontier_gain_after_seeing_answer]
```

Then compare:

```text
Jeff-0 only
vs
Jeff-0 + Jeff-1
```

Measure whether Jeff-1 lowers false-local errors without causing too many additional frontier calls.

---

# 21. Hard-routing rules outside Jeff

The harness must be allowed to override Jeff.

Examples:

```text
privacy == local_only       -> never frontier
frontier disabled           -> local
required modality missing   -> compatible route
context > local limit       -> frontier or context strategy
required tool unavailable   -> compatible route
JSON hard validation failed -> verify/escalate
local generation crashed    -> frontier
```

These are capability constraints, not learned preferences.

---

# 22. Transition to subagents

Once local-vs-frontier routing works, generalize Jeff from:

```text
state -> frontier gain
```

to:

```text
(state, candidate_action) -> expected marginal gain
```

Candidate actions:

```text
continue_main
call_frontier
verify
web_search
retrieve_docs
spawn_researcher
spawn_coder
spawn_critic
another_research_turn
another_critic
stop
```

Do **not** immediately build a giant multi-agent controller.

Start with one additional action:

```text
spawn_researcher
```

---

# 23. Subagent training data

For each parent-agent state, run controlled interventions.

Example:

```text
Task T

A: parent alone
B: parent + researcher
```

Measure:

```text
Q_A
Q_B
delta_research = Q_B - Q_A
extra_tokens
extra_cost
extra_latency
```

Training row:

```json
{
  "state": "...",
  "candidate_action": "spawn_researcher",
  "baseline_quality": 0.71,
  "action_quality": 0.87,
  "delta_q": 0.16,
  "action_cost": 0.0031,
  "action_latency_ms": 4100
}
```

Then Jeff learns:

```text
(state, "spawn_researcher") -> E[delta_q]
```

Repeat later for:

```text
spawn_coder
spawn_critic
verify
continue
```

---

# 24. Important causal issue for subagents

Do not train only from actions the current policy happened to choose.

That creates selection bias:

```text
Jeff chooses researcher only for hard tasks
-> logs make researcher look associated with hard/low-quality outcomes
-> model cannot learn the true counterfactual benefit
```

Use randomized exploration on a controlled fraction of training episodes.

For example:

```text
90% policy-selected
10% randomized intervention
```

The exact fraction is configurable.

The randomized episodes are valuable because they reveal:

```text
What happened WITH the subagent
versus
what typically happens WITHOUT it
```

For offline experiments, even better: explicitly fork the same state and run both branches.

---

# 25. Subagent Jeff architecture

Represent each candidate action separately:

```text
state encoder
     |
state embedding ---- action embedding
          \           /
           \         /
             MLP
              |
     expected delta_q
```

This avoids a fixed output head tied to a permanently fixed set of agents.

New actions can be represented using:

```json
{
  "action": "spawn_researcher",
  "expected_cost": 0.003,
  "capabilities": ["web", "citations", "retrieval"]
}
```

Long term:

```text
Jeff(state, action) -> utility distribution
```

The harness scores every permissible action.

---

# 26. Adaptive agent width

For parallel subagents, measure marginal value sequentially.

Example:

```text
critic #1 -> +0.09 expected quality
critic #2 -> +0.02
critic #3 -> +0.003
```

Stop spawning when marginal utility falls below the policy threshold.

This replaces fixed patterns such as:

```python
spawn_critics(n=5)
```

with:

```python
while jeff.expected_gain(next_critic, state) > threshold:
    spawn_critic()
```

---

# 27. Adaptive agent depth

Use the same idea for continuing an existing subagent.

After each step:

```text
current research state
       |
      Jeff
     /    \
 STOP    CONTINUE
```

Train from forked trajectories:

```text
stop now
vs
one more research step
```

Measure the improvement in final answer quality.

This turns Jeff into a learned stopping rule for inference-time computation.

---

# 28. Eventual unified Jeff schema

Long-term input:

```json
{
  "state": {
    "prompt": "...",
    "current_answer": "...",
    "completed_tools": [],
    "subagent_results": [],
    "remaining_context": 22000
  },
  "candidate_action": {
    "type": "spawn_researcher",
    "model": "local_or_frontier",
    "estimated_cost": 0.003,
    "estimated_latency_ms": 4000
  }
}
```

Output:

```json
{
  "expected_quality_gain": 0.084,
  "p_positive_gain": 0.91,
  "uncertainty": 0.07
}
```

Deterministic harness:

```text
utility =
    expected_quality_gain
    - lambda_cost * monetary_cost
    - lambda_latency * latency
    - policy penalties
```

Choose the best legal positive-utility action.

Jeff remains a predictor.

---

# 29. When RL becomes justified

Do not use RL for local-vs-frontier v0.

Consider contextual-bandit/RL methods only when:

1. decisions become sequential;
2. one routing action changes future states;
3. action interactions matter;
4. a reliable episode-level reward exists;
5. enough logged exploration data exists.

Even then, preserve the supervised counterfactual model as a baseline.

The likely progression is:

```text
supervised regression/classification
-> cost-sensitive contextual bandit
-> sequential policy optimization only if necessary
```

---

# 30. Minimum viable experiment

If the goal is to establish whether Jeff is real as cheaply as possible:

## Dataset

```text
10k–20k heterogeneous tasks
```

## Generate

```text
Gemma 4 E4B answer
frontier answer
```

## Grade

Prefer deterministic graders.

## Train

Only:

```text
TF-IDF logistic regression
MiniLM embeddings + LightGBM
```

## Evaluate

Produce the cost-quality Pareto curve.

### Go criterion

Proceed to a custom/fine-tuned Jeff if the learned router materially beats:

```text
always-local
simple heuristic routing
```

and captures a meaningful fraction of the oracle routing gain.

### Stop / rethink criterion

If prompt-only routing barely beats heuristics:

1. add task metadata;
2. train Jeff-1 using the actual E4B answer;
3. test local verifier signals;
4. only then consider a larger router.

Do not solve weak labels with a bigger model.

---

# 31. Experiment matrix

Run and record:

| ID | Router | Input | Target | Purpose |
|---|---|---|---|---|
| E0 | Always local | — | — | lower-cost endpoint |
| E1 | Always frontier | — | — | quality endpoint |
| E2 | Heuristic | metadata | route | trivial baseline |
| E3 | TF-IDF + logistic | prompt | sufficient | lexical baseline |
| E4 | Frozen MiniLM + LightGBM | prompt | sufficient | semantic cheap baseline |
| E5 | Frozen MiniLM + regression | prompt | delta_q | marginal-gain baseline |
| E6 | Fine-tuned MiniLM dual-head | prompt | sufficient + delta_q | Jeff v0 |
| E7 | Jeff v0 + calibration | prompt | calibrated | production candidate |
| E8 | Jeff-1 | prompt + E4B answer | post-answer sufficiency | verifier |
| E9 | Jeff-0 + Jeff-1 cascade | both | system utility | full local/frontier system |
| E10 | Action Jeff | state + action | marginal action gain | first subagent experiment |

---

# 32. Logging

Every experiment should write a machine-readable manifest:

```json
{
  "git_commit": "...",
  "dataset_version": "...",
  "local_model_revision": "...",
  "frontier_model": "...",
  "router_model": "...",
  "seed": 42,
  "train_rows": 0,
  "val_rows": 0,
  "test_rows": 0,
  "quality_floor": 0.0,
  "epsilon": 0.0
}
```

Never produce an untraceable "best model".

---

# 33. Reproducibility

Set seeds for:

```text
Python
NumPy
PyTorch
dataset shuffling
```

Store:

```text
requirements lock
config YAML
dataset manifest
model revision IDs
generation config
grader versions
```

Raw generations should be immutable.

Derived datasets can be regenerated.

---

# 34. Failure analysis categories

For every false-local case, classify the likely cause:

```text
deep reasoning
math
coding
long context
missing knowledge
current-information requirement
instruction-following failure
hallucination
tool-selection failure
format failure
ambiguous prompt
multimodal failure
router confidence/calibration failure
```

The error categories themselves can become future features or curriculum strata.

---

# 35. Product-level benchmark

The end-to-end benchmark should simulate a user workload.

For each prompt:

```text
1. Jeff receives request.
2. Harness selects local/frontier.
3. Selected model produces answer.
4. Answer is graded.
5. Cost and latency are logged.
```

Compare against:

```text
Always E4B
Always frontier
Random route matched for frontier-call rate
Simple heuristic
Jeff
Oracle
```

Report both overall and by task family.

---

# 36. Deliverables from v0

The first serious run is complete only when the repository contains:

```text
artifacts/
├── dataset_report.md
├── router_comparison.csv
├── threshold_sweep.csv
├── pareto_curve.png
├── calibration_curve.png
├── error_analysis.csv
├── metrics.json
├── jeff-v0/
│   ├── model.*
│   ├── tokenizer/
│   ├── calibrator.*
│   ├── config.yaml
│   └── manifest.json
└── report.md
```

`report.md` must answer:

1. Can prompt-only Jeff predict E4B failure?
2. How much frontier usage can be removed at fixed quality-retention levels?
3. Which task families still require frontier most often?
4. How well calibrated are Jeff's probabilities?
5. How close is Jeff to the oracle router?
6. Is a fine-tuned encoder materially better than the frozen-embedding baseline?
7. Does Jeff-1 justify its extra local inference?
8. What should be tried next?

---

# 37. Recommended execution order for the agent

Execute in this order and do not skip cheap baselines:

```text
[ ] initialize repository
[ ] implement schemas/tests
[ ] get Colab CLI smoke test working
[ ] fetch + normalize objective datasets
[ ] generate E4B outputs
[ ] generate frontier outputs with strict cache/budget
[ ] deterministic grading
[ ] build leakage-resistant splits
[ ] always-local / always-frontier metrics
[ ] heuristic baseline
[ ] TF-IDF router
[ ] frozen MiniLM router
[ ] threshold sweep + Pareto curve
[ ] decide whether a fine-tuned Jeff is warranted
[ ] train dual-head Jeff v0
[ ] calibrate
[ ] error analysis
[ ] active-learning round
[ ] Mac calibration using actual local E4B build
[ ] export smallest acceptable router
[ ] build Jeff-1 only if false-local errors warrant it
[ ] begin one-action subagent experiment only after local/frontier routing is validated
```

---

# 38. Immediate commands

From the Mac:

```bash
uv tool install google-colab-cli

colab new -s jeff --gpu L4 --high-mem
colab status -s jeff

colab install -s jeff -r requirements.txt

colab exec -s jeff -f scripts/00_smoke_test.py
colab exec -s jeff -f scripts/01_fetch_tasks.py
colab exec -s jeff -f scripts/02_normalize_tasks.py
colab exec -s jeff -f scripts/03_generate_e4b.py
colab exec -s jeff -f scripts/04_generate_frontier.py
colab exec -s jeff -f scripts/05_grade_objective.py
colab exec -s jeff -f scripts/07_build_router_dataset.py
colab exec -s jeff -f scripts/08_train_tfidf.py
colab exec -s jeff -f scripts/09_embed_prompts.py
colab exec -s jeff -f scripts/10_train_embedding_router.py
colab exec -s jeff -f scripts/13_evaluate.py
```

Only if the baseline results justify it:

```bash
colab exec -s jeff -f scripts/11_train_minilm_router.py
colab exec -s jeff -f scripts/12_calibrate.py
colab exec -s jeff -f scripts/13_evaluate.py
colab exec -s jeff -f scripts/14_active_learning.py
colab exec -s jeff -f scripts/15_export_router.py
```

Retrieve artifacts:

```bash
colab download -s jeff /content/jeff/artifacts/jeff-v0.tar.gz ./artifacts/jeff-v0.tar.gz
```

End session:

```bash
colab stop -s jeff
```

---

# 39. Definition of success

Jeff v0 is successful if it demonstrates a stable, calibrated cost-quality frontier where a substantial fraction of requests can remain on Gemma 4 E4B while retaining nearly all of the measured application quality of the always-frontier baseline.

The exact threshold is a product choice, not a training label.

The important scientific result is:

> **A very small model can predict the marginal value of invoking substantially more capable inference.**

If that holds, the same machinery should be tested on:

```text
frontier escalation
verification
tool calls
retrieval
subagent spawning
subagent continuation
parallel-agent width
```

At that point Jeff becomes a general **inference-time compute allocator** rather than merely a local/cloud router.

---

# 40. External references checked for this plan

- Google Colab CLI: `googlecolab/google-colab-cli`
- Gemma 4 E4B instruction checkpoint: `google/gemma-4-E4B-it`

The Colab CLI command surface in this plan follows the current official Google CLI (`colab new`, `colab exec`, `colab install`, `colab upload`, `colab download`, `colab run`, `colab stop`).
