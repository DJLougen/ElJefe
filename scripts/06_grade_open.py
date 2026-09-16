#!/usr/bin/env python3
"""Stage 4b — blind pairwise judging for open-ended tasks.

For scored rows whose task grader is ``llm_judge`` or ``reference_similarity``
(or flagged ``needs_judgment``), a frontier model judges local vs frontier
answers in both presentation orders. Consistent verdicts get confidence 1.0;
disagreement collapses to 0.5/0.5 with confidence 0.5 (plan §6 Phase B).

Safe no-op when no such rows exist.
"""

from __future__ import annotations

import argparse
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

JUDGE_GRADERS = {"llm_judge", "reference_similarity"}

JUDGE_TEMPLATE = """You are grading two answers to the same task. Judge correctness, \
completeness, and instruction compliance. Be strict.

## Task
{prompt}

## Answer A
{answer_a}

## Answer B
{answer_b}

Reply with exactly one token: A, B, or TIE."""


def parse_verdict(text: str | None) -> str | None:
    if not text:
        return None
    m = re.search(r"\b(A|B|TIE)\b", text.strip().upper())
    return m.group(1) if m else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs" / "frontier.yaml"))
    parser.add_argument(
        "--tasks", default=str(ROOT / "data" / "prompts" / "tasks.jsonl")
    )
    parser.add_argument(
        "--scored", default=str(ROOT / "data" / "scores" / "scored.jsonl")
    )
    parser.add_argument(
        "--local", default=str(ROOT / "data" / "generations" / "local_e4b.jsonl")
    )
    parser.add_argument(
        "--frontier", default=str(ROOT / "data" / "generations" / "frontier.jsonl")
    )
    parser.add_argument(
        "--max-rows", type=int, default=None, help="pilot cap on judged rows"
    )
    args = parser.parse_args()

    import yaml

    from jeff.datasets import load_tasks, update_manifest
    from jeff.frontier import FrontierClient
    from jeff.schema import Generation, ScoredRow, read_jsonl, write_jsonl

    cfg = yaml.safe_load(Path(args.config).read_text())
    tasks = {t.id: t for t in load_tasks(args.tasks)}
    scored_path = Path(args.scored)
    rows = list(read_jsonl(scored_path, ScoredRow)) if scored_path.exists() else []
    rows_by_id = {r.id: r for r in rows}
    local_path = Path(args.local)
    frontier_path = Path(args.frontier)
    local_gens = (
        {g.task_id: g for g in read_jsonl(local_path, Generation)}
        if local_path.exists()
        else {}
    )
    frontier_gens = (
        {g.task_id: g for g in read_jsonl(frontier_path, Generation)}
        if frontier_path.exists()
        else {}
    )

    # Rows needing judgment: existing scored rows flagged for it, plus
    # llm_judge tasks deferred by 05 (no ScoredRow yet).
    targets: list[ScoredRow] = []
    for row in rows:
        task = tasks.get(row.id)
        if (task is not None and task.grader in JUDGE_GRADERS) or row.metadata.get(
            "needs_judgment"
        ):
            targets.append(row)
    for task in tasks.values():
        if task.grader != "llm_judge" or task.id in rows_by_id:
            continue
        lg, fg = local_gens.get(task.id), frontier_gens.get(task.id)
        if lg is None or fg is None or lg.error or fg.error:
            continue  # nothing to judge; 05's quarantine covers missing gens
        row = ScoredRow(
            id=task.id,
            source=task.source,
            task_family=task.task_family,
            prompt=task.prompt,
            system_prompt=task.system_prompt,
            local_model=lg.model,
            frontier_model=fg.model,
            local_answer=lg.answer,
            frontier_answer=fg.answer,
            local_input_tokens=lg.input_tokens,
            local_output_tokens=lg.output_tokens,
            frontier_input_tokens=fg.input_tokens,
            frontier_output_tokens=fg.output_tokens,
            frontier_cost=fg.cost_usd,
            local_latency_ms=lg.latency_ms,
            frontier_latency_ms=fg.latency_ms,
            grader_type="llm_judge",
            metadata=dict(task.metadata),
        )
        rows.append(row)
        rows_by_id[row.id] = row
        targets.append(row)

    if args.max_rows is not None:
        targets = targets[: args.max_rows]
    if not targets:
        print("[judge] no open-ended rows to judge; no-op")
        return 0

    client = FrontierClient(cfg)
    rng = random.Random(int(cfg.get("seed", 42)))
    judged = 0
    for row in targets:
        if (
            client.max_total_cost_usd is not None
            and client.spent_usd() >= float(client.max_total_cost_usd)
        ):
            print(f"[judge] budget cap reached at ${client.spent_usd():.4f}; stopping")
            break
        answers = {"local": row.local_answer or "", "frontier": row.frontier_answer or ""}
        wins = {"local": 0, "frontier": 0, "tie": 0}
        orders = [("local", "frontier"), ("frontier", "local")]
        rng.shuffle(orders)  # randomize which order is presented first
        for a_key, b_key in orders:
            prompt = JUDGE_TEMPLATE.format(
                prompt=row.prompt, answer_a=answers[a_key], answer_b=answers[b_key]
            )
            text, _info = client.complete(prompt, max_tokens=8)
            verdict = parse_verdict(text)
            if verdict == "TIE":
                wins["tie"] += 1
            elif verdict == "A":
                wins[a_key] += 1
            elif verdict == "B":
                wins[b_key] += 1
            # unparseable verdicts count for neither side
        if wins["local"] == wins["frontier"]:
            row.local_score = row.frontier_score = 0.5
            row.grader_confidence = 0.5 if wins["tie"] < 2 else 1.0
        elif wins["local"] > wins["frontier"]:
            row.local_score, row.frontier_score = 1.0, 0.0
            row.grader_confidence = 1.0 if wins["local"] == 2 else 0.5
        else:
            row.local_score, row.frontier_score = 0.0, 1.0
            row.grader_confidence = 1.0 if wins["frontier"] == 2 else 0.5
        row.delta_q = row.frontier_score - row.local_score
        row.grader_type = "llm_judge"
        row.metadata["judge_wins"] = wins
        row.metadata["judge_model"] = client.model
        judged += 1

    if scored_path.exists():
        scored_path.unlink()
    write_jsonl(scored_path, rows)
    print(
        f"[judge] {judged}/{len(targets)} rows judged, spent=${client.spent_usd():.4f}"
        f" -> {scored_path}"
    )
    update_manifest(
        ROOT,
        "06_grade_open",
        {
            "scored": str(scored_path),
            "judged": judged,
            "judge_model": client.model,
            "spent_usd": client.spent_usd(),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
