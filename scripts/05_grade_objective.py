#!/usr/bin/env python3
"""Stage 4a — grade objective tasks -> data/scores/scored.jsonl.

Joins tasks with local + frontier generations, grades both answers via
jeff.graders.grade, and quarantines rows missing a generation or whose
grader raised.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import os

try:
    ROOT = Path(__file__).resolve().parents[1]
except NameError:  # colab exec / jupyter kernel has no __file__
    ROOT = Path(os.environ.get('JEFF_ROOT') or '/content/jeff')
    if not (ROOT / 'src').exists():
        ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / "src"))
from jeff.cli import parse_args


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs" / "data_v0.yaml"))
    parser.add_argument(
        "--tasks", default=str(ROOT / "data" / "prompts" / "tasks.jsonl")
    )
    parser.add_argument(
        "--local", default=str(ROOT / "data" / "generations" / "local_e4b.jsonl")
    )
    parser.add_argument(
        "--frontier", default=str(ROOT / "data" / "generations" / "frontier.jsonl")
    )
    parser.add_argument(
        "--out", default=str(ROOT / "data" / "scores" / "scored.jsonl")
    )
    parser.add_argument(
        "--quarantine", default=str(ROOT / "data" / "scores" / "quarantine.jsonl")
    )
    args = parse_args(parser)

    from jeff.datasets import load_tasks, update_manifest
    from jeff.graders import grade
    from jeff.schema import Generation, ScoredRow, read_jsonl, write_jsonl

    tasks = {t.id: t for t in load_tasks(args.tasks)}
    local_gens = {g.task_id: g for g in read_jsonl(args.local, Generation)}
    frontier_gens = {g.task_id: g for g in read_jsonl(args.frontier, Generation)}

    scored: list[ScoredRow] = []
    quarantined: list[ScoredRow] = []

    def quarantine(task, lg, fg, reason: str) -> None:
        quarantined.append(
            ScoredRow(
                id=task.id,
                source=task.source,
                task_family=task.task_family,
                prompt=task.prompt,
                system_prompt=task.system_prompt,
                local_model=lg.model if lg else "",
                frontier_model=fg.model if fg else "",
                local_answer=lg.answer if lg else None,
                frontier_answer=fg.answer if fg else None,
                metadata={**task.metadata, "quarantine_reason": reason},
            )
        )

    deferred = 0
    for task_id, task in tasks.items():
        if task.grader == "llm_judge":
            deferred += 1  # judged pairwise by 06_grade_open.py
            continue
        lg = local_gens.get(task_id)
        fg = frontier_gens.get(task_id)
        if lg is None or fg is None:
            missing = [n for n, g in (("local", lg), ("frontier", fg)) if g is None]
            quarantine(task, lg, fg, f"missing generation: {','.join(missing)}")
            continue
        if lg.error or fg.error:
            quarantine(task, lg, fg, f"generation error: {lg.error or fg.error}")
            continue
        try:
            local_res = grade(task, lg.answer)
            frontier_res = grade(task, fg.answer)
        except Exception as exc:
            quarantine(task, lg, fg, f"grader exception: {type(exc).__name__}: {exc}")
            continue
        delta = frontier_res.score - local_res.score
        scored.append(
            ScoredRow(
                id=task.id,
                source=task.source,
                task_family=task.task_family,
                prompt=task.prompt,
                system_prompt=task.system_prompt,
                local_model=lg.model,
                frontier_model=fg.model,
                local_answer=lg.answer,
                frontier_answer=fg.answer,
                local_score=local_res.score,
                frontier_score=frontier_res.score,
                delta_q=delta,
                local_input_tokens=lg.input_tokens,
                local_output_tokens=lg.output_tokens,
                frontier_input_tokens=fg.input_tokens,
                frontier_output_tokens=fg.output_tokens,
                frontier_cost=fg.cost_usd,
                local_latency_ms=lg.latency_ms,
                frontier_latency_ms=fg.latency_ms,
                grader_type=local_res.grader_type,
                grader_confidence=min(
                    local_res.confidence, frontier_res.confidence
                ),
                metadata={
                    **task.metadata,
                    "local_grader_detail": local_res.detail,
                    "frontier_grader_detail": frontier_res.detail,
                },
            )
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()
    write_jsonl(out_path, scored)

    q_path = Path(args.quarantine)
    q_path.parent.mkdir(parents=True, exist_ok=True)
    if q_path.exists():
        q_path.unlink()
    write_jsonl(q_path, quarantined)

    print(f"[grade] {len(scored)} scored -> {out_path}")
    print(f"[grade] {len(quarantined)} quarantined -> {q_path}; {deferred} deferred to 06")
    update_manifest(
        ROOT,
        "05_grade_objective",
        {
            "out": str(out_path),
            "quarantine": str(q_path),
            "scored": len(scored),
            "quarantined": len(quarantined),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
