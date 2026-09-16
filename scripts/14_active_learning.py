#!/usr/bin/env python3
"""Stage 10 — active-learning selection (plan §17).

Scores an unlabeled prompt pool with the best available router and selects
the top-K prompts closest to the routing threshold (plus an uncertainty
bonus). Writes data/prompts/active_selection.jsonl — those rows are the
next candidates for paid frontier counterfactual generation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

# preference order: strongest available router wins
ROUTER_CANDIDATES = [
    ("minilm", "models/jeff-v0"),
    ("embedding", "baselines/embedding"),
    ("tfidf", "baselines/tfidf"),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=str(REPO_ROOT / "configs" / "router_baseline.yaml"))
    p.add_argument("--data-path", default=str(REPO_ROOT / "data" / "router" / "router_dataset.parquet"))
    p.add_argument("--pool", default=str(REPO_ROOT / "data" / "prompts" / "unlabeled.jsonl"),
                   help="JSONL pool of unlabeled prompts; falls back to train prompts")
    p.add_argument("--artifacts-dir", default=str(REPO_ROOT / "artifacts"))
    p.add_argument("--calibration-dir", default=str(REPO_ROOT / "artifacts" / "calibration"))
    p.add_argument("--embeddings", default=None)
    p.add_argument("--out", default=str(REPO_ROOT / "data" / "prompts" / "active_selection.jsonl"))
    p.add_argument("--top-k", type=int, default=200)
    p.add_argument("--threshold", type=float, default=0.9)
    p.add_argument("--max-rows", type=int, default=None)
    return p.parse_args()


def load_pool(path: Path) -> list[dict]:
    """Read a JSONL prompt pool; each row needs a `prompt` field."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if d.get("prompt"):
                rows.append(d)
    return rows


def main() -> None:
    args = parse_args()
    import numpy as np
    import yaml

    from jeff.calibration import Calibrator
    from jeff.router import (
        Router,
        load_router_dataset,
        predict_proba_rows,
        update_manifest,
    )
    from jeff.schema import RouterRow, stable_id

    cfg = yaml.safe_load(Path(args.config).read_text()) or {}
    emb_cfg = dict(cfg.get("embedding") or {})
    cache_path = Path(args.embeddings or emb_cfg.get(
        "cache_path", "artifacts/baselines/prompt_embeddings.npy"))
    if not cache_path.is_absolute():
        cache_path = REPO_ROOT / cache_path

    # ---- pick the best available router ------------------------------------
    artifacts = Path(args.artifacts_dir)
    router = None
    router_name = "heuristic"
    for name, rel in ROUTER_CANDIDATES:
        rdir = artifacts / rel
        if (rdir / "router.pkl").exists():
            try:
                router = Router.load(rdir)
                router_name = name
                break
            except Exception as e:
                print(f"[14] {name} load failed ({e}); trying next")
    if router is None:
        router = Router("heuristic", {"cfg": dict(cfg.get("heuristic") or {})})
    print(f"[14] scoring with router={router_name}")

    cal_path = Path(args.calibration_dir) / f"{router_name}.cal.pkl"
    calibrator = Calibrator.load(cal_path) if cal_path.exists() else None

    # ---- load the pool ------------------------------------------------------
    pool_path = Path(args.pool)
    pool: list[dict] = []
    if pool_path.exists():
        pool = load_pool(pool_path)
        print(f"[14] pool={len(pool)} from {pool_path}")
    else:
        # demo fallback: reuse train prompts as the unlabeled pool
        rows = load_router_dataset(args.data_path)
        pool = [{"id": r.id, "prompt": r.prompt,
                 "system_prompt": r.system_prompt,
                 "task_family": r.task_family, "source": r.source}
                for r in rows if r.split == "train"]
        print(f"[14] no pool at {pool_path}; using {len(pool)} train prompts")
    if args.max_rows:
        pool = pool[: args.max_rows]
    if not pool:
        raise SystemExit("empty prompt pool")

    # ---- score ---------------------------------------------------------------
    shell_rows = [
        RouterRow(
            id=str(d.get("id") or stable_id(d["prompt"])),
            source=str(d.get("source") or "pool"),
            task_family=str(d.get("task_family") or "unknown"),
            prompt=d["prompt"],
            system_prompt=d.get("system_prompt"),
            group_id=str(d.get("id") or stable_id(d["prompt"])),
            split="train",
            local_score=0.0, frontier_score=0.0, delta_q=0.0,
        )
        for d in pool
    ]
    p = np.asarray(predict_proba_rows(router, shell_rows, cache_path),
                   dtype=np.float64)
    if calibrator is not None:
        p = np.asarray(calibrator.predict(p), dtype=np.float64)
    uncertainty = 1.0 - np.abs(2.0 * p - 1.0)          # 1.0 at p=0.5
    margin = np.abs(p - args.threshold)                # 0 at the boundary
    score = margin - 0.1 * uncertainty                 # lower = more informative

    order = np.argsort(score, kind="stable")[: args.top_k]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for rank, i in enumerate(order):
            d = dict(pool[int(i)])
            d.update({
                "id": shell_rows[int(i)].id,
                "router": router_name,
                "p_local": float(p[int(i)]),
                "uncertainty": float(uncertainty[int(i)]),
                "margin_to_threshold": float(margin[int(i)]),
                "selection_rank": rank,
                "threshold": args.threshold,
            })
            f.write(json.dumps(d, default=str) + "\n")
            n += 1
    print(f"[14] wrote {n} selected prompts -> {out_path}")
    update_manifest("active_learning", {
        "router": router_name,
        "pool_size": len(pool),
        "selected": n,
        "threshold": args.threshold,
        "out": str(out_path),
    })


if __name__ == "__main__":
    main()
