#!/usr/bin/env python3
"""Stage 6c — train the frozen-embedding + LightGBM router baseline.

Uses the cached prompt embeddings from 09_embed_prompts.py when present
(recomputes otherwise). Trains a classifier for P(local_sufficient) and a
regressor for E[delta_q] on embedding + metadata features.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=str(REPO_ROOT / "configs" / "router_baseline.yaml"))
    p.add_argument("--data-path", default=str(REPO_ROOT / "data" / "router" / "router_dataset.parquet"))
    p.add_argument("--embeddings", default=None,
                   help="override cfg.embedding.cache_path")
    p.add_argument("--out-dir", default=str(REPO_ROOT / "artifacts" / "baselines" / "embedding"))
    p.add_argument("--threshold", type=float, default=0.9)
    p.add_argument("--max-rows", type=int, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    import numpy as np
    import yaml

    from jeff.metrics import routing_metrics, threshold_sweep
    from jeff.router import (
        load_embedding_cache,
        load_router_dataset,
        predict_proba_rows,
        prompt_key,
        train_embedding_router,
        update_manifest,
    )

    cfg = yaml.safe_load(Path(args.config).read_text()) or {}
    emb_cfg = dict(cfg.get("embedding") or {})
    cache_path = Path(args.embeddings or emb_cfg.get(
        "cache_path", "artifacts/baselines/prompt_embeddings.npy"))
    if not cache_path.is_absolute():
        cache_path = REPO_ROOT / cache_path

    rows = load_router_dataset(args.data_path)
    train = [r for r in rows if r.split == "train"]
    val = [r for r in rows if r.split == "validation"]
    if args.max_rows:
        train = train[: args.max_rows]
    if not train:
        raise SystemExit("no train rows in router dataset")
    print(f"[10] train={len(train)} val={len(val)}")

    # Cached embeddings by prompt key; embed on demand when absent.
    index, emb_matrix = load_embedding_cache(cache_path)
    if emb_matrix is not None:
        print(f"[10] loaded embedding cache {emb_matrix.shape} from {cache_path}")

    def matrix_for(rs):
        if emb_matrix is None:
            return None
        keys = [prompt_key(r.prompt, r.system_prompt) for r in rs]
        if all(k in index for k in keys):
            return emb_matrix[[index[k] for k in keys]]
        return None

    train_emb = matrix_for(train)
    if train_emb is None and emb_matrix is not None:
        print("[10] cache missing some train prompts — embedding on demand")
    router = train_embedding_router(
        train, emb_cfg.get("model"), cfg, embeddings=train_emb)
    out_dir = router.save(args.out_dir)

    metrics: dict = {"train_rows": len(train), "val_rows": len(val),
                     "router": router.meta}
    if val:
        p = predict_proba_rows(router, val, cache_path)
        m = routing_metrics(val, list(map(float, p)), threshold=args.threshold)
        metrics["validation"] = m
        sweep = threshold_sweep(val, list(map(float, p)))
        sweep.to_csv(Path(args.out_dir) / "threshold_sweep.csv", index=False)
        print(f"[10] validation @t={args.threshold}: " +
              json.dumps(m, indent=2, default=str))
    (Path(args.out_dir) / "metrics.json").write_text(
        json.dumps(metrics, indent=2, default=str) + "\n")
    update_manifest("embedding", {
        "router_model": router.meta.get("embedder", "embedding"),
        "seed": cfg.get("seed", 42),
        "train_rows": len(train),
        "val_rows": len(val),
        "out_dir": str(out_dir),
        "validation": metrics.get("validation"),
    })
    print(f"[10] saved router -> {out_dir}")


if __name__ == "__main__":
    main()
