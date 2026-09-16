#!/usr/bin/env python3
"""Stage 6a — train the TF-IDF + logistic regression router baseline.

Reads data/router/router_dataset.parquet, trains on split=='train',
evaluates on split=='validation', saves artifacts/baselines/tfidf/.
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
    p.add_argument("--out-dir", default=str(REPO_ROOT / "artifacts" / "baselines" / "tfidf"))
    p.add_argument("--threshold", type=float, default=0.9,
                   help="routing threshold for headline metrics")
    p.add_argument("--max-rows", type=int, default=None, help="pilot cap on train rows")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    import yaml

    from jeff.metrics import routing_metrics, threshold_sweep
    from jeff.router import load_router_dataset, train_tfidf, update_manifest

    cfg = yaml.safe_load(Path(args.config).read_text()) or {}
    rows = load_router_dataset(args.data_path)
    train = [r for r in rows if r.split == "train"]
    val = [r for r in rows if r.split == "validation"]
    if args.max_rows:
        train = train[: args.max_rows]
    if not train:
        raise SystemExit("no train rows in router dataset")
    print(f"[08] train={len(train)} val={len(val)}")

    router = train_tfidf(train, cfg)
    out_dir = router.save(args.out_dir)

    metrics: dict = {"train_rows": len(train), "val_rows": len(val),
                     "router": router.meta}
    if val:
        p = router.predict_proba([r.prompt for r in val])
        m = routing_metrics(val, list(map(float, p)), threshold=args.threshold)
        metrics["validation"] = m
        sweep = threshold_sweep(val, list(map(float, p)))
        sweep.to_csv(Path(args.out_dir) / "threshold_sweep.csv", index=False)
        print(f"[08] validation @t={args.threshold}: " +
              json.dumps(m, indent=2, default=str))
    (Path(args.out_dir) / "metrics.json").write_text(
        json.dumps(metrics, indent=2, default=str) + "\n")
    update_manifest("tfidf", {
        "router_model": "tfidf+logistic",
        "seed": cfg.get("seed", 42),
        "train_rows": len(train),
        "val_rows": len(val),
        "out_dir": str(out_dir),
        "validation": metrics.get("validation"),
    })
    print(f"[08] saved router -> {out_dir}")


if __name__ == "__main__":
    main()
