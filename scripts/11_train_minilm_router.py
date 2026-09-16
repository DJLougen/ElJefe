#!/usr/bin/env python3
"""Stage 7 — train Jeff v0: fine-tuned dual-head MiniLM router.

Loss = BCE(local_sufficient) + lambda * Huber(delta_q); early stopping on
validation routing utility (quality_retention @ threshold 0.9). Writes
weights, tokenizer, config.yaml and manifest.json to cfg.out_dir
(default artifacts/models/jeff-v0).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import os

try:
    REPO_ROOT = Path(__file__).resolve().parents[1]
except NameError:  # colab exec / jupyter kernel has no __file__
    REPO_ROOT = Path(os.environ.get('JEFF_ROOT') or '/content/jeff')
    if not (REPO_ROOT / 'src').exists():
        REPO_ROOT = Path.cwd()
sys.path.insert(0, str(REPO_ROOT / "src"))
from jeff.cli import parse_args as _jeff_parse_args


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=str(REPO_ROOT / "configs" / "router_minilm.yaml"))
    p.add_argument("--data-path", default=str(REPO_ROOT / "data" / "router" / "router_dataset.parquet"))
    p.add_argument("--out-dir", default=None, help="override cfg.out_dir")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--max-rows", type=int, default=None, help="pilot cap on train rows")
    return _jeff_parse_args(p)


def main() -> None:
    args = parse_args()
    import yaml

    from jeff.router import load_router_dataset, train_minilm_router, update_manifest

    cfg = yaml.safe_load(Path(args.config).read_text()) or {}
    if args.out_dir:
        cfg["out_dir"] = args.out_dir
    if args.epochs is not None:
        cfg.setdefault("train", {})["epochs"] = args.epochs

    rows = load_router_dataset(args.data_path)
    if args.max_rows:
        rows = [r for r in rows if r.split != "train"] + \
               [r for r in rows if r.split == "train"][: args.max_rows]
    if not rows:
        raise SystemExit("empty router dataset")

    router = train_minilm_router(rows, cfg)
    out_dir = Path(cfg.get("out_dir", "artifacts/models/jeff-v0"))
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir
    update_manifest("minilm", {
        "router_model": cfg.get("encoder"),
        "seed": cfg.get("seed", 42),
        "train_rows": router.meta.get("train_rows"),
        "val_rows": router.meta.get("val_rows"),
        "out_dir": str(out_dir),
        "best_val": router.meta.get("best_val"),
    })
    print(f"[11] saved Jeff v0 -> {out_dir}")


if __name__ == "__main__":
    main()
