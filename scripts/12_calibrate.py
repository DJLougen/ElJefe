#!/usr/bin/env python3
"""Stage 8 — fit probability calibrators on VALIDATION predictions only.

For every trained router found under --artifacts-dir, predict
P(local_sufficient) on the validation split, fit a Calibrator
(isotonic | platt | temperature), and persist it to
artifacts/calibration/<router>.cal.pkl. The test split is never touched.
"""

from __future__ import annotations

import argparse
import json
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

# router name -> candidate artifact dir (relative to --artifacts-dir)
ROUTER_DIRS = {
    "tfidf": "baselines/tfidf",
    "embedding": "baselines/embedding",
    "minilm": "models/jeff-v0",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=str(REPO_ROOT / "configs" / "router_baseline.yaml"))
    p.add_argument("--data-path", default=str(REPO_ROOT / "data" / "router" / "router_dataset.parquet"))
    p.add_argument("--artifacts-dir", default=str(REPO_ROOT / "artifacts"))
    p.add_argument("--calibration-dir", default=str(REPO_ROOT / "artifacts" / "calibration"))
    p.add_argument("--embeddings", default=None,
                   help="override cfg.embedding.cache_path for the embedding router")
    p.add_argument("--method", default="isotonic",
                   choices=["isotonic", "platt", "temperature"])
    p.add_argument("--router", action="append", default=None,
                   choices=sorted(ROUTER_DIRS),
                   help="calibrate only these routers (default: all found)")
    p.add_argument("--max-rows", type=int, default=None)
    return _jeff_parse_args(p)


def brier_ece(p, y, n_bins: int = 10) -> tuple[float, float]:
    import numpy as np

    p = np.asarray(p, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    brier = float(np.mean((p - y) ** 2))
    ece = 0.0
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi if hi < 1.0 else p <= hi)
        if m.any():
            ece += float(m.mean()) * abs(float(y[m].mean()) - float(p[m].mean()))
    return brier, ece


def main() -> None:
    args = parse_args()
    import numpy as np
    import yaml

    from jeff.calibration import Calibrator
    from jeff.router import (
        Router,
        labels_for,
        load_router_dataset,
        predict_proba_rows,
        update_manifest,
    )

    cfg = yaml.safe_load(Path(args.config).read_text()) or {}
    emb_cfg = dict(cfg.get("embedding") or {})
    cache_path = Path(args.embeddings or emb_cfg.get(
        "cache_path", "artifacts/baselines/prompt_embeddings.npy"))
    if not cache_path.is_absolute():
        cache_path = REPO_ROOT / cache_path

    rows = load_router_dataset(args.data_path)
    val = [r for r in rows if r.split == "validation"]
    if args.max_rows:
        val = val[: args.max_rows]
    if not val:
        raise SystemExit("no validation rows — calibration must fit on validation only")
    y, _ = labels_for(val)

    artifacts = Path(args.artifacts_dir)
    cal_dir = Path(args.calibration_dir)
    cal_dir.mkdir(parents=True, exist_ok=True)

    wanted = args.router or list(ROUTER_DIRS)
    report: dict[str, dict] = {}
    for name in wanted:
        rdir = artifacts / ROUTER_DIRS[name]
        if not (rdir / "router.pkl").exists():
            print(f"[12] skip {name}: no router at {rdir}")
            continue
        try:
            router = Router.load(rdir)
        except Exception as e:  # e.g. torch missing for minilm locally
            print(f"[12] skip {name}: load failed ({e})")
            continue
        p = predict_proba_rows(router, val, cache_path)
        cal = Calibrator(method=args.method)
        cal.fit(np.asarray(p, dtype=np.float64), y)
        p_cal = cal.predict(np.asarray(p, dtype=np.float64))
        out = cal_dir / f"{name}.cal.pkl"
        cal.save(out)
        b0, e0 = brier_ece(p, y)
        b1, e1 = brier_ece(p_cal, y)
        report[name] = {
            "method": args.method,
            "n_val": len(val),
            "brier_before": b0, "brier_after": b1,
            "ece_before": e0, "ece_after": e1,
            "calibrator": str(out),
        }
        print(f"[12] {name}: ECE {e0:.4f}->{e1:.4f} "
              f"Brier {b0:.4f}->{b1:.4f} -> {out}")

    (cal_dir / "calibration_report.json").write_text(
        json.dumps(report, indent=2) + "\n")
    update_manifest("calibration", {
        "method": args.method,
        "seed": cfg.get("seed", 42),
        "val_rows": len(val),
        "routers": report,
    })


if __name__ == "__main__":
    main()
