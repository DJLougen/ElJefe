#!/usr/bin/env python3
"""Stage 9 — full routing evaluation.

Loads the router dataset plus every available router (always_local,
always_frontier, heuristic, tfidf, embedding, minilm, oracle), applies any
fitted calibrators from artifacts/calibration/, and writes the report
bundle to artifacts/reports/:

    metrics.json, threshold_sweep.csv, router_comparison.csv,
    pareto_curve.png, calibration_curve.png, error_analysis.csv, report.md
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

LEARNED_DIRS = {
    "tfidf": "baselines/tfidf",
    "embedding": "baselines/embedding",
    "minilm": "models/jeff-v0",
}
ENDPOINTS = ("always_local", "always_frontier")
RETENTION_TARGETS = (0.95, 0.98, 0.99, 1.0)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=str(REPO_ROOT / "configs" / "router_baseline.yaml"))
    p.add_argument("--data-config", default=str(REPO_ROOT / "configs" / "data_v0.yaml"),
                   help="for quality_floor / epsilon used by the oracle")
    p.add_argument("--data-path", default=str(REPO_ROOT / "data" / "router" / "router_dataset.parquet"))
    p.add_argument("--artifacts-dir", default=str(REPO_ROOT / "artifacts"))
    p.add_argument("--calibration-dir", default=str(REPO_ROOT / "artifacts" / "calibration"))
    p.add_argument("--embeddings", default=None,
                   help="override cfg.embedding.cache_path for the embedding router")
    p.add_argument("--out-dir", default=str(REPO_ROOT / "artifacts" / "reports"))
    p.add_argument("--split", default="test_iid",
                   help="eval split; falls back to validation when empty")
    p.add_argument("--threshold", type=float, default=0.9,
                   help="operating threshold for headline metrics")
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
        m = (p >= lo) & (p <= hi if hi >= 1.0 else p < hi)
        if m.any():
            ece += float(m.mean()) * abs(float(y[m].mean()) - float(p[m].mean()))
    return brier, ece


def md_table(df) -> str:
    """Render a small DataFrame as a markdown table (no tabulate dep)."""
    cols = [str(c) for c in df.columns]
    body = []
    for _, r in df.iterrows():
        body.append(["" if v is None else
                     (f"{v:.4f}" if isinstance(v, float) else str(v))
                     for v in r.tolist()])
    out = ["| " + " | ".join(cols) + " |",
           "|" + "|".join(["---"] * len(cols)) + "|"]
    out += ["| " + " | ".join(row) + " |" for row in body]
    return "\n".join(out)


def guess_failure_category(row, feats: dict, p_local: float) -> str:
    """Plan §34 failure-category guess for a false-local error."""
    fam = str(row.task_family or "").lower()
    guess = str(feats.get("task_family_guess") or "").lower()
    tok = 0.0
    for k in ("approx_token_count", "context_len", "word_count", "char_count"):
        try:
            tok = float(feats.get(k) or 0)
        except (TypeError, ValueError):
            continue
        if tok:
            break
    try:
        n_code = float(feats.get("code_block_count", 0) or 0)
    except (TypeError, ValueError):
        n_code = 0.0
    if "math" in fam or "math" in guess or "reason" in fam:
        return "math"
    if "code" in fam or "code" in guess or n_code > 0:
        return "coding"
    if tok > 4000:
        return "long context"
    if "instruction" in fam or "ifeval" in str(row.source).lower() \
            or float(feats.get("requested_json", 0) or 0) > 0:
        return "format failure"
    if p_local < 0.6:
        return "ambiguous prompt"
    return "router confidence/calibration failure"


def main() -> None:
    args = parse_args()
    import numpy as np
    import pandas as pd
    import yaml

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from jeff.calibration import Calibrator
    from jeff.metrics import oracle_route, routing_metrics, threshold_sweep
    from jeff.router import (
        Router,
        extract_row_features,
        labels_for,
        load_router_dataset,
        predict_proba_rows,
        update_manifest,
    )

    cfg = yaml.safe_load(Path(args.config).read_text()) or {}
    data_cfg = yaml.safe_load(Path(args.data_config).read_text()) or {} \
        if Path(args.data_config).exists() else {}
    quality_floor = float(data_cfg.get("quality_floor", 0.70))
    epsilon = float(data_cfg.get("epsilon", 0.10))
    heur_cfg = dict(cfg.get("heuristic") or {})
    emb_cfg = dict(cfg.get("embedding") or {})
    cache_path = Path(args.embeddings or emb_cfg.get(
        "cache_path", "artifacts/baselines/prompt_embeddings.npy"))
    if not cache_path.is_absolute():
        cache_path = REPO_ROOT / cache_path

    rows = load_router_dataset(args.data_path)
    eval_rows = [r for r in rows if r.split == args.split]
    split_used = args.split
    if not eval_rows:
        split_used = "validation"
        eval_rows = [r for r in rows if r.split == "validation"]
    if not eval_rows:
        eval_rows = rows
        split_used = "all"
    if args.max_rows:
        eval_rows = eval_rows[: args.max_rows]
    print(f"[13] eval split={split_used} n={len(eval_rows)} "
          f"(quality_floor={quality_floor} epsilon={epsilon})")

    artifacts = Path(args.artifacts_dir)
    cal_dir = Path(args.calibration_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- assemble routers --------------------------------------------------
    routers: dict[str, Router] = {
        "always_local": Router("constant", {"p": 1.0}),
        "always_frontier": Router("constant", {"p": 0.0}),
        "heuristic": Router("heuristic", {"cfg": heur_cfg}),
    }
    for name, rel in LEARNED_DIRS.items():
        rdir = artifacts / rel
        if (rdir / "router.pkl").exists():
            try:
                routers[name] = Router.load(rdir)
                print(f"[13] loaded {name} from {rdir}")
            except Exception as e:
                print(f"[13] skip {name}: load failed ({e})")

    y, _ = labels_for(eval_rows)
    feats = [extract_row_features(r) for r in eval_rows]

    # ---- predict + calibrate ----------------------------------------------
    probs: dict[str, np.ndarray] = {}
    calibrated: dict[str, bool] = {}
    for name, router in routers.items():
        p = np.asarray(
            predict_proba_rows(router, eval_rows, cache_path), dtype=np.float64)
        cal_path = cal_dir / f"{name}.cal.pkl"
        if cal_path.exists() and name not in ENDPOINTS:
            try:
                p = np.asarray(Calibrator.load(cal_path).predict(p),
                               dtype=np.float64)
                calibrated[name] = True
            except Exception as e:
                print(f"[13] calibrator for {name} failed ({e}); using raw")
        probs[name] = p
        calibrated.setdefault(name, False)

    # oracle: knows the counterfactual outcome
    p_oracle = np.array(
        [1.0 if oracle_route(r, quality_floor, epsilon) == "local" else 0.0
         for r in eval_rows],
        dtype=np.float64,
    )
    probs["oracle"] = p_oracle
    calibrated["oracle"] = False

    # ---- metrics -----------------------------------------------------------
    metrics: dict[str, dict] = {}
    sweeps: list[pd.DataFrame] = []
    for name, p in probs.items():
        m = routing_metrics(eval_rows, list(map(float, p)),
                            threshold=args.threshold)
        brier, ece = brier_ece(p, y)
        m["brier"] = brier
        m["ece"] = ece
        metrics[name] = m
        sw = threshold_sweep(eval_rows, list(map(float, p)))
        sw.insert(0, "router", name)
        sweeps.append(sw)

    sweep_df = pd.concat(sweeps, ignore_index=True)
    sweep_df.to_csv(out_dir / "threshold_sweep.csv", index=False)

    comp = pd.DataFrame(
        [{"router": n, **m} for n, m in metrics.items()]
    )
    comp.to_csv(out_dir / "router_comparison.csv", index=False)

    learned = [n for n in probs if n not in ENDPOINTS + ("oracle",)]
    trained = [n for n in learned if n != "heuristic"]

    def _score(n):
        m = metrics[n]
        return (float(m.get("utility", m.get("quality_retention", 0.0)) or 0.0),
                float(m.get("quality_retention", 0.0) or 0.0))

    best = max(trained or learned, key=_score) if (trained or learned) else "heuristic"

    # ---- pareto curve ------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 5))
    for name in probs:
        if name in ENDPOINTS or name == "oracle":
            continue
        sub = sweep_df[sweep_df["router"] == name].sort_values("frontier_rate")
        ax.plot(sub["frontier_rate"], sub["quality_retention"],
                marker=".", label=name)
    ax.scatter([0.0], [metrics["always_local"]["quality_retention"]],
               marker="s", label="always_local", zorder=5)
    ax.scatter([1.0], [metrics["always_frontier"]["quality_retention"]],
               marker="s", label="always_frontier", zorder=5)
    ax.scatter([metrics["oracle"]["frontier_rate"]],
               [metrics["oracle"]["quality_retention"]],
               marker="*", s=180, label="oracle", zorder=6)
    ax.set_xlabel("frontier rate")
    ax.set_ylabel("quality retention (vs always-frontier)")
    ax.set_title(f"Jeff routing Pareto — split={split_used}")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "pareto_curve.png", dpi=150)
    plt.close(fig)

    # ---- calibration curve -------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="perfect")
    edges = np.linspace(0.0, 1.0, 11)
    for name in learned:
        p = probs[name]
        xs, ys = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (p >= lo) & (p <= hi if hi >= 1.0 else p < hi)
            if m.any():
                xs.append(float(p[m].mean()))
                ys.append(float(y[m].mean()))
        ax.plot(xs, ys, marker="o", ms=4,
                label=f"{name} (ECE={metrics[name]['ece']:.3f})")
    ax.set_xlabel("predicted P(local sufficient)")
    ax.set_ylabel("observed local-sufficient rate")
    ax.set_title(f"Calibration — split={split_used}")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "calibration_curve.png", dpi=150)
    plt.close(fig)

    # ---- error analysis ----------------------------------------------------
    err_rows = []
    for name in learned:
        p = probs[name]
        for r, f, pi in zip(eval_rows, feats, p):
            routed_local = pi >= args.threshold
            materially_better = r.delta_q > epsilon
            if routed_local and materially_better:
                err_rows.append({
                    "router": name,
                    "id": r.id,
                    "source": r.source,
                    "task_family": r.task_family,
                    "p_local": float(pi),
                    "local_score": r.local_score,
                    "frontier_score": r.frontier_score,
                    "delta_q": r.delta_q,
                    "failure_category": guess_failure_category(r, f, float(pi)),
                    "prompt_preview": r.prompt[:200],
                })
    pd.DataFrame(err_rows).to_csv(out_dir / "error_analysis.csv", index=False)

    # ---- frontier usage at fixed retention ---------------------------------
    retention_table: dict[str, dict[str, float | None]] = {}
    for name in probs:
        if name in ENDPOINTS:
            continue
        sub = sweep_df[sweep_df["router"] == name]
        row: dict[str, float | None] = {}
        for target in RETENTION_TARGETS:
            ok = sub[sub["quality_retention"] >= target]
            row[f"frontier_rate@ret>={target}"] = (
                float(ok["frontier_rate"].min()) if len(ok) else None)
        retention_table[name] = row

    # ---- metrics.json ------------------------------------------------------
    payload = {
        "split": split_used,
        "n_rows": len(eval_rows),
        "threshold": args.threshold,
        "quality_floor": quality_floor,
        "epsilon": epsilon,
        "best_router": best,
        "routers": metrics,
        "retention_table": retention_table,
    }
    (out_dir / "metrics.json").write_text(
        json.dumps(payload, indent=2, default=str) + "\n")

    # ---- report.md ---------------------------------------------------------
    bm = metrics[best]
    om = metrics["oracle"]
    fam_stats = (
        pd.DataFrame([{
            "task_family": r.task_family,
            "delta_q": r.delta_q,
            "frontier_helpful": bool(r.frontier_helpful)
            if r.frontier_helpful is not None else r.delta_q > epsilon,
        } for r in eval_rows])
        .groupby("task_family")
        .agg(n=("delta_q", "size"), mean_delta_q=("delta_q", "mean"),
             frontier_helpful_rate=("frontier_helpful", "mean"))
        .sort_values("mean_delta_q", ascending=False)
        .reset_index()
    )
    auc_note = ""
    try:
        from sklearn.metrics import roc_auc_score

        if len(np.unique(y)) > 1:
            aucs = {n: float(roc_auc_score(y, probs[n])) for n in learned}
            auc_note = "ROC-AUC: " + ", ".join(
                f"{n}={a:.3f}" for n, a in sorted(aucs.items()))
    except Exception:
        pass

    lines = [
        "# Jeff v0 evaluation report",
        "",
        f"- split: `{split_used}` (n={len(eval_rows)})",
        f"- operating threshold: {args.threshold}",
        f"- quality_floor={quality_floor}, epsilon={epsilon}",
        f"- best learned router: **{best}**",
        "",
        "## Router comparison (operating threshold)",
        "",
        md_table(comp),
        "",
        "## 1. Can prompt-only Jeff predict E4B failure?",
        "",
        f"Best router `{best}` reaches quality_retention="
        f"{bm['quality_retention']:.3f} at frontier_rate="
        f"{bm['frontier_rate']:.3f} (false_local_rate="
        f"{bm['false_local_rate']:.3f}). Heuristic baseline: "
        f"quality_retention={metrics['heuristic']['quality_retention']:.3f}, "
        f"frontier_rate={metrics['heuristic']['frontier_rate']:.3f}. "
        + (auc_note or "ROC-AUC unavailable (single-class labels)."),
        "",
        "## 2. Frontier usage removable at fixed quality retention",
        "",
        md_table(pd.DataFrame(retention_table).T.reset_index()
                 .rename(columns={"index": "router"})),
        "",
        "## 3. Task families that still require frontier most often",
        "",
        md_table(fam_stats),
        "",
        "## 4. Probability calibration",
        "",
        " | ".join(
            f"{n}: ECE={metrics[n]['ece']:.3f} Brier={metrics[n]['brier']:.3f}"
            f"{' (calibrated)' if calibrated[n] else ''}"
            for n in learned),
        "",
        "See calibration_curve.png; calibrators were fit on validation only.",
        "",
        "## 5. Distance to the oracle router",
        "",
        f"Oracle utility={om.get('utility', float('nan')):.4f} "
        f"(frontier_rate={om['frontier_rate']:.3f}, "
        f"quality_retention={om['quality_retention']:.3f}). "
        f"Best router utility={bm.get('utility', float('nan')):.4f}; "
        f"regret={bm.get('regret', float('nan')):.4f}.",
        "",
        "## 6. Fine-tuned encoder vs frozen-embedding baseline",
        "",
    ]
    if "minilm" in metrics and "embedding" in metrics:
        mm, em = metrics["minilm"], metrics["embedding"]
        verdict = ("materially better" if (mm.get("utility") or 0) >
                   (em.get("utility") or 0) + 0.01 else "not materially better")
        lines.append(
            f"minilm utility={mm.get('utility', float('nan')):.4f} vs "
            f"embedding utility={em.get('utility', float('nan')):.4f} — "
            f"fine-tuning is {verdict} on this split.")
    else:
        lines.append(
            "Fine-tuned MiniLM router not present "
            f"(available learned routers: {', '.join(learned) or 'none'}).")
    lines += [
        "",
        "## False-local errors",
        "",
        f"{len(err_rows)} false-local rows across learned routers; "
        "see error_analysis.csv for per-row failure categories.",
        "",
    ]
    (out_dir / "report.md").write_text("\n".join(lines))

    update_manifest("evaluation", {
        "split": split_used,
        "n_rows": len(eval_rows),
        "threshold": args.threshold,
        "quality_floor": quality_floor,
        "epsilon": epsilon,
        "best_router": best,
        "routers": {n: {"quality_retention": m["quality_retention"],
                        "frontier_rate": m["frontier_rate"],
                        "utility": m.get("utility"),
                        "calibrated": calibrated[n]}
                    for n, m in metrics.items()},
    })
    print(f"[13] report bundle -> {out_dir}")
    print(f"[13] best router: {best} "
          f"(quality_retention={bm['quality_retention']:.3f} "
          f"frontier_rate={bm['frontier_rate']:.3f})")


if __name__ == "__main__":
    main()
