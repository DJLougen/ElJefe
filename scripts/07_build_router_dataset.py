#!/usr/bin/env python3
"""Stage 5 — build the router dataset.

split_rows -> data/router/router_dataset.parquet +
data/router/mac_calibration.jsonl (Task rows) +
artifacts/reports/dataset_report.md
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
import os

try:
    ROOT = Path(__file__).resolve().parents[1]
except NameError:  # colab exec / jupyter kernel has no __file__
    ROOT = Path(os.environ.get('ELJEFE_ROOT') or '/content/eljefe')
    if not (ROOT / 'src').exists():
        ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / "src"))
from eljefe.cli import parse_args


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs" / "data_v0.yaml"))
    parser.add_argument(
        "--scored", default=str(ROOT / "data" / "scores" / "scored.jsonl")
    )
    parser.add_argument(
        "--tasks", default=str(ROOT / "data" / "prompts" / "tasks.jsonl")
    )
    parser.add_argument(
        "--out", default=str(ROOT / "data" / "router" / "router_dataset.parquet")
    )
    parser.add_argument(
        "--mac-out",
        default=str(ROOT / "data" / "router" / "mac_calibration.jsonl"),
    )
    parser.add_argument(
        "--report",
        default=str(ROOT / "artifacts" / "reports" / "dataset_report.md"),
    )
    args = parse_args(parser)

    import pandas as pd
    import yaml

    from eljefe.datasets import load_tasks, split_rows, update_manifest
    from eljefe.schema import ScoredRow, read_jsonl, write_jsonl

    cfg = yaml.safe_load(Path(args.config).read_text())
    scored = list(read_jsonl(args.scored, ScoredRow))
    rows = split_rows(scored, cfg)
    if not rows:
        print("[dataset] no scored rows; nothing to build")
        return 1

    # --- parquet ------------------------------------------------------------
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    records = []
    for r in rows:
        d = r.model_dump()
        d["metadata"] = json.dumps(d["metadata"], sort_keys=True)
        records.append(d)
    df = pd.DataFrame(records)
    df.to_parquet(out_path, index=False)

    # --- mac calibration tasks ----------------------------------------------
    tasks_by_id = {t.id: t for t in load_tasks(args.tasks)}
    mac_ids = {r.id for r in rows if r.split == "mac_calibration"}
    mac_tasks = [tasks_by_id[i] for i in mac_ids if i in tasks_by_id]
    mac_path = Path(args.mac_out)
    mac_path.parent.mkdir(parents=True, exist_ok=True)
    if mac_path.exists():
        mac_path.unlink()
    write_jsonl(mac_path, mac_tasks)

    # --- report ---------------------------------------------------------------
    by_split = Counter(r.split for r in rows)
    by_source = Counter(r.source for r in rows)
    by_family = Counter(r.task_family for r in rows)
    split_source = Counter((r.split, r.source) for r in rows)
    n_local_suff = sum(1 for r in rows if r.local_sufficient)
    n_front_help = sum(1 for r in rows if r.frontier_helpful)
    deltas = [r.delta_q for r in rows]
    deltas_sorted = sorted(deltas)

    def pct(p: float) -> float:
        return deltas_sorted[min(len(deltas_sorted) - 1, int(p * len(deltas_sorted)))]

    lines = [
        "# ElJefe router dataset report",
        "",
        f"- rows: {len(rows)}",
        f"- quality_floor: {cfg.get('quality_floor')}",
        f"- epsilon: {cfg.get('epsilon')}",
        "",
        "## Rows by split",
        "",
    ]
    for split, n in sorted(by_split.items()):
        lines.append(f"- {split}: {n}")
    lines += ["", "## Rows by source", ""]
    for src, n in sorted(by_source.items()):
        lines.append(f"- {src}: {n}")
    lines += ["", "## Rows by task family", ""]
    for fam, n in sorted(by_family.items()):
        lines.append(f"- {fam}: {n}")
    lines += ["", "## Split x source", ""]
    for (split, src), n in sorted(split_source.items()):
        lines.append(f"- {split} / {src}: {n}")
    lines += [
        "",
        "## Class balance",
        "",
        f"- local_sufficient=true: {n_local_suff} ({n_local_suff / len(rows):.1%})",
        f"- frontier_helpful=true: {n_front_help} ({n_front_help / len(rows):.1%})",
        "",
        "## delta_q distribution",
        "",
        f"- min: {deltas_sorted[0]:.3f}",
        f"- p25: {pct(0.25):.3f}",
        f"- median: {pct(0.50):.3f}",
        f"- p75: {pct(0.75):.3f}",
        f"- max: {deltas_sorted[-1]:.3f}",
        f"- mean: {sum(deltas) / len(deltas):.3f}",
        "",
        f"## Mac calibration: {len(mac_tasks)} tasks -> {mac_path}",
    ]
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n")

    print(f"[dataset] {len(rows)} rows -> {out_path}")
    for split, n in sorted(by_split.items()):
        print(f"  {split}: {n}")
    print(f"[dataset] mac calibration: {len(mac_tasks)} -> {mac_path}")
    print(f"[dataset] report -> {report_path}")
    update_manifest(
        ROOT,
        "07_build_router_dataset",
        {
            "out": str(out_path),
            "mac_calibration": str(mac_path),
            "report": str(report_path),
            "rows": len(rows),
            "by_split": dict(by_split),
            "quality_floor": cfg.get("quality_floor"),
            "epsilon": cfg.get("epsilon"),
            "seed": cfg.get("seed", 42),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
