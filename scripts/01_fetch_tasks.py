#!/usr/bin/env python3
"""Stage 1a — fetch raw tasks from enabled sources -> data/prompts/tasks_raw.jsonl."""

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
        "--max-per-source",
        type=int,
        default=None,
        help="pilot cap applied per source (overrides config limit)",
    )
    parser.add_argument(
        "--out", default=str(ROOT / "data" / "prompts" / "tasks_raw.jsonl")
    )
    args = parse_args(parser)

    import yaml

    from jeff.datasets import fetch_source, update_manifest
    from jeff.schema import write_jsonl

    cfg = yaml.safe_load(Path(args.config).read_text())
    sources_cfg = cfg.get("sources", {}) or {}

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()  # fetch is a fresh snapshot, not a resume

    total = 0
    per_source: dict[str, int] = {}
    for name, src_cfg in sources_cfg.items():
        if not (src_cfg or {}).get("enabled", True):
            print(f"[skip] {name} (disabled)")
            continue
        eff_cfg = dict(cfg)
        if args.max_per_source is not None:
            eff_cfg = dict(cfg)
            eff_cfg["sources"] = dict(sources_cfg)
            merged = dict(src_cfg or {})
            merged["limit"] = args.max_per_source
            eff_cfg["sources"][name] = merged
        rows = list(fetch_source(name, eff_cfg))
        write_jsonl(out_path, rows)
        per_source[name] = len(rows)
        total += len(rows)
        print(f"[fetch] {name}: {len(rows)} tasks")

    print(f"[done] {total} tasks -> {out_path}")
    update_manifest(
        ROOT,
        "01_fetch_tasks",
        {"out": str(out_path), "total": total, "per_source": per_source},
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
