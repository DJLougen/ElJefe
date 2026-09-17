#!/usr/bin/env python3
"""Stage 1b — normalize/dedupe tasks -> data/prompts/tasks.jsonl."""

from __future__ import annotations

import argparse
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
        "--in",
        dest="in_path",
        default=str(ROOT / "data" / "prompts" / "tasks_raw.jsonl"),
    )
    parser.add_argument(
        "--out", default=str(ROOT / "data" / "prompts" / "tasks.jsonl")
    )
    args = parse_args(parser)

    from eljefe.datasets import load_tasks, normalize, update_manifest
    from eljefe.schema import write_jsonl

    raw = load_tasks(args.in_path)
    tasks = normalize(raw)
    n_dupes = len(raw) - len(tasks)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()
    write_jsonl(out_path, tasks)

    by_source = Counter(t.source for t in tasks)
    print(f"[normalize] {len(raw)} raw -> {len(tasks)} tasks ({n_dupes} dupes removed)")
    for src, n in sorted(by_source.items()):
        print(f"  {src}: {n}")
    print(f"[done] -> {out_path}")
    update_manifest(
        ROOT,
        "02_normalize_tasks",
        {
            "in": args.in_path,
            "out": str(out_path),
            "raw": len(raw),
            "kept": len(tasks),
            "dupes_removed": n_dupes,
            "per_source": dict(by_source),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
