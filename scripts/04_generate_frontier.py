#!/usr/bin/env python3
"""Stage 3 — generate frontier counterfactuals -> data/generations/frontier.jsonl.

Resumable, budget-capped, cost-logged. --dry-run plans requests and prints
projected cost without calling the API.
"""

from __future__ import annotations

import argparse
import json
import sys
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
    parser.add_argument("--config", default=str(ROOT / "configs" / "frontier.yaml"))
    parser.add_argument(
        "--max-rows",
        "--limit",
        dest="max_rows",
        type=int,
        default=None,
        help="pilot cap on number of tasks to generate",
    )
    parser.add_argument(
        "--tasks", default=str(ROOT / "data" / "prompts" / "tasks.jsonl")
    )
    parser.add_argument("--out", default=None, help="override cfg out_path")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="plan requests and print projected cost; no API calls",
    )
    args = parse_args(parser)

    import yaml

    from eljefe.datasets import load_tasks, update_manifest
    from eljefe.frontier import FrontierClient

    cfg = yaml.safe_load(Path(args.config).read_text())
    out_path = Path(args.out or cfg.get("out_path") or "data/generations/frontier.jsonl")
    if not out_path.is_absolute():
        out_path = ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tasks = load_tasks(args.tasks)
    if args.max_rows is not None:
        tasks = tasks[: args.max_rows]

    client = FrontierClient(cfg)

    if args.dry_run:
        plan = client.plan(tasks, str(out_path))
        print(json.dumps(plan, indent=2))
        return 0

    produced = 0
    for g in client.generate(tasks, str(out_path)):
        produced += 1
        if produced % 25 == 0:
            print(
                f"[frontier] {produced} rows, spent=${client.spent_usd():.4f}",
                flush=True,
            )
    print(
        f"[done] {produced} new generations -> {out_path}; "
        f"total spent=${client.spent_usd():.4f}"
    )
    if client.stop_reason:
        # Clean budget-cap stop is not an error.
        print(f"[budget] stopped early: {client.stop_reason}")
    update_manifest(
        ROOT,
        "04_generate_frontier",
        {
            "out": str(out_path),
            "model": client.model,
            "new_rows": produced,
            "spent_usd": client.spent_usd(),
            "stop_reason": client.stop_reason,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
