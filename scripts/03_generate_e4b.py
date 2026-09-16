#!/usr/bin/env python3
"""Stage 2 — generate local Gemma-4-E4B answers -> data/generations/local_e4b.jsonl.

Resumable: task_ids already in the output file are skipped; rows are appended
incrementally after every batch.
"""

from __future__ import annotations

import argparse
import sys
import time
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
    parser.add_argument("--config", default=str(ROOT / "configs" / "local_e4b.yaml"))
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
    parser.add_argument("--batch-size", type=int, default=None,
                        help="override cfg batch_size")
    args = parse_args(parser)

    import yaml

    from jeff.datasets import load_tasks, update_manifest
    from jeff.local_model import LocalGenerator
    from jeff.schema import completed_ids, write_jsonl

    cfg = yaml.safe_load(Path(args.config).read_text())
    out_path = Path(args.out or cfg.get("out_path") or "data/generations/local_e4b.jsonl")
    if not out_path.is_absolute():
        out_path = ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tasks = load_tasks(args.tasks)
    done = completed_ids(out_path)
    pending = [t for t in tasks if t.id not in done]
    if args.max_rows is not None:
        pending = pending[: args.max_rows]
    print(
        f"[e4b] {len(tasks)} tasks, {len(done)} already done, {len(pending)} pending"
    )
    if not pending:
        print("[e4b] nothing to do")
        return 0

    gen = LocalGenerator(
        model_id=cfg["model_id"],
        revision=cfg.get("revision"),
        gen_params=cfg.get("gen_params"),
        dtype=cfg.get("dtype"),
        device_map=cfg.get("device_map"),
        batch_size=int(args.batch_size or cfg.get("batch_size") or 8),
    )

    batch_size = int(args.batch_size or cfg.get("batch_size") or 8)
    produced = 0
    t0 = time.time()
    buffer = []
    for g in gen.generate(pending):
        buffer.append(g)
        produced += 1
        if len(buffer) >= batch_size:
            write_jsonl(out_path, buffer)
            buffer.clear()
            rate = produced / max(time.time() - t0, 1e-9)
            print(
                f"[e4b] {produced}/{len(pending)} "
                f"({rate:.1f} rows/s, spent {time.time() - t0:.0f}s)",
                flush=True,
            )
    if buffer:
        write_jsonl(out_path, buffer)
    print(f"[done] {produced} generations -> {out_path}")
    update_manifest(
        ROOT,
        "03_generate_e4b",
        {
            "out": str(out_path),
            "model_id": cfg["model_id"],
            "model_revision": gen._model_revision,
            "new_rows": produced,
            "gen_params": cfg.get("gen_params"),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
