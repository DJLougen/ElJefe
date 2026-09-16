#!/usr/bin/env python3
"""Stage 6b — embed every unique prompt once and cache the matrix.

Writes cfg.embedding.cache_path (.npy) plus a sibling .ids.json holding
{"model": ..., "keys": [prompt_key, ...]} aligned row-for-row with the
matrix. Resumable: existing keys are never re-embedded.
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


def ids_path(cache_path: Path) -> Path:
    return cache_path.with_suffix(cache_path.suffix + ".ids.json")


def load_cache(cache_path: Path):
    """Return (keys, matrix, cached_model_name)."""
    import numpy as np

    keys: list[str] = []
    emb = np.zeros((0, 0), dtype=np.float32)
    cached_model = None
    ip = ids_path(cache_path)
    if cache_path.exists() and ip.exists():
        meta = json.loads(ip.read_text())
        keys = list(meta.get("keys", []))
        cached_model = meta.get("model")
        emb = np.load(cache_path)
        if emb.shape[0] != len(keys):
            raise SystemExit(
                f"cache corrupt: {cache_path} has {emb.shape[0]} rows "
                f"but {ip} lists {len(keys)} keys")
    return keys, emb, cached_model


def save_cache(cache_path: Path, keys: list[str], emb, model: str) -> None:
    import numpy as np

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, emb)
    ids_path(cache_path).write_text(
        json.dumps({"model": model, "keys": keys}) + "\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=str(REPO_ROOT / "configs" / "router_baseline.yaml"))
    p.add_argument("--data-path", default=str(REPO_ROOT / "data" / "router" / "router_dataset.parquet"))
    p.add_argument("--out", default=None,
                   help="override cfg.embedding.cache_path")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--max-rows", type=int, default=None, help="pilot cap on rows read")
    return _jeff_parse_args(p)


def main() -> None:
    args = parse_args()
    import numpy as np
    import yaml

    from jeff.router import embed_prompts, load_router_dataset, prompt_key

    cfg = yaml.safe_load(Path(args.config).read_text()) or {}
    emb_cfg = dict(cfg.get("embedding") or {})
    model = emb_cfg.get("model", "sentence-transformers/all-MiniLM-L6-v2")
    cache_path = Path(args.out or emb_cfg.get(
        "cache_path", "artifacts/baselines/prompt_embeddings.npy"))
    if not cache_path.is_absolute():
        cache_path = REPO_ROOT / cache_path
    batch_size = int(args.batch_size or emb_cfg.get("batch_size", 64))

    rows = load_router_dataset(args.data_path)
    if args.max_rows:
        rows = rows[: args.max_rows]

    # unique prompts, first-seen order
    seen: dict[str, None] = {}
    for r in rows:
        seen.setdefault(prompt_key(r.prompt, r.system_prompt), None)
    wanted = list(seen)

    keys, emb, cached_model = load_cache(cache_path)
    if cached_model and cached_model != model:
        print(f"[09] WARNING: cache was built with {cached_model}, "
              f"config requests {model} — appending anyway")
    have = set(keys)
    missing = [k for k in wanted if k not in have]
    print(f"[09] unique prompts={len(wanted)} cached={len(keys)} to_embed={len(missing)}")

    if missing:
        missing_set = set(missing)
        by_key: dict[str, tuple[str, str | None]] = {}
        for r in rows:
            k = prompt_key(r.prompt, r.system_prompt)
            if k in missing_set and k not in by_key:
                by_key[k] = (r.prompt, r.system_prompt)
        prompts = [by_key[k][0] for k in missing]
        new_emb = embed_prompts(prompts, model, batch_size=batch_size)
        emb = new_emb if emb.size == 0 else np.vstack([emb, new_emb])
        keys = keys + missing
        save_cache(cache_path, keys, emb, model)
    print(f"[09] cache -> {cache_path} shape={emb.shape}")


if __name__ == "__main__":
    main()
