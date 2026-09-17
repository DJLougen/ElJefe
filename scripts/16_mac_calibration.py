#!/usr/bin/env python3
"""Mac calibration — run the exact local production model on the calibration set.

Reads data/router/mac_calibration.jsonl (Task rows), generates with mlx_lm
when available (Apple Silicon) and falls back to transformers. Records answer,
TTFT, tokens/sec, wall latency, and peak memory per row into
data/scores/mac_calibration_results.jsonl. No GPU required.
"""

from __future__ import annotations

import argparse
import os
import resource
import sys
import time
from pathlib import Path

try:
    ROOT = Path(__file__).resolve().parents[1]
except NameError:  # colab exec / jupyter kernel has no __file__
    ROOT = Path(os.environ.get('ELJEFE_ROOT') or '/content/eljefe')
    if not (ROOT / 'src').exists():
        ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / "src"))
from eljefe.cli import parse_args

DEFAULT_MODEL = "mlx-community/gemma-4-e4b-it-4bit"


def _peak_rss_bytes() -> int:
    # macOS reports ru_maxrss in bytes; Linux in KiB.
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return int(rss)
    return int(rss) * 1024


def _render_prompt(task, tokenizer) -> str:
    if getattr(tokenizer, "chat_template", None):
        messages = []
        if task.system_prompt:
            messages.append({"role": "system", "content": task.system_prompt})
        messages.append({"role": "user", "content": task.prompt})
        try:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            pass
    if task.system_prompt:
        return f"{task.system_prompt}\n\n{task.prompt}"
    return task.prompt


def _gen_mlx(task, model, tokenizer, max_tokens: int) -> dict:
    from mlx_lm import stream_generate

    prompt = _render_prompt(task, tokenizer)
    t0 = time.perf_counter()
    ttft_ms = None
    text_parts: list[str] = []
    n_out = 0
    gen_tps = None
    peak_mem = None
    for resp in stream_generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens):
        if ttft_ms is None:
            ttft_ms = (time.perf_counter() - t0) * 1000.0
        text_parts.append(getattr(resp, "text", ""))
        n_out = int(getattr(resp, "generation_tokens", n_out))
        gen_tps = getattr(resp, "generation_tps", gen_tps)
        peak_mem = getattr(resp, "peak_memory", peak_mem)
    wall_ms = (time.perf_counter() - t0) * 1000.0
    if gen_tps is None and n_out and wall_ms > ttft_ms:
        gen_tps = n_out / ((wall_ms - (ttft_ms or 0)) / 1000.0)
    mem = int(peak_mem * 2**30) if peak_mem else _peak_rss_bytes()
    return {
        "answer": "".join(text_parts).strip(),
        "ttft_ms": ttft_ms,
        "tokens_per_sec": gen_tps,
        "wall_ms": wall_ms,
        "output_tokens": n_out,
        "memory_bytes": mem,
    }


def _gen_transformers(task, model, tokenizer, max_tokens: int) -> dict:
    import threading

    import torch
    from transformers import TextIteratorStreamer

    prompt = _render_prompt(task, tokenizer)
    enc = tokenizer(prompt, return_tensors="pt")
    device = next(model.parameters()).device
    enc = {k: v.to(device) for k, v in enc.items()}
    streamer = TextIteratorStreamer(
        tokenizer, skip_prompt=True, skip_special_tokens=True
    )
    kwargs = dict(
        **enc,
        max_new_tokens=max_tokens,
        do_sample=False,
        streamer=streamer,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )
    t0 = time.perf_counter()
    thread = threading.Thread(target=model.generate, kwargs=kwargs)
    thread.start()
    ttft_ms = None
    text_parts: list[str] = []
    for chunk in streamer:
        if ttft_ms is None:
            ttft_ms = (time.perf_counter() - t0) * 1000.0
        text_parts.append(chunk)
    thread.join()
    wall_ms = (time.perf_counter() - t0) * 1000.0
    text = "".join(text_parts).strip()
    n_out = len(tokenizer(text)["input_ids"]) if text else 0
    decode_s = (wall_ms - (ttft_ms or 0)) / 1000.0
    tps = (n_out / decode_s) if decode_s > 0 and n_out else None
    return {
        "answer": text,
        "ttft_ms": ttft_ms,
        "tokens_per_sec": tps,
        "wall_ms": wall_ms,
        "output_tokens": n_out,
        "memory_bytes": _peak_rss_bytes(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs" / "local_e4b.yaml"))
    parser.add_argument(
        "--model-path",
        default=os.environ.get("ELJEFE_MAC_MODEL", DEFAULT_MODEL),
        help="local model path or HF id (default: $ELJEFE_MAC_MODEL or "
        f"{DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--calibration",
        default=str(ROOT / "data" / "router" / "mac_calibration.jsonl"),
    )
    parser.add_argument(
        "--out",
        default=str(ROOT / "data" / "scores" / "mac_calibration_results.jsonl"),
    )
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    args = parse_args(parser)

    import json

    from eljefe.datasets import load_tasks, update_manifest
    from eljefe.schema import completed_ids

    tasks = load_tasks(args.calibration)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = completed_ids(out_path)
    pending = [t for t in tasks if t.id not in done]
    if args.max_rows is not None:
        pending = pending[: args.max_rows]
    print(
        f"[mac-cal] {len(tasks)} calibration tasks, {len(done)} done, "
        f"{len(pending)} pending; model={args.model_path}"
    )
    if not pending:
        print("[mac-cal] nothing to do")
        return 0

    backend = None
    model = tokenizer = None
    try:
        from mlx_lm import load as mlx_load

        model, tokenizer = mlx_load(args.model_path)
        backend = "mlx_lm"
    except Exception as exc:
        print(f"[mac-cal] mlx_lm unavailable ({type(exc).__name__}: {exc}); "
              "falling back to transformers")
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        device = (
            "mps"
            if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
            else "cpu"
        )
        try:
            model = AutoModelForCausalLM.from_pretrained(
                args.model_path, dtype="auto"
            ).to(device)
        except TypeError:
            model = AutoModelForCausalLM.from_pretrained(
                args.model_path, torch_dtype="auto"
            ).to(device)
        model.eval()
        backend = f"transformers:{device}"
    print(f"[mac-cal] backend={backend}")

    produced = 0
    with open(out_path, "a", encoding="utf-8") as out_f:
        for task in pending:
            try:
                if backend == "mlx_lm":
                    res = _gen_mlx(task, model, tokenizer, args.max_new_tokens)
                else:
                    res = _gen_transformers(
                        task, model, tokenizer, args.max_new_tokens
                    )
                row = {
                    "task_id": task.id,
                    "model_path": args.model_path,
                    "backend": backend,
                    **res,
                    "error": None,
                }
            except Exception as exc:
                row = {
                    "task_id": task.id,
                    "model_path": args.model_path,
                    "backend": backend,
                    "answer": None,
                    "ttft_ms": None,
                    "tokens_per_sec": None,
                    "wall_ms": None,
                    "output_tokens": None,
                    "memory_bytes": _peak_rss_bytes(),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            out_f.write(json.dumps(row) + "\n")
            out_f.flush()
            produced += 1
            if produced % 10 == 0 or produced == len(pending):
                print(
                    f"[mac-cal] {produced}/{len(pending)} "
                    f"(last ttft={row['ttft_ms'] and round(row['ttft_ms'])}ms, "
                    f"tps={row['tokens_per_sec'] and round(row['tokens_per_sec'], 1)})",
                    flush=True,
                )
    print(f"[done] {produced} rows -> {out_path}")
    update_manifest(
        ROOT,
        "16_mac_calibration",
        {
            "out": str(out_path),
            "model_path": args.model_path,
            "backend": backend,
            "new_rows": produced,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
