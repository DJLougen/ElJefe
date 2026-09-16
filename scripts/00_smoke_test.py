#!/usr/bin/env python3
"""Stage 0 — smoke test (plan §16).

Prints Python/torch/CUDA/GPU info, loads a tiny HF tokenizer, verifies the
datasets library can reach the Hub, and writes artifacts/reports/smoke.json.
Prints SMOKE_TEST_OK on success.
"""

from __future__ import annotations

import argparse
import json
import platform
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
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs" / "data_v0.yaml"),
        help="config path (unused; kept for interface consistency)",
    )
    args = parse_args(parser)

    report: dict = {"python": sys.version, "platform": platform.platform()}
    ok = True

    # --- torch / CUDA / GPU -------------------------------------------------
    try:
        import torch

        report["torch"] = torch.__version__
        report["cuda_available"] = bool(torch.cuda.is_available())
        report["cuda_version"] = getattr(torch.version, "cuda", None)
        if torch.cuda.is_available():
            dev = torch.cuda.get_device_properties(0)
            report["gpu"] = {
                "name": dev.name,
                "total_memory_gb": round(dev.total_memory / 2**30, 2),
            }
        else:
            report["gpu"] = None
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            report["mps_available"] = True
    except Exception as exc:
        report["torch_error"] = f"{type(exc).__name__}: {exc}"
        report["cuda_available"] = False
        report["gpu"] = None

    # --- tiny HF tokenizer ---------------------------------------------------
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-gpt2")
        ids = tok("hello jeff")["input_ids"]
        report["tokenizer"] = {
            "model": "hf-internal-testing/tiny-random-gpt2",
            "encoded_len": len(ids),
        }
    except Exception as exc:
        report["tokenizer_error"] = f"{type(exc).__name__}: {exc}"
        ok = False

    # --- datasets can reach HF ----------------------------------------------
    try:
        from datasets import load_dataset

        ds = load_dataset("openai/gsm8k", "main", split="test[:5]")
        report["datasets"] = {"dataset": "openai/gsm8k", "rows": len(ds)}
    except Exception as exc:
        # Fallback: hub reachability without the datasets lib.
        try:
            from huggingface_hub import HfApi

            info = HfApi().dataset_info("openai/gsm8k")
        except Exception as exc2:
            report["datasets_error"] = (
                f"{type(exc).__name__}: {exc}; fallback {type(exc2).__name__}: {exc2}"
            )
            ok = False

    # --- artifact writing -----------------------------------------------------
    try:
        out_dir = ROOT / "artifacts" / "reports"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "smoke.json"
        out_path.write_text(json.dumps(report, indent=2) + "\n")
        report["smoke_json"] = str(out_path)
    except Exception as exc:
        report["artifact_error"] = f"{type(exc).__name__}: {exc}"
        ok = False

    print(json.dumps(report, indent=2))
    if ok:
        print("SMOKE_TEST_OK")
        return 0
    print("SMOKE_TEST_FAILED")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
