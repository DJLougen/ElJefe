#!/usr/bin/env python3
"""Stage 11 — export ElJefe v0 (MiniLM dual-head) to ONNX (plan §19).

Exports the encoder + heads to ONNX, optionally produces fp16 and
dynamic-int8 variants, and verifies routing-curve parity against the
PyTorch model on the validation split (max probability deviation).
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
    REPO_ROOT = Path(os.environ.get('ELJEFE_ROOT') or '/content/eljefe')
    if not (REPO_ROOT / 'src').exists():
        REPO_ROOT = Path.cwd()
sys.path.insert(0, str(REPO_ROOT / "src"))
from eljefe.cli import parse_args as _eljefe_parse_args


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=str(REPO_ROOT / "configs" / "router_minilm.yaml"))
    p.add_argument("--data-path", default=str(REPO_ROOT / "data" / "router" / "router_dataset.parquet"))
    p.add_argument("--model-dir", default=str(REPO_ROOT / "artifacts" / "models" / "eljefe-v0"))
    p.add_argument("--out-dir", default=None,
                   help="default: <model-dir>/onnx")
    p.add_argument("--fp16", action="store_true", help="also write an fp16 ONNX")
    p.add_argument("--int8", action="store_true",
                   help="also write a dynamic-int8 ONNX")
    p.add_argument("--max-rows", type=int, default=512,
                   help="cap validation rows used for the parity check")
    return _eljefe_parse_args(p)


def main() -> None:
    args = parse_args()
    import numpy as np
    import torch
    import yaml
    from torch import nn

    from eljefe.router import (
        Router,
        load_router_dataset,
        update_manifest,
    )

    cfg = yaml.safe_load(Path(args.config).read_text()) or {}
    model_dir = Path(args.model_dir)
    out_dir = Path(args.out_dir) if args.out_dir else model_dir / "onnx"
    out_dir.mkdir(parents=True, exist_ok=True)

    router = Router.load(model_dir)
    if router.kind != "minilm":
        raise SystemExit(f"router at {model_dir} is kind={router.kind}, "
                         "expected minilm — run 11_train_minilm_router.py first")
    module, tokenizer = router._ensure_minilm()
    module.eval()

    class _Exportable(nn.Module):
        """input_ids/attention_mask -> (p_local, delta_q) for ONNX export."""

        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, input_ids, attention_mask):
            cls_logit, reg = self.inner(
                input_ids=input_ids, attention_mask=attention_mask)
            return torch.sigmoid(cls_logit), reg

    wrapper = _Exportable(module).eval()
    max_length = int(router.payload.get("max_length", 512))
    dummy = tokenizer(
        ["export dummy prompt"], padding=True, truncation=True,
        max_length=max_length, return_tensors="pt")
    fp32_path = out_dir / "router_fp32.onnx"
    torch.onnx.export(
        wrapper,
        (dummy["input_ids"], dummy["attention_mask"]),
        str(fp32_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["p_local", "delta_q"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "p_local": {0: "batch"},
            "delta_q": {0: "batch"},
        },
        opset_version=17,
        dynamo=False,
    )
    print(f"[15] wrote {fp32_path}")

    variants = {"fp32": fp32_path}
    if args.fp16:
        try:
            import onnx

            try:
                from onnxconverter_common import float16

                convert = float16.convert_float_to_float16
            except ImportError:
                from onnxmltools.utils import float16_converter  # type: ignore

                convert = float16_converter.convert_float_to_float16
            m = onnx.load(str(fp32_path))
            fp16_path = out_dir / "router_fp16.onnx"
            onnx.save(convert(m), str(fp16_path))
            variants["fp16"] = fp16_path
            print(f"[15] wrote {fp16_path}")
        except Exception as e:
            print(f"[15] fp16 conversion skipped ({e})")
    if args.int8:
        try:
            from onnxruntime.quantization import QuantType, quantize_dynamic

            int8_path = out_dir / "router_int8.onnx"
            quantize_dynamic(str(fp32_path), str(int8_path),
                             weight_type=QuantType.QInt8)
            variants["int8"] = int8_path
            print(f"[15] wrote {int8_path}")
        except Exception as e:
            print(f"[15] int8 quantization skipped ({e})")

    # ---- parity check on validation ----------------------------------------
    rows = load_router_dataset(args.data_path)
    val = [r for r in rows if r.split == "validation"] or rows
    if args.max_rows:
        val = val[: args.max_rows]
    prompts = [r.prompt for r in val]
    p_ref = np.asarray(router.predict_proba(prompts), dtype=np.float64)

    parity: dict[str, dict] = {}
    try:
        import onnxruntime as ort
    except ImportError:
        print("[15] onnxruntime not installed — parity check skipped")
        ort = None
    if ort is not None:
        for tag, path in variants.items():
            sess = ort.InferenceSession(
                str(path), providers=["CPUExecutionProvider"])
            outs = []
            for i in range(0, len(prompts), 64):
                enc = tokenizer(
                    prompts[i : i + 64], padding=True, truncation=True,
                    max_length=max_length, return_tensors="np")
                o = sess.run(
                    ["p_local"],
                    {"input_ids": enc["input_ids"].astype(np.int64),
                     "attention_mask": enc["attention_mask"].astype(np.int64)})
                outs.append(np.asarray(o[0]).reshape(-1))
            p_onnx = np.concatenate(outs) if outs else np.zeros(0)
            dev = float(np.max(np.abs(p_onnx - p_ref))) if len(p_onnx) else 0.0
            parity[tag] = {
                "path": str(path),
                "n": int(len(p_onnx)),
                "max_prob_deviation": dev,
                "mean_prob_deviation": float(
                    np.mean(np.abs(p_onnx - p_ref))) if len(p_onnx) else 0.0,
            }
            print(f"[15] parity {tag}: max|dp|={dev:.6f}")

    report = {
        "model_dir": str(model_dir),
        "variants": {k: str(v) for k, v in variants.items()},
        "parity": parity,
        "n_validation": len(val),
    }
    (out_dir / "export_report.json").write_text(
        json.dumps(report, indent=2) + "\n")
    update_manifest("onnx_export", {
        "router_model": router.meta.get("encoder"),
        "seed": cfg.get("seed", 42),
        "out_dir": str(out_dir),
        "parity": parity,
    })
    print(f"[15] export bundle -> {out_dir}")


if __name__ == "__main__":
    main()
