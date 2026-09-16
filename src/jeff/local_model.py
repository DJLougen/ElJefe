"""Local model generation (Gemma 4 E4B) via transformers.

Heavy imports (torch, transformers) are lazy — inside methods — so this
module imports with only light deps installed.
"""

from __future__ import annotations

import time
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from .schema import Generation, Task


class LocalGenerator:
    """Batched greedy generation over Tasks.

    The model is loaded lazily on first ``generate`` call so constructing the
    object is cheap and importable without torch installed.
    """

    def __init__(
        self,
        model_id: str,
        revision: str | None = None,
        gen_params: dict | None = None,
        dtype: str | None = None,
        device_map: str | None = None,
        batch_size: int = 8,
    ) -> None:
        self.model_id = model_id
        self.revision = revision
        self.gen_params = dict(gen_params or {})
        self.dtype = dtype
        self.device_map = device_map
        self.batch_size = int(batch_size or 8)
        self._model = None
        self._tokenizer = None
        self._model_revision: str | None = None

    # -- lazy load -----------------------------------------------------------

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        kwargs: dict[str, Any] = {}
        if self.revision:
            kwargs["revision"] = self.revision
        if self.device_map:
            kwargs["device_map"] = self.device_map
        if self.dtype:
            torch_dtype = getattr(torch, self.dtype, None)
            if torch_dtype is None:
                raise ValueError(f"unknown torch dtype {self.dtype!r}")
            try:
                self._model = AutoModelForCausalLM.from_pretrained(
                    self.model_id, dtype=torch_dtype, **kwargs
                )
            except TypeError:  # transformers < 4.56 kwarg name
                self._model = AutoModelForCausalLM.from_pretrained(
                    self.model_id, torch_dtype=torch_dtype, **kwargs
                )
        else:
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_id, **kwargs
            )
        self._model.eval()
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id, **kwargs)
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._tokenizer.padding_side = "left"
        self._model_revision = self._resolve_revision()

    def _resolve_revision(self) -> str | None:
        if self.revision:
            return self.revision
        commit = getattr(getattr(self._model, "config", None), "_commit_hash", None)
        if commit:
            return str(commit)
        # Local path: use the snapshot directory name.
        p = Path(self.model_id)
        if p.exists():
            return p.name
        return None

    # -- prompt formatting ---------------------------------------------------

    def _render(self, task: Task) -> str:
        tok = self._tokenizer
        if getattr(tok, "chat_template", None):
            messages = []
            if task.system_prompt:
                messages.append({"role": "system", "content": task.system_prompt})
            messages.append({"role": "user", "content": task.prompt})
            try:
                return tok.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                pass  # fall back to raw prompt
        if task.system_prompt:
            return f"{task.system_prompt}\n\n{task.prompt}"
        return task.prompt

    # -- generation ----------------------------------------------------------

    def generate(self, tasks: list[Task]) -> Iterator[Generation]:
        """Greedy batched generation; yields one Generation per task."""
        self._load()
        import torch

        gen_kwargs = dict(self.gen_params)
        if not gen_kwargs.get("do_sample"):
            # Greedy: drop sampling knobs so temperature=0.0 isn't passed to
            # HF generate (it warns/errors on temperature with do_sample=False).
            gen_kwargs["do_sample"] = False
            for k in ("temperature", "top_p", "top_k"):
                gen_kwargs.pop(k, None)
        max_new = gen_kwargs.get("max_new_tokens")

        for start in range(0, len(tasks), self.batch_size):
            batch = tasks[start : start + self.batch_size]
            texts = [self._render(t) for t in batch]
            try:
                enc = self._tokenizer(
                    texts, return_tensors="pt", padding=True, truncation=False
                )
                device = getattr(self._model, "device", None)
                if device is not None:
                    enc = {k: v.to(device) for k, v in enc.items()}
                t0 = time.perf_counter()
                with torch.no_grad():
                    out = self._model.generate(
                        **enc,
                        pad_token_id=self._tokenizer.pad_token_id,
                        **gen_kwargs,
                    )
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                input_len = enc["input_ids"].shape[1]
                pad_id = self._tokenizer.pad_token_id
                for i, (task, seq) in enumerate(zip(batch, out)):
                    new_tokens = seq[input_len:]
                    answer = self._tokenizer.decode(
                        new_tokens, skip_special_tokens=True
                    ).strip()
                    # count real generated tokens, not right-side pad fill
                    if pad_id is not None:
                        n_out = int((new_tokens != pad_id).sum().item())
                    else:
                        n_out = int(new_tokens.shape[0])
                    if max_new and n_out >= int(max_new):
                        finish = "length"
                    else:
                        finish = "stop"
                    yield Generation(
                        task_id=task.id,
                        model=self.model_id,
                        model_revision=self._model_revision,
                        answer=answer,
                        input_tokens=int(input_len),
                        output_tokens=n_out,
                        latency_ms=elapsed_ms / len(batch),
                        finish_reason=finish,
                        gen_params=dict(self.gen_params),
                    )
            except Exception as exc:  # batch failure -> per-row error rows
                for task in batch:
                    yield Generation(
                        task_id=task.id,
                        model=self.model_id,
                        model_revision=self._model_revision,
                        answer="",
                        error=f"{type(exc).__name__}: {exc}",
                        gen_params=dict(self.gen_params),
                    )
