"""Frontier model client: OpenAI-compatible chat completions with budget caps.

Resumable: skips task_ids already present in the output JSONL, appends
incrementally, and hard-stops when the cost or example cap is reached.
The API key is read from the environment and never written to files/logs.
"""

from __future__ import annotations

import json
import logging
import os
import random
import threading
import time
from collections.abc import Iterator
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any

from .schema import Generation, Task, completed_ids

log = logging.getLogger("jeff.frontier")


def estimate_cost_usd(
    input_tokens: int,
    output_tokens: int,
    price_input_per_mtok: float,
    price_output_per_mtok: float,
) -> float:
    return (
        input_tokens * float(price_input_per_mtok)
        + output_tokens * float(price_output_per_mtok)
    ) / 1_000_000.0


class FrontierClient:
    """OpenAI-compatible chat.completions client with retry + budget caps.

    cfg keys (see configs/frontier.yaml):
        provider, base_url, api_key_env, model,
        generation: {temperature, max_tokens},
        max_examples, max_total_cost_usd, max_concurrency,
        request_timeout_s, max_retries,
        price_input_per_mtok, price_output_per_mtok, cache
    """

    def __init__(self, cfg: dict) -> None:
        cfg = dict(cfg or {})
        self.provider = cfg.get("provider", "fireworks")
        self.base_url = cfg.get("base_url")
        self.api_key_env = cfg.get("api_key_env", "FIREWORKS_API_KEY")
        self.model = cfg.get("model")
        self.generation = dict(cfg.get("generation") or {})
        self.max_examples = cfg.get("max_examples")
        self.max_total_cost_usd = cfg.get("max_total_cost_usd")
        self.max_concurrency = int(cfg.get("max_concurrency") or 4)
        self.request_timeout_s = float(cfg.get("request_timeout_s") or 120)
        self.max_retries = int(cfg.get("max_retries") or 5)
        self.price_in = float(cfg.get("price_input_per_mtok") or 0.0)
        self.price_out = float(cfg.get("price_output_per_mtok") or 0.0)
        self.cache = bool(cfg.get("cache", True))
        self.seed = int(cfg.get("seed", 42))
        self._rng = random.Random(self.seed)
        self._out_path: str | None = None
        self.stop_reason: str | None = None
        self._lock = threading.Lock()
        self._spent = 0.0
        self._spent_loaded = False
        self._out_path: str | None = None
        self._client = None

    # -- client --------------------------------------------------------------

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI  # lazy import

            api_key = os.environ.get(self.api_key_env)
            if not api_key:
                raise RuntimeError(
                    f"{self.api_key_env} is not set; export it before running "
                    "frontier generation"
                )
            self._client = OpenAI(
                api_key=api_key,
                base_url=self.base_url,
                timeout=self.request_timeout_s,
            )
        return self._client

    # -- cost accounting ------------------------------------------------------

    def _load_prior_spend(self, out_path: str) -> None:
        if self._spent_loaded:
            return
        self._spent = 0.0
        p = Path(out_path)
        if p.exists():
            with open(p, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    cost = row.get("cost_usd")
                    if cost:
                        self._spent += float(cost)
        self._spent_loaded = True

    def spent_usd(self) -> float:
        """Total USD spent, including rows already in the cache file."""
        if self._out_path is not None:
            self._load_prior_spend(self._out_path)
        return self._spent

    def _record_cost(self, cost: float | None) -> None:
        if cost:
            with self._lock:
                self._spent += float(cost)

    # -- single request --------------------------------------------------------

    def _call_once(self, messages: list[dict], max_tokens: int | None = None):
        client = self._get_client()
        kwargs: dict[str, Any] = dict(self.generation)
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        return client.chat.completions.create(
            model=self.model, messages=messages, **kwargs
        )

    def _call_with_retry(self, messages: list[dict], max_tokens: int | None = None):
        """One logical request with exponential backoff on 429/5xx."""
        from openai import (  # lazy import
            APIConnectionError,
            APIStatusError,
            APITimeoutError,
            RateLimitError,
        )

        delay = 1.0
        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                return self._call_once(messages, max_tokens)
            except (RateLimitError, APIConnectionError, APITimeoutError) as exc:
                last_exc = exc
            except APIStatusError as exc:
                last_exc = exc
                status = getattr(exc, "status_code", 0) or 0
                if status < 500 and status != 429:
                    raise  # client error: not retryable
            except Exception:
                raise
            if attempt < self.max_retries:
                sleep_s = delay * (0.5 + self._rng.random())
                log.warning(
                    "frontier request failed (attempt %d/%d), retrying in %.1fs",
                    attempt + 1,
                    self.max_retries + 1,
                    sleep_s,
                )
                time.sleep(sleep_s)
                delay = min(delay * 2.0, 60.0)
        assert last_exc is not None
        raise last_exc

    def complete(
        self, prompt: str, system: str | None = None, max_tokens: int | None = None
    ) -> tuple[str | None, dict[str, Any]]:
        """Single completion for ad-hoc use (e.g. judging). Returns
        (content, info) where info has input_tokens/output_tokens/cost_usd.
        Cost is added to spent_usd()."""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        resp = self._call_with_retry(messages, max_tokens)
        usage = getattr(resp, "usage", None)
        in_tok = int(getattr(usage, "prompt_tokens", 0) or 0)
        out_tok = int(getattr(usage, "completion_tokens", 0) or 0)
        cost = estimate_cost_usd(in_tok, out_tok, self.price_in, self.price_out)
        self._record_cost(cost)
        content = None
        if resp.choices:
            content = resp.choices[0].message.content
        return content, {
            "input_tokens": in_tok,
            "output_tokens": out_tok,
            "cost_usd": cost,
            "finish_reason": resp.choices[0].finish_reason if resp.choices else None,
        }

    # -- task generation --------------------------------------------------------

    def _task_messages(self, task: Task) -> list[dict]:
        messages = []
        if task.system_prompt:
            messages.append({"role": "system", "content": task.system_prompt})
        messages.append({"role": "user", "content": task.prompt})
        return messages

    def _generate_one(self, task: Task) -> Generation:
        t0 = time.perf_counter()
        try:
            resp = self._call_with_retry(self._task_messages(task))
            latency_ms = (time.perf_counter() - t0) * 1000.0
            usage = getattr(resp, "usage", None)
            in_tok = int(getattr(usage, "prompt_tokens", 0) or 0)
            out_tok = int(getattr(usage, "completion_tokens", 0) or 0)
            cost = estimate_cost_usd(in_tok, out_tok, self.price_in, self.price_out)
            self._record_cost(cost)
            answer = ""
            finish = None
            if resp.choices:
                answer = resp.choices[0].message.content or ""
                finish = resp.choices[0].finish_reason
            return Generation(
                task_id=task.id,
                model=self.model,
                model_revision=None,
                answer=answer,
                input_tokens=in_tok or None,
                output_tokens=out_tok or None,
                latency_ms=latency_ms,
                cost_usd=cost,
                finish_reason=finish,
                gen_params=dict(self.generation),
            )
        except Exception as exc:
            return Generation(
                task_id=task.id,
                model=self.model,
                answer="",
                error=f"{type(exc).__name__}: {exc}",
                gen_params=dict(self.generation),
            )

    def plan(self, tasks: list[Task], out_path: str) -> dict[str, Any]:
        """Dry-run plan: how many requests would run and projected cost.

        Output tokens are unknown before calling; the projection uses
        generation.max_tokens as a worst-case bound.
        """
        self._out_path = out_path
        done = completed_ids(out_path) if self.cache else set()
        pending = [t for t in tasks if t.id not in done]
        if self.max_examples is not None:
            remaining = max(0, int(self.max_examples) - len(done))
            pending = pending[:remaining]
        max_out = int(self.generation.get("max_tokens") or 0)
        # Rough input estimate: ~4 chars/token.
        est_in = sum(max(1, len(t.prompt) // 4) for t in pending)
        est_cost = estimate_cost_usd(
            est_in, max_out * len(pending), self.price_in, self.price_out
        )
        return {
            "model": self.model,
            "total_tasks": len(tasks),
            "already_done": len(done),
            "pending": len(pending),
            "estimated_input_tokens": est_in,
            "assumed_output_tokens_per_request": max_out,
            "projected_cost_usd_worst_case": round(est_cost, 4),
            "spent_usd_so_far": round(self.spent_usd(), 4)
            if self._out_path
            else None,
            "max_total_cost_usd": self.max_total_cost_usd,
        }

    def generate(self, tasks: list[Task], out_path: str) -> Iterator[Generation]:
        """Resumable generation. Appends each completed row to out_path.

        Stops submitting new work once spent >= max_total_cost_usd or the
        number of completed rows reaches max_examples; in-flight requests are
        drained and kept.
        """
        self._out_path = out_path
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        self._load_prior_spend(out_path)
        done = completed_ids(out_path) if self.cache else set()
        pending = [t for t in tasks if t.id not in done]
        if self.max_examples is not None:
            # max_examples caps total generated rows, including cached ones.
            remaining = max(0, int(self.max_examples) - len(done))
            pending = pending[:remaining]
        log.info(
            "frontier generate: %d tasks, %d cached, %d pending, spent=$%.4f",
            len(tasks),
            len(done),
            len(pending),
            self._spent,
        )

        produced = 0
        stop_reason: str | None = None
        with open(out_path, "a", encoding="utf-8") as out_f:
            with ThreadPoolExecutor(max_workers=self.max_concurrency) as pool:
                in_flight: dict[Any, Task] = {}
                idx = 0

                def submit_next() -> bool:
                    nonlocal idx, stop_reason
                    while idx < len(pending):
                        if (
                            self.max_total_cost_usd is not None
                            and self._spent >= float(self.max_total_cost_usd)
                        ):
                            stop_reason = (
                                f"budget cap reached: spent ${self._spent:.4f} "
                                f">= ${float(self.max_total_cost_usd):.4f}"
                            )
                            return False
                        fut = pool.submit(self._generate_one, pending[idx])
                        in_flight[fut] = pending[idx]
                        idx += 1
                        return True
                    return False

                for _ in range(self.max_concurrency):
                    if not submit_next():
                        break
                while in_flight:
                    finished, _ = wait(in_flight, return_when=FIRST_COMPLETED)
                    for fut in finished:
                        in_flight.pop(fut)
                        gen = fut.result()
                        if gen.error:
                            # Errors go to a sidecar file so they never poison
                            # the resume cache — a retry will re-attempt them.
                            err_path = out_path + ".errors.jsonl"
                            with open(err_path, "a", encoding="utf-8") as ef:
                                ef.write(gen.model_dump_json() + "\n")
                        else:
                            out_f.write(gen.model_dump_json() + "\n")
                            out_f.flush()
                        produced += 1
                        yield gen
                        submit_next()
                if stop_reason:
                    log.warning("frontier generation stopped early: %s", stop_reason)

        self.stop_reason = stop_reason
        log.info(
            "frontier generate done: %d new rows, total spent=$%.4f%s",
            produced,
            self._spent,
            f" ({stop_reason})" if stop_reason else "",
        )
