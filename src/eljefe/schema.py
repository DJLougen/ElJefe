"""Canonical data schemas for ElJefe.

Every pipeline stage reads/writes these models as JSONL. Never discard raw
fields when deriving labels — downstream policies re-derive them differently.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Normalized task (output of 01_fetch + 02_normalize)
# ---------------------------------------------------------------------------

GraderName = Literal[
    "exact_match",
    "math_numeric",
    "multiple_choice",
    "code_tests",
    "json_schema",
    "ifeval",
    "tool_call",
    "reference_similarity",
    "llm_judge",
]


class Task(BaseModel):
    """A normalized, grader-addressable task row."""

    id: str
    source: str  # e.g. "gsm8k", "math500", "mmlu_pro", "mbpp", "ifeval"
    task_family: str  # e.g. "math", "reasoning", "code", "instruction_following"
    prompt: str
    system_prompt: Optional[str] = None
    reference: Optional[str] = None  # gold answer / canonical solution text
    grader: GraderName
    metadata: dict[str, Any] = Field(default_factory=dict)
    # metadata conventions:
    #   choices: list[str]            (multiple_choice)
    #   answer_index / answer_letter  (multiple_choice)
    #   test_code: str                (code_tests: asserts appended after candidate code)
    #   entry_point: str              (code_tests)
    #   schema: dict                  (json_schema)
    #   constraints: list[dict]       (ifeval: verifier kwargs)
    #   group_id: str                 (split-grouping key; defaults to id)
    #   privacy: "normal" | "local_only"

    def group_id(self) -> str:
        return str(self.metadata.get("group_id") or self.id)


# ---------------------------------------------------------------------------
# Generation (output of 03_generate_e4b / 04_generate_frontier)
# ---------------------------------------------------------------------------


class Generation(BaseModel):
    task_id: str
    model: str  # model id actually used
    model_revision: Optional[str] = None
    answer: str
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    latency_ms: Optional[float] = None
    cost_usd: Optional[float] = None
    finish_reason: Optional[str] = None
    gen_params: dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None  # set instead of answer when generation failed


# ---------------------------------------------------------------------------
# Scored row (output of 05/06 grading; input to router dataset)
# ---------------------------------------------------------------------------


class ScoredRow(BaseModel):
    """Canonical training row — plan §5. Raw scores are never discarded."""

    id: str
    source: str
    task_family: str
    prompt: str
    system_prompt: Optional[str] = None

    local_model: str
    frontier_model: str

    local_answer: Optional[str] = None
    frontier_answer: Optional[str] = None

    local_score: Optional[float] = None
    frontier_score: Optional[float] = None
    delta_q: Optional[float] = None  # frontier_score - local_score

    local_input_tokens: Optional[int] = None
    local_output_tokens: Optional[int] = None
    frontier_input_tokens: Optional[int] = None
    frontier_output_tokens: Optional[int] = None

    frontier_cost: Optional[float] = None
    local_latency_ms: Optional[float] = None
    frontier_latency_ms: Optional[float] = None

    grader_type: str = "deterministic"
    grader_confidence: float = 1.0

    metadata: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Router dataset row (output of 07_build_router_dataset)
# ---------------------------------------------------------------------------

SplitName = Literal["train", "validation", "test_iid", "test_ood", "mac_calibration"]


class RouterRow(BaseModel):
    id: str
    source: str
    task_family: str
    prompt: str
    system_prompt: Optional[str] = None
    group_id: str
    split: SplitName

    local_score: float
    frontier_score: float
    delta_q: float
    frontier_cost: Optional[float] = None
    local_input_tokens: Optional[int] = None
    local_output_tokens: Optional[int] = None
    frontier_input_tokens: Optional[int] = None
    frontier_output_tokens: Optional[int] = None
    grader_type: str = "deterministic"
    grader_confidence: float = 1.0
    metadata: dict[str, Any] = Field(default_factory=dict)

    # Derived labels are computed by build_router_dataset with configurable
    # thresholds and stored here so training is reproducible.
    local_sufficient: Optional[bool] = None
    frontier_helpful: Optional[bool] = None


# ---------------------------------------------------------------------------
# ElJefe-0 input / output
# ---------------------------------------------------------------------------


class RouterInput(BaseModel):
    """What ElJefe-0 sees before invoking the local model (plan §4)."""

    prompt: str
    system_prompt: Optional[str] = None
    input_tokens: Optional[int] = None
    modalities: list[str] = Field(default_factory=lambda: ["text"])
    tools_available: list[str] = Field(default_factory=list)
    local_model: str = "google/gemma-4-E4B-it"
    local_context_limit: int = 131072
    privacy: Literal["normal", "local_only"] = "normal"
    metadata: dict[str, Any] = Field(default_factory=dict)


class RouterOutput(BaseModel):
    p_local_sufficient: float
    expected_frontier_gain: float
    uncertainty: Optional[float] = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def stable_id(*parts: str, n: int = 16) -> str:
    """Deterministic short id from content parts."""
    h = hashlib.sha256("\x00".join(p or "" for p in parts).encode()).hexdigest()
    return h[:n]


def write_jsonl(path: str | "os.PathLike[str]", rows: list[BaseModel]) -> int:
    import os

    with open(path, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(r.model_dump_json() + "\n")
    return len(rows)


def read_jsonl(path: str | "os.PathLike[str]", model: type[BaseModel]) -> list[BaseModel]:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(model.model_validate(json.loads(line)))
    return out


def completed_ids(path: str | "os.PathLike[str]", key: str = "task_id") -> set[str]:
    """IDs already present in a JSONL file — for resume semantics."""
    import os

    ids: set[str] = set()
    if not os.path.exists(path):
        return ids
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    ids.add(json.loads(line)[key])
                except (json.JSONDecodeError, KeyError):
                    continue
    return ids
