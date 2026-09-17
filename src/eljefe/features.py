"""Cheap pre-inference prompt features for ElJefe-0 (plan §4).

Pure stdlib — no model downloads, no heavy deps. Flags are emitted as 0/1
ints so the whole dict feeds straight into a tabular model.
"""

from __future__ import annotations

import re
from typing import Any, Optional

_URL_RE = re.compile(r"https?://|www\.", re.IGNORECASE)
_IMAGE_RE = re.compile(
    r"!\[|\b(image|picture|photo|screenshot|diagram|figure)\b|\.(png|jpe?g|gif|webp|svg)\b",
    re.IGNORECASE,
)
_WH_WORDS = {"what", "why", "how", "when", "where", "which", "who", "whom", "whose"}
_CODE_HINT_RE = re.compile(
    r"\b(function|code|script|program|debug|compile|class|def |import |error|bug|python|"
    r"javascript|typescript|sql|regex|api)\b",
    re.IGNORECASE,
)
_MATH_HINT_RE = re.compile(
    r"\b(solve|equation|calculate|compute|math|arithmetic|algebra|integral|derivative|"
    r"probability|prove|theorem|sum|product|ratio|percent)\b|\d+\s*[-+*/^=]\s*\d+",
    re.IGNORECASE,
)
_REWRITE_HINT_RE = re.compile(
    r"\b(rewrite|rephrase|paraphrase|proofread|edit this|improve this|make this|"
    r"polish|shorten|expand)\b",
    re.IGNORECASE,
)
_SUMMARIZE_HINT_RE = re.compile(
    r"\b(summari[sz]e|summary|tl;?dr|condense|key points|main idea|gist|abstract)\b",
    re.IGNORECASE,
)


def _task_family_guess(text: str, code_blocks: int) -> str:
    if code_blocks or _CODE_HINT_RE.search(text):
        return "code"
    if _MATH_HINT_RE.search(text):
        return "math"
    if _REWRITE_HINT_RE.search(text):
        return "rewrite"
    if _SUMMARIZE_HINT_RE.search(text):
        return "summarize"
    return "other"


def extract_features(
    prompt: str,
    system_prompt: Optional[str] = None,
    metadata: Optional[dict] = None,
) -> dict[str, Any]:
    """Derived features available before invoking the local model."""
    prompt = prompt or ""
    md = metadata or {}
    combined = prompt + "\n" + (system_prompt or "")

    char_count = len(prompt)
    word_count = len(prompt.split())
    line_count = prompt.count("\n") + (1 if prompt else 0)
    code_block_count = len(re.findall(r"```", combined)) // 2
    first_word = (prompt.strip().split() or [""])[0].lower().strip(".,!?:;\"'")

    requested_code = bool(_CODE_HINT_RE.search(combined)) or code_block_count > 0

    return {
        "char_count": char_count,
        "word_count": word_count,
        "approx_token_count": char_count / 4.0,
        "line_count": line_count,
        "code_block_count": code_block_count,
        "has_url": int(bool(_URL_RE.search(combined))),
        "has_image_ref": int(bool(_IMAGE_RE.search(combined))),
        "question_mark": int("?" in prompt),
        "starts_with_wh_word": int(first_word in _WH_WORDS),
        "requested_json": int("json" in combined.lower()),
        "requested_code": int(requested_code),
        "num_turns": int(md.get("num_turns", 1)),
        "context_len": int(
            md.get("context_len") or md.get("input_tokens") or round(char_count / 4.0)
        ),
        "task_family_guess": _task_family_guess(combined, code_block_count),
    }
