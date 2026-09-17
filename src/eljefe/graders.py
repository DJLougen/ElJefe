"""Deterministic graders for ElJefe tasks (plan §8 grading hierarchy).

Every grader maps ``(Task, answer_text) -> GradeResult`` and never raises on
malformed input: garbage in, score 0.0 out with evidence in ``detail``.
``llm_judge`` is deliberately not implemented here — the open-ended grading
script (Phase B) owns judging; calling it raises ``NotImplementedError``.
"""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
import sys
import tempfile
from collections import Counter
from typing import Any, Callable, Optional

from pydantic import BaseModel, Field

from .schema import Task


class GradeResult(BaseModel):
    """Outcome of grading one answer against one task."""

    score: float  # in [0, 1]
    grader_type: str = "deterministic"  # "deterministic" | "llm_judge"
    confidence: float = 1.0
    detail: dict[str, Any] = Field(default_factory=dict)


def _ok(score: float, **detail: Any) -> GradeResult:
    return GradeResult(score=max(0.0, min(1.0, float(score))), detail=detail)


# ---------------------------------------------------------------------------
# Shared extraction helpers
# ---------------------------------------------------------------------------

_FENCE_RE = re.compile(r"```(?:[a-zA-Z0-9_-]+)?\s*\n(.*?)```", re.DOTALL)
_NUMBER_RE = r"[-+]?\$?\d[\d,]*(?:\.\d+)?(?:\s*/\s*\$?\d[\d,]*(?:\.\d+)?)?%?"


def _strip_fences(text: str) -> str:
    """Return the contents of the first fenced code block, else the raw text."""
    m = _FENCE_RE.search(text or "")
    return m.group(1).strip() if m else (text or "").strip()


def _norm_text(s: Optional[str]) -> str:
    return " ".join(str(s or "").split()).lower()


def _parse_number(s: Optional[str]) -> Optional[float]:
    """Parse '1,234.5', '$42', '75%', '3/4' -> float; None on failure."""
    if s is None:
        return None
    t = str(s).strip().replace("$", "").replace(",", "").replace("%", "").strip()
    if not t:
        return None
    try:
        if "/" in t:
            num, den = t.split("/", 1)
            den_f = float(den.strip())
            if den_f == 0:
                return None
            return float(num.strip()) / den_f
        return float(t)
    except (ValueError, ZeroDivisionError):
        return None


def _extract_final_number(text: str) -> Optional[float]:
    """Pull the final numeric answer out of model text.

    Order: '#### x' (GSM8K), '\\boxed{x}', 'the answer is x', last number.
    """
    text = text or ""
    for pat in (
        r"####\s*(" + _NUMBER_RE + r")",
        r"\\boxed\{([^{}]+)\}",
        r"answer\s+is\s*[:=]?\s*(" + _NUMBER_RE + r")",
    ):
        hits = re.findall(pat, text, re.IGNORECASE)
        if hits:
            val = _parse_number(hits[-1])
            if val is not None:
                return val
    hits = re.findall(_NUMBER_RE, text)
    return _parse_number(hits[-1]) if hits else None


def _extract_choice_letter(text: str) -> Optional[str]:
    """Extract a multiple-choice letter A-J from answer text."""
    text = (text or "").strip()
    if not text:
        return None
    for pat in (
        r"\\boxed\{\s*\(?\s*([A-J])\s*\)?\s*\}",
        r"answer\s*(?:is|:)?\s*\(?\s*([A-J])\b",
        r"\(\s*([A-J])\s*\)",
        r"\b([A-J])\s*[.\)]",
    ):
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return m.group(1).upper()
    m = re.fullmatch(r"\(?\s*([A-J])\s*\)?\s*\.?", text, re.IGNORECASE)
    return m.group(1).upper() if m else None


def _extract_json(text: str) -> Any:
    """Parse a JSON value from answer text (fences and prose tolerated)."""
    stripped = _strip_fences(text)
    try:
        return json.loads(stripped)
    except Exception:
        pass
    # Scan for the first decodable JSON object/array inside prose.
    decoder = json.JSONDecoder()
    for i, ch in enumerate(stripped):
        if ch in "{[":
            try:
                obj, _ = decoder.raw_decode(stripped[i:])
                return obj
            except Exception:
                continue
    raise ValueError("no JSON found")


def _token_f1(prediction: str, reference: str) -> float:
    """SQuAD-style token-level F1."""
    pred = re.findall(r"[a-z0-9]+", (prediction or "").lower())
    ref = re.findall(r"[a-z0-9]+", (reference or "").lower())
    if not pred and not ref:
        return 1.0
    if not pred or not ref:
        return 0.0
    common = Counter(pred) & Counter(ref)
    n_same = sum(common.values())
    if n_same == 0:
        return 0.0
    precision = n_same / len(pred)
    recall = n_same / len(ref)
    return 2 * precision * recall / (precision + recall)


# ---------------------------------------------------------------------------
# Graders
# ---------------------------------------------------------------------------


def grade_exact_match(task: Task, answer: str) -> GradeResult:
    pred = _norm_text(answer)
    ref = _norm_text(task.reference)
    match = bool(ref) and pred == ref
    return _ok(1.0 if match else 0.0, extracted=pred, expected=task.reference)


def grade_math_numeric(task: Task, answer: str) -> GradeResult:
    pred = _extract_final_number(answer)
    ref = _parse_number(task.reference)
    if ref is None:
        # Reference is not numeric — fall back to normalized string equality.
        match = _norm_text(answer) == _norm_text(task.reference)
        return _ok(1.0 if match else 0.0, extracted=pred, expected=task.reference,
                   note="non-numeric reference; string match")
    if pred is None:
        return _ok(0.0, extracted=None, expected=ref, error="no number found")
    match = math.isclose(pred, ref, rel_tol=1e-4, abs_tol=1e-9)
    return _ok(1.0 if match else 0.0, extracted=pred, expected=ref)


def grade_multiple_choice(task: Task, answer: str) -> GradeResult:
    pred = _extract_choice_letter(answer)
    md = task.metadata or {}
    expected: Optional[str] = None
    if md.get("answer_letter") is not None:
        expected = str(md["answer_letter"]).strip().upper()
    elif md.get("answer_index") is not None:
        try:
            expected = chr(ord("A") + int(md["answer_index"]))
        except (TypeError, ValueError):
            expected = None
    elif task.reference and re.fullmatch(r"\(?\s*[A-J]\s*\)?\.?", task.reference.strip(), re.IGNORECASE):
        expected = task.reference.strip().strip("(). ").upper()
    if expected is None:
        return _ok(0.0, extracted=pred, error="no expected letter in metadata")
    return _ok(1.0 if pred == expected else 0.0, extracted=pred, expected=expected)


def grade_code_tests(task: Task, answer: str) -> GradeResult:
    """Run candidate code + assert block in a subprocess; 1.0 iff exit 0."""
    code = _strip_fences(answer)
    test_code = str((task.metadata or {}).get("test_code") or "")
    if not code.strip():
        return _ok(0.0, error="no code extracted")
    program = code + "\n\n" + test_code + "\n"
    tmpdir = tempfile.mkdtemp(prefix="eljefe_grade_")
    path = os.path.join(tmpdir, "candidate.py")
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(program)
        proc = subprocess.run(
            [sys.executable or "python3", path],
            capture_output=True,
            text=True,
            timeout=15,
            cwd=tmpdir,
        )
        passed = proc.returncode == 0
        return _ok(
            1.0 if passed else 0.0,
            returncode=proc.returncode,
            stderr=(proc.stderr or "")[-2000:],
            stdout=(proc.stdout or "")[-2000:],
            timed_out=False,
        )
    except subprocess.TimeoutExpired:
        return _ok(0.0, error="timeout", timed_out=True)
    except Exception as exc:  # OSError etc. — never raise on garbage
        return _ok(0.0, error=f"{type(exc).__name__}: {exc}")
    finally:
        try:
            os.unlink(path)
            os.rmdir(tmpdir)
        except OSError:
            pass


def grade_json_schema(task: Task, answer: str) -> GradeResult:
    import jsonschema  # lazy: light dep, but keep module import cheap

    schema = (task.metadata or {}).get("schema")
    if not isinstance(schema, dict):
        return _ok(0.0, error="no schema in task.metadata")
    try:
        instance = _extract_json(answer)
    except Exception as exc:
        return _ok(0.0, error=f"json parse: {exc}")
    try:
        jsonschema.validate(instance=instance, schema=schema)
        return _ok(1.0, instance=instance)
    except Exception as exc:
        return _ok(0.0, error=f"schema: {exc}", instance=instance)


# ---------------------------------------------------------------------------
# IFEval verifier subset
# ---------------------------------------------------------------------------


def _sentences(text: str) -> list[str]:
    return [p for p in re.split(r"(?<=[.!?])\s+", (text or "").strip()) if p.strip()]


def _relation_ok(value: float, target: float, relation: Optional[str]) -> bool:
    rel = (relation or "").strip().lower()
    if rel in ("less than", "fewer than", "<"):
        return value < target
    if rel in ("at most", "no more than", "<="):
        return value <= target
    if rel in ("more than", ">"):
        return value > target
    if rel in ("equal to", "exactly", "=="):
        return value == target
    if rel in ("around", "approximately", "about"):
        return abs(value - target) <= max(1.0, 0.1 * abs(target))
    # "at least" and IFEval's default for missing relations.
    return value >= target


def _v_num_sentences(ans: str, kw: dict) -> bool:
    return _relation_ok(len(_sentences(ans)), float(kw.get("num_sentences", 0)), kw.get("relation"))


def _v_num_paragraphs(ans: str, kw: dict) -> bool:
    n = int(kw.get("num_paragraphs", 0))
    if "***" in ans:
        paras = [p for p in ans.split("***") if p.strip()]
    else:
        paras = [p for p in re.split(r"\n\s*\n", ans.strip()) if p.strip()]
    return len(paras) == n


def _v_num_words(ans: str, kw: dict) -> bool:
    return _relation_ok(len(ans.split()), float(kw.get("num_words", 0)), kw.get("relation"))


def _v_nth_paragraph_first_word(ans: str, kw: dict) -> bool:
    paras = [p for p in re.split(r"\n\s*\n", ans.strip()) if p.strip()]
    n, nth = int(kw.get("num_paragraphs", 0)), int(kw.get("nth_paragraph", 1))
    if len(paras) != n or not (1 <= nth <= len(paras)):
        return False
    words = paras[nth - 1].split()
    if not words:
        return False
    return words[0].strip(".,!?:;\"'").lower() == str(kw.get("first_word", "")).strip().lower()


def _v_kw_existence(ans: str, kw: dict) -> bool:
    low = ans.lower()
    return all(str(k).lower() in low for k in kw.get("keywords", []))


def _v_kw_forbidden(ans: str, kw: dict) -> bool:
    low = ans.lower()
    return not any(str(w).lower() in low for w in kw.get("forbidden_words", []))


def _v_kw_frequency(ans: str, kw: dict) -> bool:
    count = ans.lower().count(str(kw.get("keyword", "")).lower())
    return _relation_ok(count, float(kw.get("frequency", 0)), kw.get("relation"))


def _v_letter_frequency(ans: str, kw: dict) -> bool:
    letter = str(kw.get("letter", "")).lower()
    if not letter:
        return False
    count = ans.lower().count(letter)
    return _relation_ok(count, float(kw.get("let_frequency", 0)), kw.get("let_relation"))


def _v_json_format(ans: str, kw: dict) -> bool:
    try:
        json.loads(_strip_fences(ans))
        return True
    except Exception:
        return False


def _v_num_bullets(ans: str, kw: dict) -> bool:
    n = int(kw.get("num_bullets", 0))
    bullets = [ln for ln in ans.splitlines() if re.match(r"^\s*[*-]\s+", ln)]
    return len(bullets) == n


def _v_title(ans: str, kw: dict) -> bool:
    return bool(re.search(r"<<[^\n<>]+>>", ans))


def _v_constrained_response(ans: str, kw: dict) -> bool:
    low = ans.lower()
    return any(p in low for p in ("my answer is yes", "my answer is no", "my answer is maybe"))


def _v_num_highlights(ans: str, kw: dict) -> bool:
    n = int(kw.get("num_highlights", 0))
    return len(re.findall(r"\*\*?[^*\n]+\*\*?", ans)) >= n


def _v_multiple_sections(ans: str, kw: dict) -> bool:
    splitter = str(kw.get("section_spliter", "Section"))
    n = int(kw.get("num_sections", 0))
    return len(re.findall(re.escape(splitter) + r"\s*\d+", ans)) >= n


def _v_postscript(ans: str, kw: dict) -> bool:
    marker = str(kw.get("postscript_marker", "P.S."))
    return marker.lower() in ans.lower()


def _v_num_placeholders(ans: str, kw: dict) -> bool:
    n = int(kw.get("num_placeholders", 0))
    return len(re.findall(r"\[[^\[\]\n]*\]", ans)) >= n


def _v_end_checker(ans: str, kw: dict) -> bool:
    end = str(kw.get("end_phrase", "")).strip().lower()
    return bool(end) and ans.strip().lower().rstrip("\"'").endswith(end.rstrip("\"'"))


def _v_quotation(ans: str, kw: dict) -> bool:
    s = ans.strip()
    return len(s) >= 2 and s.startswith('"') and s.endswith('"')


def _v_capital_word_frequency(ans: str, kw: dict) -> bool:
    words = ans.split()
    if not words:
        return False
    caps = sum(1 for w in words if w.isupper() and any(c.isalpha() for c in w))
    ratio = caps / len(words)
    return _relation_ok(ratio, float(kw.get("capital_frequency", 0)), kw.get("capital_relation"))


def _v_english_capital(ans: str, kw: dict) -> bool:
    return ans.isupper()  # requires >=1 cased char, all cased upper


def _v_english_lowercase(ans: str, kw: dict) -> bool:
    return ans.islower()  # requires >=1 cased char, all cased lower


def _v_no_comma(ans: str, kw: dict) -> bool:
    return "," not in ans


def _v_two_responses(ans: str, kw: dict) -> bool:
    parts = [p.strip() for p in ans.split("******")]
    parts = [p for p in parts if p]
    return len(parts) == 2 and parts[0] != parts[1]


def _v_repeat_prompt(ans: str, kw: dict) -> bool:
    prompt = str(kw.get("prompt_to_repeat", "")).strip().lower()
    return bool(prompt) and ans.strip().lower().startswith(prompt)


_IFEVAL: dict[str, Callable[[str, dict], bool]] = {
    "length_constraints:num_sentences": _v_num_sentences,
    "length_constraints:num_paragraphs": _v_num_paragraphs,
    "length_constraints:number_words": _v_num_words,
    "length_constraints:nth_paragraph_first_word": _v_nth_paragraph_first_word,
    "keywords:existence": _v_kw_existence,
    "keywords:forbidden_words": _v_kw_forbidden,
    "keywords:frequency": _v_kw_frequency,
    "keywords:letter_frequency": _v_letter_frequency,
    "detectable_format:json_format": _v_json_format,
    "detectable_format:number_bullet_lists": _v_num_bullets,
    "detectable_format:title": _v_title,
    "detectable_format:constrained_response": _v_constrained_response,
    "detectable_format:number_highlighted_sections": _v_num_highlights,
    "detectable_format:multiple_sections": _v_multiple_sections,
    "detectable_content:postscript": _v_postscript,
    "detectable_content:number_placeholders": _v_num_placeholders,
    "startend:end_checker": _v_end_checker,
    "startend:quotation": _v_quotation,
    "change_case:capital_word_frequency": _v_capital_word_frequency,
    "change_case:english_capital": _v_english_capital,
    "change_case:english_lowercase": _v_english_lowercase,
    "punctuation:no_comma": _v_no_comma,
    "combination:two_responses": _v_two_responses,
    "combination:repeat_prompt": _v_repeat_prompt,
}


def grade_ifeval(task: Task, answer: str) -> GradeResult:
    """Score = fraction of verifiable constraints satisfied.

    Constraints come from ``task.metadata['constraints']`` as a list of
    ``{"instruction_id": ..., "kwargs": {...}}``. Unknown instruction ids are
    skipped (removed from the denominator) and recorded in ``detail.unknown``.
    """
    constraints = (task.metadata or {}).get("constraints") or []
    if not constraints:
        return _ok(1.0, note="no constraints", satisfied=0, checked=0, unknown=[])
    results: list[dict[str, Any]] = []
    unknown: list[str] = []
    satisfied = 0
    for c in constraints:
        iid = str((c or {}).get("instruction_id", ""))
        kwargs = (c or {}).get("kwargs") or {}
        fn = _IFEVAL.get(iid)
        if fn is None:
            unknown.append(iid)
            continue
        try:
            ok = bool(fn(answer or "", kwargs))
        except Exception:
            ok = False
        satisfied += int(ok)
        results.append({"instruction_id": iid, "ok": ok})
    checked = len(results)
    score = (satisfied / checked) if checked else 0.0
    return _ok(score, satisfied=satisfied, checked=checked,
               unknown=unknown, constraints=results)


def grade_tool_call(task: Task, answer: str) -> GradeResult:
    """Compare a JSON function call {name, arguments} to metadata.expected.

    score = 0.5 * name_match + 0.5 * fraction_of_expected_args_matched
    """
    expected = (task.metadata or {}).get("expected") or {}
    try:
        call = _extract_json(answer)
    except Exception as exc:
        return _ok(0.0, error=f"json parse: {exc}")
    if not isinstance(call, dict):
        return _ok(0.0, error="tool call is not a JSON object", parsed=call)

    def _val_eq(a: Any, b: Any) -> bool:
        fa, fb = _parse_number(str(a)), _parse_number(str(b))
        if fa is not None and fb is not None:
            return math.isclose(fa, fb, rel_tol=1e-6, abs_tol=1e-9)
        return str(a).strip().lower() == str(b).strip().lower()

    name_match = str(call.get("name", "")).strip() == str(expected.get("name", "")).strip()
    exp_args = expected.get("arguments") or {}
    got_args = call.get("arguments") or {}
    if not isinstance(exp_args, dict) or not isinstance(got_args, dict):
        args_frac = 1.0 if exp_args == got_args else 0.0
    elif not exp_args:
        args_frac = 1.0
    else:
        hits = sum(1 for k, v in exp_args.items() if k in got_args and _val_eq(got_args[k], v))
        args_frac = hits / len(exp_args)
    score = 0.5 * float(name_match) + 0.5 * args_frac
    return _ok(score, name_match=name_match, args_match_fraction=args_frac,
               parsed=call, expected=expected)


def grade_reference_similarity(task: Task, answer: str) -> GradeResult:
    f1 = _token_f1(answer or "", task.reference or "")
    return _ok(f1, token_f1=f1)


def grade_llm_judge(task: Task, answer: str) -> GradeResult:
    raise NotImplementedError(
        "llm_judge is Phase B only; the open-ended grading script judges "
        "pairwise and writes ScoredRow directly."
    )


# ---------------------------------------------------------------------------
# Registry + dispatch
# ---------------------------------------------------------------------------

GRADERS: dict[str, Callable[[Task, str], GradeResult]] = {
    "exact_match": grade_exact_match,
    "math_numeric": grade_math_numeric,
    "multiple_choice": grade_multiple_choice,
    "code_tests": grade_code_tests,
    "json_schema": grade_json_schema,
    "ifeval": grade_ifeval,
    "tool_call": grade_tool_call,
    "reference_similarity": grade_reference_similarity,
    "llm_judge": grade_llm_judge,
}


def grade(task: Task, answer: str) -> GradeResult:
    """Dispatch on ``task.grader``. Never raises on malformed answers."""
    fn = GRADERS.get(task.grader)
    if fn is None:
        return _ok(0.0, error=f"unknown grader {task.grader!r}")
    try:
        return fn(task, answer or "")
    except NotImplementedError:
        raise
    except Exception as exc:  # grader bug on weird input -> 0, not a crash
        return _ok(0.0, error=f"{type(exc).__name__}: {exc}")
