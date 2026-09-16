"""Dataset acquisition, normalization, and group-aware splitting for Jeff.

All HuggingFace imports are lazy (inside functions) so this module imports
with only light deps installed.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import random
import re
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from .schema import RouterRow, ScoredRow, Task, read_jsonl, stable_id


# ---------------------------------------------------------------------------
# Source fetchers — each yields Task rows from a raw HF dataset
# ---------------------------------------------------------------------------

def _iter_hf(dataset: str, config: str | None, split: str):
    from datasets import load_dataset  # lazy heavy import

    if config:
        return load_dataset(dataset, config, split=split)
    return load_dataset(dataset, split=split)


def _fetch_gsm8k(src_cfg: dict) -> Iterator[Task]:
    ds = _iter_hf("openai/gsm8k", "main", "train")
    for idx, row in enumerate(ds):
        answer = row["answer"]
        reference = answer.split("####")[-1].strip() if "####" in answer else answer.strip()
        yield Task(
            id=stable_id("gsm8k", str(idx)),
            source="gsm8k",
            task_family="math",
            prompt=row["question"].strip(),
            reference=reference,
            grader="math_numeric",
            metadata={"solution": answer, "row_index": idx},
        )


def _fetch_math500(src_cfg: dict) -> Iterator[Task]:
    ds = _iter_hf("HuggingFaceH4/MATH-500", None, "test")
    for idx, row in enumerate(ds):
        unique_id = str(row.get("unique_id") or idx)
        yield Task(
            id=stable_id("math500", unique_id),
            source="math500",
            task_family="math",
            prompt=row["problem"].strip(),
            reference=str(row.get("answer") or "").strip() or None,
            grader="math_numeric",
            metadata={
                "solution": row.get("solution"),
                "subject": row.get("subject"),
                "level": row.get("level"),
                "unique_id": unique_id,
                "group_id": f"math500:{unique_id}",
            },
        )


_MMLU_LETTERS = "ABCDEFGHIJ"


def _fetch_mmlu_pro(src_cfg: dict) -> Iterator[Task]:
    ds = _iter_hf("TIGER-Lab/MMLU-Pro", None, "test")
    for idx, row in enumerate(ds):
        options = list(row.get("options") or [])
        answer_index = row.get("answer_index")
        answer_letter = row.get("answer")
        if answer_letter is None and answer_index is not None:
            answer_letter = _MMLU_LETTERS[int(answer_index)]
        lines = [row["question"].strip(), ""]
        for i, opt in enumerate(options):
            lines.append(f"{_MMLU_LETTERS[i]}. {opt}")
        lines.append("")
        lines.append("Answer with the letter of the correct option only.")
        row_key = str(row.get("question_id") or idx)
        yield Task(
            id=stable_id("mmlu_pro", row_key),
            source="mmlu_pro",
            task_family="reasoning",
            prompt="\n".join(lines),
            reference=answer_letter,
            grader="multiple_choice",
            metadata={
                "choices": options,
                "answer_letter": answer_letter,
                "answer_index": answer_index,
                "category": row.get("category"),
                "row_index": idx,
            },
        )


_ENTRY_STOP = {
    "assert", "math", "np", "numpy", "isclose", "allclose", "approx", "pytest",
    "raises", "set", "list", "tuple", "dict", "sorted", "len", "abs", "round",
    "str", "int", "float", "bool", "type", "isinstance", "print", "sum", "min",
    "max", "any", "all", "map", "filter", "zip", "enumerate", "range",
    "frozenset", "re", "eq", "ne", "deepcopy", "Counter", "TestCase",
}


def _parse_entry_point(test_code: str) -> str | None:
    """First non-builtin call name inside the first assert line."""
    for line in test_code.splitlines():
        line = line.strip()
        if not line.startswith("assert"):
            continue
        for m in re.finditer(r"([A-Za-z_]\w*)\s*\(", line):
            name = m.group(1)
            if name not in _ENTRY_STOP:
                return name
    return None


def _fetch_mbpp(src_cfg: dict) -> Iterator[Task]:
    from datasets import load_dataset  # lazy heavy import

    ds = load_dataset("google-research-datasets/mbpp", "full")
    for split in ("train", "test", "validation", "prompt"):
        if split not in ds:
            continue
        for row in ds[split]:
            task_id = str(row.get("task_id"))
            test_list = list(row.get("test_list") or [])
            test_code = "\n".join(test_list)
            yield Task(
                id=stable_id("mbpp", task_id),
                source="mbpp",
                task_family="code",
                prompt=row["text"].strip() + "\nWrite a Python function.",
                reference=row.get("code"),
                grader="code_tests",
                metadata={
                    "test_code": test_code,
                    "test_setup_code": row.get("test_setup_code") or "",
                    "entry_point": _parse_entry_point(test_code),
                    "mbpp_task_id": task_id,
                    "mbpp_split": split,
                },
            )


def _fetch_ifeval(src_cfg: dict) -> Iterator[Task]:
    ds = _iter_hf("google/IFEval", None, "train")
    for idx, row in enumerate(ds):
        instruction_ids = list(row.get("instruction_id_list") or [])
        kwargs_list = list(row.get("kwargs") or [])
        constraints = [
            {"instruction_id": iid, "kwargs": kw or {}}
            for iid, kw in zip(instruction_ids, kwargs_list)
        ]
        row_key = str(row.get("key") or idx)
        yield Task(
            id=stable_id("ifeval", row_key),
            source="ifeval",
            task_family="instruction_following",
            prompt=row["prompt"].strip(),
            reference=None,
            grader="ifeval",
            metadata={"constraints": constraints, "row_index": idx},
        )


_FETCHERS = {
    "gsm8k": _fetch_gsm8k,
    "math500": _fetch_math500,
    "mmlu_pro": _fetch_mmlu_pro,
    "mbpp": _fetch_mbpp,
    "ifeval": _fetch_ifeval,
}


def fetch_source(name: str, cfg: dict) -> Iterable[Task]:
    """Yield normalized Task rows for one source.

    `cfg` may be the full data config (with a ``sources`` mapping) or the
    per-source config dict directly. Honors the per-source ``limit``.
    """
    if name not in _FETCHERS:
        raise KeyError(f"unknown source {name!r}; known: {sorted(_FETCHERS)}")
    if isinstance(cfg, dict) and "sources" in cfg:
        src_cfg = dict(cfg.get("sources", {}).get(name) or {})
    else:
        src_cfg = dict(cfg or {})
    limit = src_cfg.get("limit")
    it = _FETCHERS[name](src_cfg)
    if limit is not None:
        it = itertools.islice(it, int(limit))
    return it


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def _norm_prompt(prompt: str) -> str:
    return re.sub(r"\s+", " ", prompt.strip().lower())


def normalize(tasks: Iterable[Task]) -> list[Task]:
    """Dedupe by normalized-prompt sha; ensure every task has a group_id."""
    seen: set[str] = set()
    out: list[Task] = []
    for task in tasks:
        digest = hashlib.sha256(_norm_prompt(task.prompt).encode()).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        task.metadata.setdefault("group_id", task.id)
        out.append(task)
    return out


def load_tasks(path: str | Path) -> list[Task]:
    return list(read_jsonl(path, Task))


# ---------------------------------------------------------------------------
# Near-duplicate grouping (bottom-k minhash-lite + exact Jaccard verify)
# ---------------------------------------------------------------------------

def _word_ngrams(text: str, n: int = 3) -> set[tuple[str, ...]]:
    toks = re.findall(r"\w+", text.lower())
    if not toks:
        return set()
    if len(toks) < n:
        return {tuple(toks)}
    return {tuple(toks[i : i + n]) for i in range(len(toks) - n + 1)}


def _bottomk(shingles: set[tuple[str, ...]], k: int = 16) -> list[int]:
    hashes = sorted(
        int.from_bytes(
            hashlib.blake2b(repr(s).encode(), digest_size=8).digest(), "little"
        )
        for s in shingles
    )
    return hashes[:k]


class _UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def _near_duplicate_groups(
    prompts: list[str], threshold: float
) -> dict[int, list[int]]:
    """Union row indices whose word-3gram Jaccard >= threshold.

    Bottom-k sketch buckets give O(n) candidate generation; candidate pairs
    are verified with the exact Jaccard before unioning.
    """
    n = len(prompts)
    uf = _UnionFind(n)
    shingles = [_word_ngrams(p) for p in prompts]
    buckets: dict[int, list[int]] = {}
    for i, sh in enumerate(shingles):
        for h in _bottomk(sh):
            buckets.setdefault(h, []).append(i)
    for members in buckets.values():
        if len(members) < 2:
            continue
        for a_pos in range(len(members)):
            i = members[a_pos]
            for j in members[a_pos + 1 :]:
                if uf.find(i) == uf.find(j):
                    continue
                si, sj = shingles[i], shingles[j]
                if not si or not sj:
                    continue
                inter = len(si & sj)
                union = len(si) + len(sj) - inter
                if union and inter / union >= threshold:
                    uf.union(i, j)
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(uf.find(i), []).append(i)
    return groups


# ---------------------------------------------------------------------------
# Group-aware split into RouterRow
# ---------------------------------------------------------------------------

def split_rows(rows: list[ScoredRow], cfg: dict) -> list[RouterRow]:
    """Assign splits and derive labels.

    - Rows missing scores are dropped (they belong in quarantine).
    - Near-duplicate prompts (word-3gram Jaccard >= threshold) are unioned
      into the same split group, on top of each row's metadata group_id.
    - Groups touching ``splits.ood_sources`` go to test_ood.
    - Remaining groups are shuffled (seeded) and greedily assigned to
      validation / test_iid / train by row-count fractions.
    - ``mac_calibration_n`` rows are stratified-sampled (by source) out of
      test_iid and relabeled ``mac_calibration``.
    """
    cfg = cfg or {}
    split_cfg = cfg.get("splits", {}) or {}
    nd_cfg = cfg.get("near_duplicate", {}) or {}
    quality_floor = float(cfg.get("quality_floor", 0.70))
    epsilon = float(cfg.get("epsilon", 0.10))
    seed = int(cfg.get("seed", 42))
    val_frac = float(split_cfg.get("validation_frac", 0.10))
    test_frac = float(split_cfg.get("test_iid_frac", 0.10))
    ood_sources = set(split_cfg.get("ood_sources") or [])
    mac_n = int(split_cfg.get("mac_calibration_n") or 0)
    nd_enabled = bool(nd_cfg.get("enabled", True))
    nd_threshold = float(nd_cfg.get("threshold", 0.80))

    scored = [
        r
        for r in rows
        if r.local_score is not None
        and r.frontier_score is not None
        and r.delta_q is not None
    ]

    # Level 1 grouping: declared group_id (or row id).
    group_of: dict[str, list[int]] = {}
    for i, r in enumerate(scored):
        key = str(r.metadata.get("group_id") or r.id)
        group_of.setdefault(key, []).append(i)
    # rep index -> group key
    group_keys = list(group_of)
    row_group = [0] * len(scored)
    for gi, key in enumerate(group_keys):
        for i in group_of[key]:
            row_group[i] = gi

    # Level 2 grouping: near-duplicate union across declared groups.
    if nd_enabled and scored:
        nd = _near_duplicate_groups([r.prompt for r in scored], nd_threshold)
        # map row-level unions to group-level unions
        guf = _UnionFind(len(group_keys))
        for members in nd.values():
            first = row_group[members[0]]
            for i in members[1:]:
                guf.union(first, row_group[i])
        merged: dict[int, list[int]] = {}
        for gi in range(len(group_keys)):
            merged.setdefault(guf.find(gi), []).append(gi)
        super_groups = list(merged.values())
    else:
        super_groups = [[gi] for gi in range(len(group_keys))]

    def group_rows(gi_list: list[int]) -> list[int]:
        out: list[int] = []
        for gi in gi_list:
            out.extend(group_of[group_keys[gi]])
        return out

    split_of_row: dict[int, str] = {}
    ood_groups, iid_groups = [], []
    for sg in super_groups:
        members = group_rows(sg)
        if any(scored[i].source in ood_sources for i in members):
            ood_groups.append(sg)
        else:
            iid_groups.append(sg)
    for sg in ood_groups:
        for i in group_rows(sg):
            split_of_row[i] = "test_ood"

    rng = random.Random(seed)
    rng.shuffle(iid_groups)
    n_iid = sum(len(group_rows(sg)) for sg in iid_groups)
    val_target = val_frac * n_iid
    test_target = test_frac * n_iid
    val_count = test_count = 0
    for sg in iid_groups:
        members = group_rows(sg)
        if val_count < val_target:
            dest, val_count = "validation", val_count + len(members)
        elif test_count < test_target:
            dest, test_count = "test_iid", test_count + len(members)
        else:
            dest = "train"
        for i in members:
            split_of_row[i] = dest

    # Mac calibration: stratified (by source) sample out of test_iid.
    if mac_n > 0:
        test_rows = [i for i, s in split_of_row.items() if s == "test_iid"]
        by_source: dict[str, list[int]] = {}
        for i in test_rows:
            by_source.setdefault(scored[i].source, []).append(i)
        total = len(test_rows)
        picked: list[int] = []
        # proportional allocation with largest-remainder rounding
        quotas = {
            src: (len(idxs) / total) * mac_n for src, idxs in by_source.items()
        } if total else {}
        alloc = {src: int(q) for src, q in quotas.items()}
        remainder = mac_n - sum(alloc.values())
        for src in sorted(quotas, key=lambda s: quotas[s] - alloc[s], reverse=True):
            if remainder <= 0:
                break
            alloc[src] += 1
            remainder -= 1
        for src, k in alloc.items():
            idxs = by_source[src]
            rng.shuffle(idxs)
            picked.extend(idxs[:k])
        for i in picked:
            split_of_row[i] = "mac_calibration"

    out: list[RouterRow] = []
    for i, r in enumerate(scored):
        local_sufficient = bool(
            r.local_score >= quality_floor and r.delta_q <= epsilon
        )
        frontier_helpful = bool(r.delta_q > epsilon)
        metadata = dict(r.metadata)
        metadata["quality_floor"] = quality_floor
        metadata["epsilon"] = epsilon
        out.append(
            RouterRow(
                id=r.id,
                source=r.source,
                task_family=r.task_family,
                prompt=r.prompt,
                system_prompt=r.system_prompt,
                group_id=str(r.metadata.get("group_id") or r.id),
                split=split_of_row[i],
                local_score=r.local_score,
                frontier_score=r.frontier_score,
                delta_q=r.delta_q,
                frontier_cost=r.frontier_cost,
                local_input_tokens=r.local_input_tokens,
                local_output_tokens=r.local_output_tokens,
                frontier_input_tokens=r.frontier_input_tokens,
                frontier_output_tokens=r.frontier_output_tokens,
                grader_type=r.grader_type,
                grader_confidence=r.grader_confidence,
                metadata=metadata,
                local_sufficient=local_sufficient,
                frontier_helpful=frontier_helpful,
            )
        )
    return out


# ---------------------------------------------------------------------------
# Manifest helper (plan §32)
# ---------------------------------------------------------------------------

def update_manifest(root: str | Path, stage: str, entry: dict[str, Any]) -> None:
    """Upsert one stage entry into artifacts/reports/manifest.json."""
    import datetime

    path = Path(root) / "artifacts" / "reports" / "manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    data: dict[str, Any] = {}
    if path.exists():
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError:
            data = {}
    data[stage] = {
        "stage": stage,
        "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        **entry,
    }
    path.write_text(json.dumps(data, indent=2) + "\n")
