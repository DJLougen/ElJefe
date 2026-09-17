"""Router models for ElJefe.

A `Router` is a uniform wrapper over every router kind in the experiment
matrix (plan §31): constant endpoints, the heuristic baseline, TF-IDF +
logistic regression, frozen-embedding + LightGBM, and the fine-tuned
dual-head MiniLM encoder (ElJefe v0).

Heavy dependencies (torch, transformers, sentence-transformers, lightgbm)
are imported lazily inside the functions that need them so this module
imports with only numpy + pydantic installed.
"""

from __future__ import annotations

import json
import math
import os
import random
import subprocess
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np

from eljefe.schema import RouterRow, stable_id

REPO_ROOT = Path(__file__).resolve().parents[2]

# Cache for lazy-loaded sentence embedders / torch modules.
_EMBEDDER_CACHE: dict[str, Any] = {}


# ---------------------------------------------------------------------------
# Small shared helpers
# ---------------------------------------------------------------------------


def seed_everything(seed: int = 42) -> None:
    """Seed python/numpy (and torch when available) for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def git_commit() -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def load_router_dataset(path: str | os.PathLike[str]) -> list[RouterRow]:
    """Load data/router/router_dataset.parquet into RouterRow models."""
    import pandas as pd

    df = pd.read_parquet(path)
    records = df.to_dict(orient="records")
    # Parquet serializes dict columns as JSON strings — decode them back.
    import json as _json
    for rec in records:
        for k in ("metadata",):
            if isinstance(rec.get(k), str):
                try:
                    rec[k] = _json.loads(rec[k])
                except Exception:
                    rec[k] = {}
    return [RouterRow.model_validate(rec) for rec in records]


def update_manifest(name: str, entry: dict[str, Any],
                    path: str | os.PathLike[str] | None = None) -> Path:
    """Merge `entry` into artifacts/reports/manifest.json under key `name`.

    Delegates to eljefe.datasets.update_manifest (the repo convention) and
    stamps git_commit into the entry. `path` is accepted for backward
    compatibility but the manifest always lives at
    <root>/artifacts/reports/manifest.json.
    """
    from eljefe.datasets import update_manifest as _update_manifest

    entry = dict(entry)
    entry.setdefault("git_commit", git_commit())
    _update_manifest(REPO_ROOT, name, entry)
    return REPO_ROOT / "artifacts" / "reports" / "manifest.json"


def prompt_key(prompt: str, system_prompt: Optional[str] = None) -> str:
    """Stable cache key for one prompt (used by the embedding cache)."""
    return stable_id(prompt, system_prompt or "")


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------


def _fallback_features(prompt: str, system_prompt: Optional[str] = None,
                       task_family: Optional[str] = None) -> dict[str, Any]:
    """Minimal feature dict used only when eljefe.features is unavailable."""
    import re

    text = prompt or ""
    words = text.split()
    lower = text.lower()
    n_code = len(re.findall(r"```", text)) // 2
    if task_family:
        guess = task_family
    elif n_code or any(k in lower for k in ("def ", "function", "code", "compile", "bug")):
        guess = "code"
    elif any(k in lower for k in ("solve", "calculate", "equation", "math", "how many", "sum", "integral")):
        guess = "math"
    elif any(k in lower for k in ("json", "format", "list exactly", "do not", "constraint")):
        guess = "instruction_following"
    else:
        guess = "general"
    return {
        "n_chars": len(text),
        "n_words": len(words),
        "n_lines": text.count("\n") + 1,
        "n_code_blocks": n_code,
        "has_url": float("http://" in lower or "https://" in lower),
        "has_question": float("?" in text),
        "task_family_guess": guess,
    }


def extract_row_features(row: RouterRow) -> dict[str, Any]:
    """Feature dict for one RouterRow via eljefe.features (lazy, with fallback)."""
    try:
        from eljefe.features import extract_features

        feats = dict(extract_features(row.prompt, row.system_prompt, row.metadata))
    except Exception:
        feats = _fallback_features(row.prompt, row.system_prompt)
    feats.setdefault("task_family_guess", row.task_family)
    return feats


class FeatureVectorizer:
    """dict-of-features -> fixed-width float32 matrix.

    Numeric values become columns; string values become one-hot columns over
    the categories seen at fit time (capped at `max_categories`). The fitted
    schema is stored so transform() is deterministic at inference time.
    """

    def __init__(self, max_categories: int = 32):
        self.max_categories = max_categories
        self.numeric_keys: list[str] = []
        self.categorical_maps: dict[str, dict[str, int]] = {}
        self.dim_: int = 0

    def fit(self, feats: list[dict[str, Any]]) -> "FeatureVectorizer":
        numeric: set[str] = set()
        cats: dict[str, set[str]] = {}
        for f in feats:
            for k, v in f.items():
                if isinstance(v, bool) or isinstance(v, (int, float)):
                    numeric.add(k)
                elif isinstance(v, str):
                    cats.setdefault(k, set()).add(v)
        self.numeric_keys = sorted(numeric)
        self.categorical_maps = {}
        for k, values in sorted(cats.items()):
            top = sorted(values)[: self.max_categories]
            self.categorical_maps[k] = {v: i for i, v in enumerate(top)}
        self.dim_ = len(self.numeric_keys) + sum(
            len(m) for m in self.categorical_maps.values()
        )
        return self

    def transform(self, feats: list[dict[str, Any]]) -> np.ndarray:
        n = len(feats)
        X = np.zeros((n, self.dim_), dtype=np.float32)
        for i, f in enumerate(feats):
            col = 0
            for k in self.numeric_keys:
                v = f.get(k, 0.0)
                try:
                    X[i, col] = float(v)
                except (TypeError, ValueError):
                    X[i, col] = 0.0
                col += 1
            for k, mapping in self.categorical_maps.items():
                v = f.get(k)
                if isinstance(v, str) and v in mapping:
                    X[i, col + mapping[v]] = 1.0
                col += len(mapping)
        return X

    def fit_transform(self, feats: list[dict[str, Any]]) -> np.ndarray:
        return self.fit(feats).transform(feats)


# ---------------------------------------------------------------------------
# Heuristic baseline (plan §9 baseline 1)
# ---------------------------------------------------------------------------


def heuristic_p_local(features: dict[str, Any], cfg: Optional[dict] = None) -> float:
    """Rule-based P(local_sufficient) from cheap features.

    Starts at 0.95 and subtracts penalties for long prompts, hard task
    families, and code blocks, per configs/router_baseline.yaml `heuristic`.
    """
    cfg = cfg or {}
    p = 0.95
    long_tokens = float(cfg.get("long_prompt_tokens", 4000))
    hard = {str(h).lower() for h in cfg.get("hard_families", ["math", "code"])}

    n_tok: Optional[float] = None
    for key in ("approx_token_count", "context_len", "input_tokens",
                "word_count", "char_count"):
        if key in features:
            try:
                n_tok = float(features[key])
            except (TypeError, ValueError):
                continue
            break
    if n_tok is not None and n_tok > long_tokens:
        p -= 0.50

    fam = str(
        features.get("task_family_guess") or features.get("task_family") or ""
    ).lower()
    if fam and any(h in fam for h in hard):
        p -= 0.40

    try:
        n_code = float(
            features.get("code_block_count", features.get("n_code_blocks", 0)) or 0
        )
        if n_code > 0:
            p -= 0.15
    except (TypeError, ValueError):
        pass

    return float(min(0.99, max(0.01, p)))


# ---------------------------------------------------------------------------
# Embedding helper (lazy sentence-transformers)
# ---------------------------------------------------------------------------


def embed_prompts(prompts: list[str], model_name: str,
                  batch_size: int = 64, device: Optional[str] = None) -> np.ndarray:
    """Encode prompts with a sentence-transformers model (lazy import)."""
    from sentence_transformers import SentenceTransformer

    device = device or os.environ.get("ELJEFE_EMBED_DEVICE")
    if model_name not in _EMBEDDER_CACHE:
        _EMBEDDER_CACHE[model_name] = SentenceTransformer(model_name, device=device)
    model = _EMBEDDER_CACHE[model_name]
    emb = model.encode(
        list(prompts),
        batch_size=batch_size,
        show_progress_bar=len(prompts) > batch_size,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )
    return np.asarray(emb, dtype=np.float32)


def load_embedding_cache(cache_path: str | os.PathLike[str]):
    """Return ({prompt_key: row_index}, matrix) for a 09_embed_prompts cache.

    Returns ({}, None) when the cache or its .ids.json sidecar is absent.
    """
    cache_path = Path(cache_path)
    ids = cache_path.with_suffix(cache_path.suffix + ".ids.json")
    if not (cache_path.exists() and ids.exists()):
        return {}, None
    meta = json.loads(ids.read_text())
    keys = list(meta.get("keys", []))
    emb = np.load(cache_path)
    return {k: i for i, k in enumerate(keys)}, emb


def predict_proba_rows(router: "Router", rows: list[RouterRow],
                       cache_path: str | os.PathLike[str] | None = None
                       ) -> np.ndarray:
    """predict_proba over RouterRows, using the embedding cache when possible.

    For kind="embedding" this avoids re-running the encoder when
    cache_path covers every row's prompt key; otherwise falls back to
    router.predict_proba (which embeds on demand).
    """
    prompts = [r.prompt for r in rows]
    feats = [extract_row_features(r) for r in rows]
    if router.kind == "embedding" and cache_path is not None:
        index, emb = load_embedding_cache(cache_path)
        if emb is not None:
            keys = [prompt_key(r.prompt, r.system_prompt) for r in rows]
            if all(k in index for k in keys):
                E = emb[[index[k] for k in keys]]
                if router.use_features and router.featurizer is not None:
                    X = np.hstack([E, router.featurizer.transform(feats)])
                else:
                    X = E
                return router.predict_proba_matrix(X)
    if router.kind in ("heuristic", "embedding"):
        return router.predict_proba(prompts, feats)
    return router.predict_proba(prompts)


# ---------------------------------------------------------------------------
# Router wrapper
# ---------------------------------------------------------------------------


class Router:
    """Uniform wrapper over all router kinds (CONTRACTS.md `eljefe.router`).

    kind:
      - "constant"  : payload {"p": float}
      - "heuristic" : payload {"cfg": dict}
      - "tfidf"     : payload {"vec", "clf", "reg"}
      - "embedding" : payload {"clf", "reg"} + featurizer + embedder_name
      - "minilm"    : payload {"state_dict", "encoder_name", "max_length"}
    """

    def __init__(
        self,
        kind: str,
        payload: Optional[dict[str, Any]] = None,
        featurizer: Optional[FeatureVectorizer] = None,
        embedder_name: Optional[str] = None,
        use_features: bool = False,
        meta: Optional[dict[str, Any]] = None,
    ):
        self.kind = kind
        self.payload = payload or {}
        self.featurizer = featurizer
        self.embedder_name = embedder_name
        self.use_features = use_features
        self.meta = meta or {}
        self._module = None  # lazy torch module for kind == "minilm"
        self._tokenizer = None

    # -- feature matrix -----------------------------------------------------

    def _feature_matrix(self, prompts: list[str],
                        features: Optional[list[dict]] = None) -> np.ndarray:
        """Embedding (+ optional metadata features) matrix for prompts."""
        emb = embed_prompts(
            list(prompts),
            self.embedder_name or "sentence-transformers/all-MiniLM-L6-v2",
        )
        if self.use_features and self.featurizer is not None:
            feats = features or [extract_row_features(_prompt_row(p)) for p in prompts]
            return np.hstack([emb, self.featurizer.transform(list(feats))])
        return emb

    # -- prediction ---------------------------------------------------------

    def predict_proba_matrix(self, X: np.ndarray) -> np.ndarray:
        """P(local_sufficient) from a precomputed feature matrix."""
        if self.kind == "tfidf":
            return np.asarray(self.payload["clf"].predict_proba(X)[:, 1])
        if self.kind == "embedding":
            clf = self.payload["clf"]
            if hasattr(clf, "predict_proba"):
                return np.asarray(clf.predict_proba(X)[:, 1])
            return np.asarray(clf.predict(X), dtype=np.float64)
        raise TypeError(f"predict_proba_matrix not supported for kind={self.kind}")

    def predict_gain_matrix(self, X: np.ndarray) -> np.ndarray:
        """E[delta_q] from a precomputed feature matrix."""
        reg = self.payload.get("reg")
        if reg is None:
            return np.zeros(X.shape[0], dtype=np.float64)
        return np.asarray(reg.predict(X), dtype=np.float64)

    def predict_proba(self, prompts: list[str],
                      features: Optional[list[dict]] = None) -> np.ndarray:
        prompts = list(prompts)
        if self.kind == "constant":
            return np.full(len(prompts), float(self.payload.get("p", 0.5)))
        if self.kind == "heuristic":
            cfg = self.payload.get("cfg", {})
            feats = features or [
                extract_row_features(_prompt_row(p)) for p in prompts
            ]
            return np.array([heuristic_p_local(f, cfg) for f in feats])
        if self.kind == "tfidf":
            X = self.payload["vec"].transform(prompts)
            return self.predict_proba_matrix(X)
        if self.kind == "embedding":
            return self.predict_proba_matrix(self._feature_matrix(prompts, features))
        if self.kind == "minilm":
            return self._minilm_forward(prompts)[0]
        raise ValueError(f"unknown router kind: {self.kind}")

    def predict_gain(self, prompts: list[str],
                     features: Optional[list[dict]] = None) -> np.ndarray:
        prompts = list(prompts)
        if self.kind in ("constant", "heuristic"):
            return np.zeros(len(prompts))
        if self.kind == "tfidf":
            X = self.payload["vec"].transform(prompts)
            return self.predict_gain_matrix(X)
        if self.kind == "embedding":
            return self.predict_gain_matrix(self._feature_matrix(prompts, features))
        if self.kind == "minilm":
            return self._minilm_forward(prompts)[1]
        raise ValueError(f"unknown router kind: {self.kind}")

    # -- minilm -------------------------------------------------------------

    def _minilm_forward(self, prompts: list[str]) -> tuple[np.ndarray, np.ndarray]:
        import torch

        module, tokenizer = self._ensure_minilm()
        max_length = int(self.payload.get("max_length", 512))
        device = next(module.parameters()).device
        probs: list[np.ndarray] = []
        gains: list[np.ndarray] = []
        module.eval()
        with torch.no_grad():
            for i in range(0, len(prompts), 64):
                batch = tokenizer(
                    prompts[i : i + 64],
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                ).to(device)
                cls_logit, reg = module(**batch)
                probs.append(torch.sigmoid(cls_logit).cpu().numpy())
                gains.append(reg.cpu().numpy())
        p = np.concatenate(probs) if probs else np.zeros(0)
        g = np.concatenate(gains) if gains else np.zeros(0)
        return p.astype(np.float64), g.astype(np.float64)

    def _ensure_minilm(self):
        if self._module is None:
            import torch
            from transformers import AutoTokenizer

            encoder_name = self.payload.get(
                "encoder_name", "sentence-transformers/all-MiniLM-L6-v2"
            )
            self._module = build_minilm_module(encoder_name)
            state = self.payload.get("state_dict")
            if state is not None:
                self._module.load_state_dict(state)
            self._module.eval()
            tok_dir = self.meta.get("tokenizer_dir")
            if tok_dir and Path(tok_dir).exists():
                self._tokenizer = AutoTokenizer.from_pretrained(tok_dir)
            else:
                self._tokenizer = AutoTokenizer.from_pretrained(encoder_name)
        return self._module, self._tokenizer

    # -- persistence --------------------------------------------------------

    def save(self, dir: str | os.PathLike[str]) -> Path:
        import joblib

        d = Path(dir)
        d.mkdir(parents=True, exist_ok=True)
        payload = dict(self.payload)
        if self.kind == "minilm":
            import torch

            state = payload.pop("state_dict", None)
            if state is None and self._module is not None:
                state = {k: v.cpu() for k, v in self._module.state_dict().items()}
            if state is not None:
                torch.save(state, d / "model.pt")
            if self._tokenizer is not None:
                self._tokenizer.save_pretrained(d / "tokenizer")
                self.meta["tokenizer_dir"] = str(d / "tokenizer")
        joblib.dump(
            {
                "kind": self.kind,
                "payload": payload,
                "featurizer": self.featurizer,
                "embedder_name": self.embedder_name,
                "use_features": self.use_features,
                "meta": self.meta,
            },
            d / "router.pkl",
        )
        return d

    @classmethod
    def load(cls, dir: str | os.PathLike[str]) -> "Router":
        import joblib

        d = Path(dir)
        blob = joblib.load(d / "router.pkl")
        router = cls(
            kind=blob["kind"],
            payload=blob.get("payload") or {},
            featurizer=blob.get("featurizer"),
            embedder_name=blob.get("embedder_name"),
            use_features=bool(blob.get("use_features")),
            meta=blob.get("meta") or {},
        )
        if router.kind == "minilm":
            import torch

            model_path = d / "model.pt"
            if model_path.exists():
                router.payload["state_dict"] = torch.load(
                    model_path, map_location="cpu", weights_only=True
                )
            tok = d / "tokenizer"
            if tok.exists():
                router.meta["tokenizer_dir"] = str(tok)
        return router


def _prompt_row(prompt: str) -> RouterRow:
    """Minimal RouterRow shell so feature extraction works on raw prompts."""
    return RouterRow(
        id=stable_id(prompt),
        source="inference",
        task_family="unknown",
        prompt=prompt,
        group_id=stable_id(prompt),
        split="train",
        local_score=0.0,
        frontier_score=0.0,
        delta_q=0.0,
    )


# ---------------------------------------------------------------------------
# Trainers
# ---------------------------------------------------------------------------


def _labels(rows: list[RouterRow], mode: str = "strict") -> tuple[np.ndarray, np.ndarray]:
    """(local_sufficient, delta_q) label arrays.

    mode="strict": plan §5 label — local_score >= floor AND delta_q <= eps
    (falls back to 0.7/0.1 when unset). Conservative: also demands absolute
    quality, not just local-beats-frontier.
    mode="oracle": matches the oracle's argmax rule — local is sufficient
    whenever local_score >= frontier_score. Trains ElJefe to imitate the
    quality-optimal cheap route rather than a stricter surrogate.
    """
    if mode == "oracle":
        y = np.array(
            [float(r.local_score) >= float(r.frontier_score) for r in rows],
            dtype=np.int64,
        )
    else:
        y = np.array(
            [bool(r.local_sufficient) if r.local_sufficient is not None
             else (r.local_score >= 0.7 and r.delta_q <= 0.1)
             for r in rows],
            dtype=np.int64,
        )
    g = np.array([float(r.delta_q) for r in rows], dtype=np.float64)
    return y, g


def labels_for(rows: list[RouterRow], mode: str = "strict") -> tuple[np.ndarray, np.ndarray]:
    """(local_sufficient, delta_q) arrays; derives labels from scores when unset."""
    return _labels(rows, mode=mode)


def train_tfidf(train_rows: list[RouterRow], cfg: dict[str, Any]) -> Router:
    """Baseline 2: TfidfVectorizer + LogisticRegression on local_sufficient.

    Also fits a Ridge regressor on delta_q so predict_gain() is meaningful.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression, Ridge

    seed = int(cfg.get("seed", 42))
    seed_everything(seed)
    tf = dict(cfg.get("tfidf") or {})
    vec = TfidfVectorizer(
        max_features=int(tf.get("max_features", 50000)),
        ngram_range=tuple(tf.get("ngram_range", [1, 2])),
    )
    prompts = [r.prompt for r in train_rows]
    X = vec.fit_transform(prompts)
    y, g = _labels(train_rows, mode=str(cfg.get("label_mode", "strict")))

    if len(np.unique(y)) < 2:
        p = float(y.mean()) if len(y) else 0.5
        return Router(
            kind="constant",
            payload={"p": p},
            meta={"router": "tfidf", "degenerate": "single_class", "seed": seed},
        )

    clf = LogisticRegression(
        C=float(tf.get("logistic_C", 1.0)),
        max_iter=1000,
        random_state=seed,
    )
    clf.fit(X, y)
    reg = Ridge()
    reg.fit(X, g)
    return Router(
        kind="tfidf",
        payload={"vec": vec, "clf": clf, "reg": reg},
        meta={
            "router": "tfidf",
            "seed": seed,
            "train_rows": len(train_rows),
            "tfidf": tf,
        },
    )


def train_embedding_router(
    rows: list[RouterRow],
    embedder_name: Optional[str] = None,
    cfg: Optional[dict[str, Any]] = None,
    embeddings: Optional[np.ndarray] = None,
) -> Router:
    """Baseline 3: frozen embeddings + metadata -> LightGBM cls + reg."""
    cfg = cfg or {}
    emb_cfg = dict(cfg.get("embedding") or {})
    seed = int(cfg.get("seed", 42))
    seed_everything(seed)
    embedder_name = embedder_name or emb_cfg.get(
        "model", "sentence-transformers/all-MiniLM-L6-v2"
    )

    if embeddings is None:
        embeddings = embed_prompts(
            [r.prompt for r in rows],
            embedder_name,
            batch_size=int(emb_cfg.get("batch_size", 64)),
        )
    embeddings = np.asarray(embeddings, dtype=np.float32)
    if embeddings.shape[0] != len(rows):
        raise ValueError(
            f"embeddings rows {embeddings.shape[0]} != dataset rows {len(rows)}"
        )

    featurizer = FeatureVectorizer()
    F = featurizer.fit_transform([extract_row_features(r) for r in rows])
    X = np.hstack([embeddings, F])
    y, g = _labels(rows, mode=str(cfg.get("label_mode", "strict")))

    common = dict(
        n_estimators=int(emb_cfg.get("n_estimators", 400)),
        learning_rate=float(emb_cfg.get("learning_rate", 0.05)),
        num_leaves=int(emb_cfg.get("num_leaves", 31)),
        random_state=seed,
        n_jobs=int(emb_cfg.get("n_jobs", 1)),  # lgbm multithread segfaults on macOS arm64
    )
    backend = "lightgbm"
    try:
        import lightgbm as lgb

        clf = lgb.LGBMClassifier(**common)
        reg = lgb.LGBMRegressor(**common)
    except ImportError:
        from sklearn.ensemble import (
            HistGradientBoostingClassifier,
            HistGradientBoostingRegressor,
        )

        backend = "hist_gradient_boosting"
        clf = HistGradientBoostingClassifier(
            max_iter=common["n_estimators"],
            learning_rate=common["learning_rate"],
            max_leaf_nodes=common["num_leaves"],
            random_state=seed,
        )
        reg = HistGradientBoostingRegressor(
            max_iter=common["n_estimators"],
            learning_rate=common["learning_rate"],
            max_leaf_nodes=common["num_leaves"],
            random_state=seed,
        )

    if len(np.unique(y)) < 2:
        p = float(y.mean()) if len(y) else 0.5
        return Router(
            kind="constant",
            payload={"p": p},
            meta={"router": "embedding", "degenerate": "single_class", "seed": seed},
        )

    clf.fit(X, y)
    reg.fit(X, g)
    return Router(
        kind="embedding",
        payload={"clf": clf, "reg": reg},
        featurizer=featurizer,
        embedder_name=embedder_name,
        use_features=True,
        meta={
            "router": "embedding",
            "backend": backend,
            "embedder": embedder_name,
            "seed": seed,
            "train_rows": len(rows),
            "feature_dim": int(X.shape[1]),
        },
    )


# ---------------------------------------------------------------------------
# ElJefe v0 — fine-tuned dual-head MiniLM (plan §9)
# ---------------------------------------------------------------------------


def build_minilm_module(encoder_name: str):
    """Encoder + sigmoid cls head + linear reg head on mean-pooled output."""
    import torch
    from torch import nn
    from transformers import AutoModel

    class MiniLMRouter(nn.Module):
        def __init__(self, name: str):
            super().__init__()
            self.encoder = AutoModel.from_pretrained(name)
            hidden = self.encoder.config.hidden_size
            self.cls_head = nn.Linear(hidden, 1)
            self.reg_head = nn.Linear(hidden, 1)

        def forward(self, input_ids=None, attention_mask=None, **kw):
            out = self.encoder(
                input_ids=input_ids, attention_mask=attention_mask, **kw
            )
            hidden = out.last_hidden_state
            mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            return self.cls_head(pooled).squeeze(-1), self.reg_head(pooled).squeeze(-1)

    return MiniLMRouter(encoder_name)


def train_minilm_router(rows: list[RouterRow], cfg: dict[str, Any]) -> Router:
    """ElJefe v0: fine-tune a MiniLM-sized encoder with dual heads.

    Loss = BCE(local_sufficient) + lambda * Huber(delta_q)  (plan §9).
    Early stopping monitors validation routing utility — quality_retention
    at threshold 0.9 via eljefe.metrics.routing_metrics (utility as tiebreak).
    Saves best checkpoint + tokenizer + config + manifest to cfg.out_dir.
    """
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoTokenizer, get_linear_schedule_with_warmup

    from eljefe.metrics import routing_metrics

    seed = int(cfg.get("seed", 42))
    seed_everything(seed)
    encoder_name = cfg.get("encoder", "sentence-transformers/all-MiniLM-L6-v2")
    max_length = int(cfg.get("max_length", 512))
    lam = float(cfg.get("lambda_regression", 1.0))
    heads = dict(cfg.get("heads") or {})
    use_cls = bool(heads.get("classification", True))
    use_reg = bool(heads.get("regression", True))
    if not (use_cls or use_reg):
        use_cls = True  # a router with no heads is meaningless
    tc = dict(cfg.get("train") or {})
    epochs = int(tc.get("epochs", 10))
    batch_size = int(tc.get("batch_size", 32))
    lr = float(tc.get("lr", 2.0e-5))
    weight_decay = float(tc.get("weight_decay", 0.01))
    warmup_ratio = float(tc.get("warmup_ratio", 0.1))
    patience = int(tc.get("early_stopping_patience", 3))
    out_dir = Path(cfg.get("out_dir", "artifacts/models/eljefe-v0"))
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    train_rows = [r for r in rows if r.split == "train"] or rows
    val_rows = [r for r in rows if r.split == "validation"]
    if not val_rows:  # hold out a deterministic slice
        rng = random.Random(seed)
        idx = list(range(len(train_rows)))
        rng.shuffle(idx)
        cut = max(1, int(0.1 * len(idx)))
        val_rows = [train_rows[i] for i in idx[:cut]]
        train_rows = [train_rows[i] for i in idx[cut:]] or train_rows

    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    use_amp = bool(tc.get("fp16", True)) and device == "cuda"
    tokenizer = AutoTokenizer.from_pretrained(encoder_name)
    model = build_minilm_module(encoder_name).to(device)

    label_mode = str(cfg.get("label_mode", "strict"))

    class _DS(Dataset):
        def __init__(self, rs: list[RouterRow]):
            self.rs = rs

        def __len__(self):
            return len(self.rs)

        def __getitem__(self, i):
            r = self.rs[i]
            y, g = _labels([r], mode=label_mode)
            return r.prompt, float(y[0]), float(g[0])

    def collate(batch):
        prompts, ys, gs = zip(*batch)
        enc = tokenizer(
            list(prompts),
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        return (
            enc,
            torch.tensor(ys, dtype=torch.float32),
            torch.tensor(gs, dtype=torch.float32),
        )

    loader = DataLoader(
        _DS(train_rows), batch_size=batch_size, shuffle=True,
        collate_fn=collate, drop_last=False,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    steps = max(1, epochs * len(loader))
    sched = get_linear_schedule_with_warmup(
        opt, int(warmup_ratio * steps), steps
    )
    bce = nn.BCEWithLogitsLoss()
    huber = nn.HuberLoss()

    def evaluate() -> dict[str, float]:
        model.eval()
        ps: list[np.ndarray] = []
        with torch.no_grad():
            for i in range(0, len(val_rows), batch_size):
                enc = tokenizer(
                    [r.prompt for r in val_rows[i : i + batch_size]],
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                ).to(device)
                cls_logit, _ = model(**enc)
                ps.append(torch.sigmoid(cls_logit).cpu().numpy())
        p = np.concatenate(ps) if ps else np.zeros(0)
        m = routing_metrics(val_rows, list(map(float, p)), threshold=0.9)
        return {"quality_retention": float(m.get("quality_retention", 0.0)),
                "utility": float(m.get("utility", m.get("quality_retention", 0.0)))}

    best_key = (-1.0, -1.0)
    best_state = None
    best_metrics: dict[str, float] = {}
    bad_epochs = 0
    history: list[dict[str, Any]] = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for enc, ys, gs in loader:
            enc = {k: v.to(device) for k, v in enc.items()}
            ys = ys.to(device)
            gs = gs.to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                enabled=use_amp):
                cls_logit, reg = model(**enc)
                loss = torch.zeros((), device=device)
                if use_cls:
                    loss = loss + bce(cls_logit, ys)
                if use_reg:
                    loss = loss + lam * huber(reg, gs)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            total_loss += float(loss.detach().cpu())
        m = evaluate()
        key = (m["quality_retention"], m["utility"])
        history.append({"epoch": epoch, "train_loss": total_loss / max(1, len(loader)), **m})
        print(f"[minilm] epoch {epoch}: loss={history[-1]['train_loss']:.4f} "
              f"val quality_retention={m['quality_retention']:.4f} utility={m['utility']:.4f}")
        if key > best_key:
            best_key = key
            best_metrics = m
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"[minilm] early stop at epoch {epoch}")
                break

    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    router = Router(
        kind="minilm",
        payload={
            "state_dict": best_state,
            "encoder_name": encoder_name,
            "max_length": max_length,
        },
        meta={
            "router": "minilm",
            "encoder": encoder_name,
            "seed": seed,
            "train_rows": len(train_rows),
            "val_rows": len(val_rows),
            "best_val": best_metrics,
            "history": history,
        },
    )
    router._tokenizer = tokenizer
    router.save(out_dir)

    # config.yaml + manifest.json (plan §32)
    import yaml

    (out_dir / "config.yaml").write_text(yaml.safe_dump(dict(cfg), sort_keys=False))
    manifest = {
        "git_commit": git_commit(),
        "router_model": encoder_name,
        "seed": seed,
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "best_val": best_metrics,
        "history": history,
        "lambda_regression": lam,
        "max_length": max_length,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return router
