"""Routing evaluation metrics (plan §11).

Simulate the routed system at a threshold over counterfactual rows (each row
has both local_score and frontier_score) and report economic metrics, not
classifier accuracy.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from .schema import RouterRow

_DEFAULT_QUALITY_FLOOR = 0.70
_DEFAULT_EPSILON = 0.10


def _row_epsilon(row: RouterRow) -> float:
    return float((row.metadata or {}).get("epsilon", _DEFAULT_EPSILON))


def _row_quality_floor(row: RouterRow) -> float:
    return float((row.metadata or {}).get("quality_floor", _DEFAULT_QUALITY_FLOOR))


def oracle_route(row: RouterRow, quality_floor: float, epsilon: float) -> str:
    """Best route given measured counterfactuals (plan §5 label rule)."""
    if row.local_score >= quality_floor and row.delta_q <= epsilon:
        return "local"
    return "frontier"


def routing_metrics(
    rows: Sequence[RouterRow],
    p_local: Sequence[float],
    threshold: float,
) -> dict:
    """Simulate 'route local iff p >= threshold' and score the outcome.

    Rates are fractions of all rows. ``false_local_rate`` counts rows routed
    local where frontier was materially better (delta_q > epsilon) — the
    critical error class. ``false_frontier_rate`` counts rows routed frontier
    where local would have sufficed (local_score >= quality_floor and
    delta_q <= epsilon).
    """
    rows = list(rows)
    p = np.asarray(list(p_local), dtype=float)
    n = len(rows)
    if n == 0:
        return {
            "n_rows": 0,
            "quality_routed": 0.0,
            "quality_retention": 0.0,
            "local_rate": 0.0,
            "frontier_rate": 0.0,
            "cost_reduction": 0.0,
            "false_local_rate": 0.0,
            "false_frontier_rate": 0.0,
            "oracle_utility": 0.0,
            "utility": 0.0,
            "regret": 0.0,
        }
    if p.shape[0] != n:
        raise ValueError(f"p_local has {p.shape[0]} entries for {n} rows")

    costs = np.array(
        [r.frontier_cost if r.frontier_cost is not None else np.nan for r in rows],
        dtype=float,
    )
    fallback_cost = float(np.nanmean(costs)) if np.isfinite(costs).any() else 0.0
    costs = np.where(np.isfinite(costs), costs, fallback_cost)

    routed_local = p >= threshold
    chosen = np.array(
        [r.local_score if routed_local[i] else r.frontier_score for i, r in enumerate(rows)]
    )
    quality_routed = float(np.mean(chosen))
    quality_frontier = float(np.mean([r.frontier_score for r in rows]))

    routed_cost = float(np.sum(np.where(routed_local, 0.0, costs)))
    always_frontier_cost = float(np.sum(costs))

    false_local = 0
    false_frontier = 0
    for i, r in enumerate(rows):
        eps = _row_epsilon(r)
        floor = _row_quality_floor(r)
        if routed_local[i] and r.delta_q > eps:
            false_local += 1
        if not routed_local[i] and r.local_score >= floor and r.delta_q <= eps:
            false_frontier += 1

    oracle_utility = float(
        np.mean([max(r.local_score, r.frontier_score) for r in rows])
    )

    return {
        "n_rows": n,
        "threshold": float(threshold),
        "quality_routed": quality_routed,
        "quality_retention": quality_routed / quality_frontier if quality_frontier else 0.0,
        "local_rate": float(np.mean(routed_local)),
        "frontier_rate": float(1.0 - np.mean(routed_local)),
        "routed_cost": routed_cost,
        "always_frontier_cost": always_frontier_cost,
        "cost_reduction": (
            1.0 - routed_cost / always_frontier_cost if always_frontier_cost else 0.0
        ),
        "false_local_rate": false_local / n,
        "false_frontier_rate": false_frontier / n,
        "oracle_utility": oracle_utility,
        "utility": quality_routed,
        "regret": oracle_utility - quality_routed,
    }


def threshold_sweep(
    rows: Sequence[RouterRow],
    p_local: Sequence[float],
    thresholds: Optional[Sequence[float]] = None,
):
    """One row of routing metrics per threshold -> pandas DataFrame."""
    import pandas as pd

    if thresholds is None:
        thresholds = np.linspace(0.0, 1.0, 51)
    records = [routing_metrics(rows, p_local, float(t)) for t in thresholds]
    return pd.DataFrame.from_records(records)
