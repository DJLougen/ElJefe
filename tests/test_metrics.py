"""Routing metrics on a hand-computed dataset + calibrator Brier check."""

import numpy as np
import pytest

from eljefe.calibration import Calibrator
from eljefe.metrics import oracle_route, routing_metrics, threshold_sweep
from eljefe.schema import RouterRow


def _row(i, local, frontier, cost=None, **md):
    return RouterRow(
        id=f"r{i}", source="s", task_family="f", prompt="p",
        group_id=f"r{i}", split="test_iid",
        local_score=local, frontier_score=frontier,
        delta_q=frontier - local, frontier_cost=cost, metadata=md,
    )


# floor=0.70, eps=0.10 (defaults). r1/r2/r4 local-sufficient; r3 needs frontier.
ROWS = [
    _row(1, 0.90, 0.95, cost=0.02),
    _row(2, 0.80, 0.80, cost=0.04),
    _row(3, 0.50, 0.90, cost=0.01),
    _row(4, 0.95, 0.90, cost=None),
]
P = [0.95, 0.95, 0.92, 0.50]  # threshold 0.9 -> routes [L, L, L, F]


def test_routing_metrics_hand_computed():
    m = routing_metrics(ROWS, P, threshold=0.9)
    assert m["n_rows"] == 4
    assert m["quality_routed"] == pytest.approx(0.775)
    assert m["quality_retention"] == pytest.approx(0.775 / 0.8875)
    assert m["local_rate"] == pytest.approx(0.75)
    assert m["frontier_rate"] == pytest.approx(0.25)
    # fallback cost for r4 = mean(0.02, 0.04, 0.01) = 0.023333...
    assert m["routed_cost"] == pytest.approx(0.07 / 3)
    assert m["always_frontier_cost"] == pytest.approx(0.07 + 0.07 / 3)
    assert m["cost_reduction"] == pytest.approx(0.75)
    assert m["false_local_rate"] == pytest.approx(0.25)   # r3
    assert m["false_frontier_rate"] == pytest.approx(0.25)  # r4
    assert m["oracle_utility"] == pytest.approx(0.9)
    assert m["utility"] == pytest.approx(m["quality_routed"])
    assert m["regret"] == pytest.approx(0.9 - 0.775)


def test_routing_metrics_per_row_overrides():
    rows = [_row(1, 0.9, 0.95, epsilon=0.01)]  # delta 0.05 > 0.01 -> false local
    m = routing_metrics(rows, [0.99], threshold=0.5)
    assert m["false_local_rate"] == 1.0
    # quality_floor override: local 0.9 < floor 0.95 -> not "would suffice"
    rows2 = [_row(1, 0.9, 0.95, quality_floor=0.95)]
    m2 = routing_metrics(rows2, [0.0], threshold=0.5)  # routed frontier
    assert m2["false_frontier_rate"] == 0.0


def test_oracle_route():
    assert oracle_route(ROWS[0], 0.70, 0.10) == "local"
    assert oracle_route(ROWS[2], 0.70, 0.10) == "frontier"
    assert oracle_route(ROWS[3], 0.70, 0.10) == "local"  # local beats frontier


def test_threshold_sweep_extremes():
    df = threshold_sweep(ROWS, P)
    assert len(df) == 51
    all_local = df.iloc[0]   # threshold 0.0
    assert all_local["local_rate"] == 1.0
    assert all_local["quality_routed"] == pytest.approx(
        np.mean([r.local_score for r in ROWS])
    )
    all_frontier = df.iloc[-1]  # threshold 1.0, all p < 1
    assert all_frontier["local_rate"] == 0.0
    assert all_frontier["quality_retention"] == pytest.approx(1.0)
    assert all_frontier["cost_reduction"] == pytest.approx(0.0)
    assert all_frontier["false_frontier_rate"] == pytest.approx(0.75)  # r1,r2,r4


def test_routing_metrics_empty_and_mismatch():
    m = routing_metrics([], [], 0.5)
    assert m["n_rows"] == 0
    with pytest.raises(ValueError):
        routing_metrics(ROWS, [0.5], 0.5)


# --- calibration ------------------------------------------------------------


def _skewed(n=800, seed=0):
    """Raw probs overconfident by ~3x on the logit scale."""
    rng = np.random.default_rng(seed)
    p_raw = rng.uniform(0.05, 0.95, n)
    logits = np.log(p_raw / (1 - p_raw))
    p_true = 1 / (1 + np.exp(-logits / 3.0))
    y = (rng.uniform(0, 1, n) < p_true).astype(float)
    return p_raw, y


def _brier(p, y):
    return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))


@pytest.mark.parametrize("method", ["isotonic", "platt", "temperature"])
def test_calibrator_improves_brier(method):
    p_raw, y = _skewed()
    tr, te = slice(0, 400), slice(400, None)
    cal = Calibrator(method=method).fit(p_raw[tr], y[tr])
    p_cal = cal.predict(p_raw[te])
    assert _brier(p_cal, y[te]) < _brier(p_raw[te], y[te])
    assert np.all((p_cal > 0) & (p_cal < 1))


def test_calibrator_save_load(tmp_path):
    p_raw, y = _skewed(n=200)
    cal = Calibrator(method="temperature").fit(p_raw, y)
    path = tmp_path / "cal.pkl"
    cal.save(path)
    back = Calibrator.load(path)
    assert back.method == "temperature"
    np.testing.assert_allclose(back.predict(p_raw), cal.predict(p_raw))


def test_calibrator_clips_extremes():
    p_raw, y = _skewed(n=200)
    cal = Calibrator(method="isotonic").fit(p_raw, y)
    out = cal.predict(np.array([0.0, 1.0, 0.5]))
    assert np.all(np.isfinite(out))
    assert np.all((out >= 0) & (out <= 1))


def test_calibrator_bad_method():
    with pytest.raises(ValueError):
        Calibrator(method="nope")
    with pytest.raises(RuntimeError):
        Calibrator("isotonic").predict(np.array([0.5]))
