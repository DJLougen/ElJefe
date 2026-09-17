"""Hard-routing rules (plan §21) and threshold policy."""

from eljefe.policy import apply_policy, hard_route
from eljefe.schema import RouterInput, RouterOutput


def _inp(**kw) -> RouterInput:
    return RouterInput(prompt="p", **kw)


def _out(p: float) -> RouterOutput:
    return RouterOutput(p_local_sufficient=p, expected_frontier_gain=0.0)


def test_privacy_local_only_forces_local():
    assert hard_route(_inp(privacy="local_only")) == "local"
    # privacy wins even over capability rules that would push frontier
    assert hard_route(
        _inp(privacy="local_only", input_tokens=10**9, modalities=["audio"])
    ) == "local"


def test_frontier_disabled_forces_local():
    assert hard_route(_inp(metadata={"frontier_disabled": True})) == "local"
    assert hard_route(_inp(metadata={"frontier_disabled": False})) is None


def test_context_over_limit_goes_frontier():
    assert hard_route(_inp(input_tokens=200_000, local_context_limit=131_072)) == "frontier"
    assert hard_route(_inp(input_tokens=131_072, local_context_limit=131_072)) is None
    # metadata-provided context length also counts
    assert hard_route(_inp(metadata={"context_len": 999_999})) == "frontier"


def test_non_text_modality_goes_frontier():
    assert hard_route(_inp(modalities=["text"])) is None
    assert hard_route(_inp(modalities=["text", "image"])) == "frontier"
    assert hard_route(_inp(modalities=["audio"])) == "frontier"


def test_required_tool_unavailable_goes_frontier():
    assert hard_route(
        _inp(tools_available=["python"], metadata={"required_tools": ["web"]})
    ) == "frontier"
    assert hard_route(
        _inp(tools_available=["python", "web"], metadata={"required_tools": ["web"]})
    ) is None
    assert hard_route(_inp(metadata={"required_tools": []})) is None


def test_no_rule_fires_returns_none():
    assert hard_route(_inp()) is None


def test_apply_policy_hard_rules_win():
    # Even a confident local score cannot override a frontier hard rule.
    assert apply_policy(_out(0.99), _inp(modalities=["audio"]), threshold=0.9) == "frontier"
    # And a low score cannot override a local hard rule.
    assert apply_policy(_out(0.01), _inp(privacy="local_only"), threshold=0.9) == "local"


def test_apply_policy_threshold():
    inp = _inp()
    assert apply_policy(_out(0.95), inp, threshold=0.90) == "local"
    assert apply_policy(_out(0.90), inp, threshold=0.90) == "local"  # >= boundary
    assert apply_policy(_out(0.89), inp, threshold=0.90) == "frontier"
    assert apply_policy(_out(0.5), inp, threshold=0.5) == "local"
