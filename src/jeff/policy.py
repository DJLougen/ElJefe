"""Hard-routing rules outside Jeff (plan §21).

These are capability constraints, not learned preferences. ``hard_route``
returns a forced route or ``None``; ``apply_policy`` layers Jeff's
probability on top with a configurable threshold.
"""

from __future__ import annotations

from typing import Optional

from .schema import RouterInput, RouterOutput

# E4B routing assumption: only pure-text requests stay eligible for local.
_LOCAL_MODALITIES = {"text"}


def hard_route(inp: RouterInput) -> Optional[str]:
    """Return "local" | "frontier" when a hard rule fires, else None.

    Order matters: privacy and a disabled frontier pin the route to local
    before any capability check can push it to frontier.
    """
    md = inp.metadata or {}

    # privacy == local_only -> never frontier
    if inp.privacy == "local_only":
        return "local"

    # frontier disabled -> local
    if md.get("frontier_disabled"):
        return "local"

    # context > local limit -> frontier (or a context strategy; v0: frontier)
    context = inp.input_tokens
    if context is None:
        context = md.get("context_len") or md.get("input_tokens")
    if context is not None and int(context) > inp.local_context_limit:
        return "frontier"

    # required modality not text -> frontier
    for modality in inp.modalities or ["text"]:
        if str(modality).lower() not in _LOCAL_MODALITIES:
            return "frontier"

    # required tool unavailable -> frontier
    required = md.get("required_tools") or []
    available = set(inp.tools_available or [])
    if any(tool not in available for tool in required):
        return "frontier"

    return None


def apply_policy(
    out: RouterOutput,
    inp: RouterInput,
    threshold: float = 0.90,
) -> str:
    """Hard rules first; else local iff p_local_sufficient >= threshold."""
    forced = hard_route(inp)
    if forced is not None:
        return forced
    return "local" if out.p_local_sufficient >= threshold else "frontier"
