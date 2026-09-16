"""Kernel-safe argument parsing for Jeff scripts.

`colab exec -f script.py` runs inside a Jupyter kernel whose sys.argv is the
kernel launcher's (e.g. `colab_kernel_launcher.py -f kernel-xxx.json`), so
plain parser.parse_args() crashes. There is also no way to pass script args
through `colab exec`, so JEFF_<ARG> environment variables act as overrides
(kernel env persists across exec calls in a session).

Usage in scripts: replace `parser.parse_args()` with
`args = jeff.cli.parse_args(parser)` (or `from jeff.cli import parse_args`).
"""

from __future__ import annotations

import argparse
import os
import sys


def _kernel_argv() -> list[str]:
    """sys.argv minus jupyter kernel launcher args."""
    argv = list(sys.argv[1:])
    out: list[str] = []
    skip = False
    for i, a in enumerate(argv):
        if skip:
            skip = False
            continue
        # kernel launcher: -f /path/kernel-*.json or --config <file>
        if a in ("-f", "--config") and i + 1 < len(argv) and "kernel-" in argv[i + 1]:
            skip = True
            continue
        if "kernel-" in a and a.endswith(".json"):
            continue
        out.append(a)
    return out


def parse_args(parser: argparse.ArgumentParser, env_prefix: str = "JEFF_") -> argparse.Namespace:
    """parse_known_args on kernel-stripped argv, then apply env overrides.

    For every optional --foo-bar, env var JEFF_FOO_BAR (uppercased, dashes to
    underscores) overrides the parsed value when set. Values are coerced with
    the argument's `type` callable when present, else kept as strings; flags
    (store_true/store_false) accept 1/true/yes/on.
    """
    args, _unknown = parser.parse_known_args(_kernel_argv())

    for action in parser._actions:  # noqa: SLF001 - argparse has no public API for this
        if not action.option_strings or action.dest in ("help",):
            continue
        env_name = env_prefix + action.dest.upper().replace("-", "_")
        raw = os.environ.get(env_name)
        if raw is None:
            continue
        if isinstance(action, (argparse._StoreTrueAction, argparse._StoreFalseAction)):
            setattr(args, action.dest, raw.strip().lower() in ("1", "true", "yes", "on"))
        elif action.type is not None:
            try:
                setattr(args, action.dest, action.type(raw))
            except Exception:
                print(f"[jeff.cli] ignoring invalid {env_name}={raw!r}")
        else:
            setattr(args, action.dest, raw)
    return args
