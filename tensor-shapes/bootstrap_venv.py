#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Create the shared virtualenv that the tensor-shape runtime tests need.

This is deliberately the only part of the harness that touches the network, so
that everything else runs offline. Running the test suites never calls this;
they fail with a pointer here instead. Bootstrapping is therefore a step a
human, or an agent launched with `claude --secure-internet-mode`, performs once
per machine.

One virtualenv serves torch, numpy, and jax together. The runtime tests only
construct arrays and read shapes, so the libraries do not interfere and there
is nothing to gain from isolating them.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

from shape_testing import DEFAULT_VENV, TENSOR_SHAPES_ROOT


PYTHON_VERSION = "3.12"
REQUIREMENTS: Path = TENSOR_SHAPES_ROOT / "test-requirements.txt"

# Meta hosts have no direct egress; `--fwdproxy` routes uv through the forward
# proxy the same way facebook/setup_cargo.sh does for Cargo.
FWDPROXY = "http://fwdproxy:8080"


def resolve_uv(explicit: Path | None) -> str:
    if explicit is not None:
        if not explicit.exists():
            raise SystemExit(f"--uv {explicit} does not exist")
        return str(explicit)
    found = shutil.which("uv")
    if found is None:
        raise SystemExit(
            "uv is not on PATH.\n\n"
            "Install it from https://docs.astral.sh/uv/getting-started/installation/, "
            "or pass `--uv /path/to/uv`."
        )
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--venv",
        type=Path,
        default=Path(os.environ.get("TENSOR_SHAPES_VENV", DEFAULT_VENV)),
        help="where to create the virtualenv (default: $TENSOR_SHAPES_VENV or ~/.tensor-shapes-venv)",
    )
    parser.add_argument("--uv", type=Path, default=None, help="path to the uv binary")
    parser.add_argument(
        "--fwdproxy",
        action="store_true",
        help="route downloads through Meta's forward proxy; required on devservers and Sandcastle",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="recreate the virtualenv even if it already exists",
    )
    args = parser.parse_args()

    uv = resolve_uv(args.uv)
    venv: Path = args.venv

    env = os.environ.copy()
    if args.fwdproxy:
        env.update({"http_proxy": FWDPROXY, "https_proxy": FWDPROXY})

    # `--allow-existing` so that re-running against an already-bootstrapped
    # virtualenv refreshes it instead of failing; `--force` recreates it.
    run(
        [uv, "venv", "--python", PYTHON_VERSION, str(venv)]
        + (["--clear"] if args.force else ["--allow-existing"]),
        env=env,
    )
    # Install runs unconditionally, even when the virtualenv already exists.
    # Returning early for an existing directory would leave a stale environment
    # behind whenever the pins change or a previous run was interrupted, and
    # that surfaces later as a confusing shape mismatch rather than as a
    # bootstrap problem. Re-installing is a fast no-op once satisfied.
    #
    # Deliberately `install` rather than `sync`: this file pins direct
    # dependencies but is not a fully resolved lock, so `sync` would treat every
    # transitive dependency as extraneous and uninstall it.
    python = venv / ("Scripts" if os.name == "nt" else "bin") / "python"
    run(
        [uv, "pip", "install", "--python", str(python), "-r", str(REQUIREMENTS)],
        env=env,
    )

    print(f"\nBootstrapped {venv}.", flush=True)
    return 0


def run(command: list[str], *, env: dict[str, str]) -> None:
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, env=env, check=True)


if __name__ == "__main__":
    sys.exit(main())
