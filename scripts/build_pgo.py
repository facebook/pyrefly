#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Build a native Pyrefly binary using profile-guided optimization."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import tempfile
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WORKLOAD = ("pyrefly/lib/test", "conformance/third_party")


def run(command: Sequence[str], *, env: dict[str, str] | None = None) -> None:
    """Run a command from the repository root."""
    print("+", shlex.join(command), flush=True)
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def rustflags(extra: Sequence[str]) -> dict[str, str]:
    """Return an environment with `extra` appended to inherited Rust flags."""
    env = os.environ.copy()
    inherited = env.pop("CARGO_ENCODED_RUSTFLAGS", "").split("\x1f")
    if inherited == [""]:
        inherited = shlex.split(env.pop("RUSTFLAGS", ""), posix=os.name != "nt")
    else:
        env.pop("RUSTFLAGS", None)
    flags = [*inherited, *extra]
    env["CARGO_ENCODED_RUSTFLAGS"] = "\x1f".join(flags)
    return env


def host_triple() -> str:
    """Return the host triple of the active Rust toolchain."""
    output = subprocess.check_output(["rustc", "-vV"], text=True)
    for line in output.splitlines():
        if line.startswith("host: "):
            return line[len("host: ") :]
    raise RuntimeError("rustc -vV did not report a host triple")


def llvm_profdata() -> Path:
    """Find llvm-profdata from the active Rust toolchain."""
    target_libdir = Path(
        subprocess.check_output(
            ["rustc", "--print", "target-libdir"], text=True
        ).strip()
    )
    executable = "llvm-profdata.exe" if os.name == "nt" else "llvm-profdata"
    path = target_libdir.parent / "bin" / executable
    if not path.is_file():
        raise RuntimeError(
            "llvm-profdata was not found in the Rust toolchain; install it with "
            "`rustup component add llvm-tools-preview`"
        )
    return path


def target_directory() -> Path:
    """Return Cargo's target directory for this workspace."""
    output = subprocess.check_output(
        ["cargo", "metadata", "--format-version=1", "--no-deps"],
        cwd=ROOT,
        text=True,
    )
    return Path(json.loads(output)["target_directory"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target",
        default=host_triple(),
        help="Native Rust target triple to build (defaults to the host triple)",
    )
    parser.add_argument(
        "--all-features", action="store_true", help="Build with all Cargo features"
    )
    parser.add_argument(
        "workload",
        nargs="*",
        default=DEFAULT_WORKLOAD,
        help="Python files or directories used to train the instrumented binary",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    host = host_triple()
    if args.target != host:
        raise RuntimeError(
            f"PGO training must execute natively; target {args.target} does not match "
            f"host {host}"
        )

    build = [
        "cargo",
        "build",
        "--release",
        "--manifest-path",
        "pyrefly/Cargo.toml",
        "-p",
        "pyrefly",
        "--target",
        args.target,
    ]
    if args.all_features:
        build.append("--all-features")

    target_dir = target_directory()
    target_dir.mkdir(parents=True, exist_ok=True)
    binary_name = "pyrefly.exe" if os.name == "nt" else "pyrefly"
    binary = target_dir / args.target / "release" / binary_name
    with tempfile.TemporaryDirectory(prefix="pyrefly-pgo-", dir=target_dir) as raw:
        raw_profiles = Path(raw)
        generate_env = rustflags([f"-Cprofile-generate={raw_profiles}"])
        run(build, env=generate_env)

        training = [
            str(binary),
            "check",
            "--output-format",
            "omit-errors",
            "--summary",
            "none",
            "--progress-bar",
            "no",
            *args.workload,
        ]
        print("+", shlex.join(training), flush=True)
        result = subprocess.run(training, cwd=ROOT, env=generate_env)
        if result.returncode not in (0, 1):
            result.check_returncode()

        profiles = list(raw_profiles.glob("*.profraw"))
        if not profiles:
            raise RuntimeError("the training workload did not produce a raw profile")
        merged_profile = raw_profiles / "merged.profdata"
        run(
            [
                str(llvm_profdata()),
                "merge",
                "-o",
                str(merged_profile),
                *map(str, profiles),
            ]
        )

        use_env = rustflags(
            [
                f"-Cprofile-use={merged_profile}",
                "-Cllvm-args=-pgo-warn-missing-function",
            ]
        )
        run(build, env=use_env)

    print(f"PGO-optimized binary: {binary}")


if __name__ == "__main__":
    main()
