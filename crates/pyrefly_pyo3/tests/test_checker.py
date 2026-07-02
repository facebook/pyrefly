# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import subprocess
import tempfile
import time
from pathlib import Path

import pyrefly_api

_HELPER = "def greet() -> int: ...\n"
_SNIPPET = "from helper import greet\nx: str = greet()"

# need the release target
_CLI = os.environ.get(
    "PYREFLY_BIN", str(Path(__file__).resolve().parents[3] / "target/release/pyrefly")
)


def _avg_ms(iterations, call):
    start = time.perf_counter()
    for i in range(iterations):
        call(i)
    return (time.perf_counter() - start) / iterations * 1000


def benchmark(iterations=50):
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        (root / "pyrefly.toml").write_text("")
        (root / "helper.py").write_text(_HELPER)
        (root / "main.py").write_text(_SNIPPET)

        old = _avg_ms(iterations, lambda i: subprocess.run(
            [_CLI, "check", str(root / "main.py")], cwd=root, capture_output=True))

        checker = pyrefly_api.Checker(project_root=str(root))
        checker.check(_SNIPPET) # warm; try to make it random
        new = _avg_ms(iterations, lambda i: checker.check(f"{_SNIPPET}  # {i}"))

    print(f"iter: {iterations}")
    print(f"subprocess : {old:.3f} ms/call")
    print(f"in process: {new:.3f} ms/call")
    print(f"mult: {old / new:.1f}x")


def test_checker_resolves_imports_and_reports_errors(tmp_path):
    (tmp_path / "pyrefly.toml").write_text("")
    (tmp_path / "helper.py").write_text("def greet() -> int: ...\n")

    checker = pyrefly_api.Checker(project_root=str(tmp_path))

    diags = checker.check("from helper import greet\nx: str = greet()")
    assert [d.kind for d in diags] == ["bad-assignment"]

    # The warm Checker is reusable across calls.
    assert checker.check("from helper import greet\ny: int = greet()") == []


if __name__ == "__main__":
    benchmark()
