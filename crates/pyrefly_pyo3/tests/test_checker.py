# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pyrefly_api


def test_checker_resolves_imports_and_reports_errors(tmp_path):
    (tmp_path / "pyrefly.toml").write_text("")
    (tmp_path / "helper.py").write_text("def greet() -> int: ...\n")

    checker = pyrefly_api.Checker(project_root=str(tmp_path))

    diags = checker.check("from helper import greet\nx: str = greet()")
    assert [d.kind for d in diags] == ["bad-assignment"]

    # The warm Checker is reusable across calls.
    assert checker.check("from helper import greet\ny: int = greet()") == []
