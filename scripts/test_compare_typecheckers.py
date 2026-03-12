#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for compare_typecheckers.py."""

import json
import os
import tempfile
import unittest

from compare_typecheckers import (
    parse_error_count,
    parse_full_errors_pyrefly,
    parse_full_errors_pyright,
    write_json_output,
)
from projects import Project


class TestParseErrorCount(unittest.TestCase):
    """Tests for the existing parse_error_count function."""

    def test_pyrefly_errors(self) -> None:
        self.assertEqual(parse_error_count("INFO 4 errors (11 suppressed)"), 4)

    def test_pyrefly_many_errors(self) -> None:
        self.assertEqual(parse_error_count("INFO 3,418 errors"), 3418)

    def test_pyrefly_no_errors(self) -> None:
        self.assertEqual(parse_error_count("INFO No errors"), 0)

    def test_pyright_errors(self) -> None:
        self.assertEqual(parse_error_count("2 errors, 0 warnings, 0 information"), 2)

    def test_no_match(self) -> None:
        self.assertEqual(parse_error_count("some random output"), -1)


class TestParseFullErrorsPyrefly(unittest.TestCase):
    """Tests for parse_full_errors_pyrefly."""

    SAMPLE_OUTPUT = json.dumps(
        {
            "errors": [
                {
                    "line": 10,
                    "column": 5,
                    "stop_line": 10,
                    "stop_column": 15,
                    "path": "mypy/nodes.py",
                    "code": -2,
                    "name": "bad-return",
                    "description": "Expected `str`, got `int`",
                    "concise_description": "bad return type",
                    "severity": "error",
                },
                {
                    "line": 20,
                    "column": 1,
                    "stop_line": 20,
                    "stop_column": 10,
                    "path": "mypy/util.py",
                    "code": -2,
                    "name": "incompatible-parameter",
                    "description": "Argument of type `float` is not assignable to `int`",
                    "concise_description": "bad param",
                    "severity": "error",
                },
            ]
        }
    )

    def test_parses_errors(self) -> None:
        errors = parse_full_errors_pyrefly(self.SAMPLE_OUTPUT)
        self.assertEqual(len(errors), 2)
        self.assertEqual(errors[0]["file"], "mypy/nodes.py")
        self.assertEqual(errors[0]["line"], 10)
        self.assertEqual(errors[0]["col"], 5)
        self.assertEqual(errors[0]["kind"], "bad-return")
        self.assertEqual(errors[0]["message"], "Expected `str`, got `int`")
        self.assertEqual(errors[0]["severity"], "error")

    def test_second_error(self) -> None:
        errors = parse_full_errors_pyrefly(self.SAMPLE_OUTPUT)
        self.assertEqual(errors[1]["file"], "mypy/util.py")
        self.assertEqual(errors[1]["line"], 20)
        self.assertEqual(errors[1]["kind"], "incompatible-parameter")

    def test_empty_errors(self) -> None:
        errors = parse_full_errors_pyrefly(json.dumps({"errors": []}))
        self.assertEqual(errors, [])

    def test_invalid_json(self) -> None:
        errors = parse_full_errors_pyrefly("not json at all")
        self.assertEqual(errors, [])

    def test_missing_fields_use_defaults(self) -> None:
        output = json.dumps({"errors": [{"path": "foo.py"}]})
        errors = parse_full_errors_pyrefly(output)
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0]["file"], "foo.py")
        self.assertEqual(errors[0]["line"], 0)
        self.assertEqual(errors[0]["col"], 0)
        self.assertEqual(errors[0]["kind"], "")
        self.assertEqual(errors[0]["severity"], "error")

    def test_mixed_severities(self) -> None:
        output = json.dumps(
            {
                "errors": [
                    {"path": "a.py", "name": "x", "severity": "error"},
                    {"path": "b.py", "name": "y", "severity": "warn"},
                    {"path": "c.py", "name": "z", "severity": "info"},
                ]
            }
        )
        errors = parse_full_errors_pyrefly(output)
        self.assertEqual(len(errors), 3)
        error_only = [e for e in errors if e["severity"] == "error"]
        self.assertEqual(len(error_only), 1)


class TestParseFullErrorsPyright(unittest.TestCase):
    """Tests for parse_full_errors_pyright."""

    # Use realpath so the test paths match what the function computes
    # (on macOS, /tmp resolves to /private/tmp)
    REPO_DIR = "/tmp/repos/mypy"
    REPO_DIR_REAL = os.path.realpath(REPO_DIR)

    SAMPLE_OUTPUT = json.dumps(
        {
            "version": "1.1.390",
            "time": "1709746800000",
            "generalDiagnostics": [
                {
                    "file": os.path.realpath("/tmp/repos/mypy") + "/mypy/nodes.py",
                    "severity": "error",
                    "message": 'Expression of type "int" is incompatible with declared type "str"',
                    "range": {
                        "start": {"line": 9, "character": 4},
                        "end": {"line": 9, "character": 15},
                    },
                    "rule": "reportAssignmentType",
                },
                {
                    "file": os.path.realpath("/tmp/repos/mypy") + "/mypy/util.py",
                    "severity": "warning",
                    "message": 'Import "foo" could not be resolved',
                    "range": {
                        "start": {"line": 0, "character": 5},
                        "end": {"line": 0, "character": 8},
                    },
                    "rule": "reportMissingImports",
                },
            ],
            "summary": {
                "filesAnalyzed": 42,
                "errorCount": 1,
                "warningCount": 1,
                "informationCount": 0,
                "timeInSec": 2.5,
            },
        }
    )

    def test_parses_diagnostics(self) -> None:
        errors = parse_full_errors_pyright(self.SAMPLE_OUTPUT, self.REPO_DIR)
        self.assertEqual(len(errors), 2)

    def test_normalizes_to_1_based(self) -> None:
        """Pyright uses 0-based lines/cols; we normalize to 1-based."""
        errors = parse_full_errors_pyright(self.SAMPLE_OUTPUT, self.REPO_DIR)
        self.assertEqual(errors[0]["line"], 10)  # 9 + 1
        self.assertEqual(errors[0]["col"], 5)  # 4 + 1

    def test_makes_paths_relative(self) -> None:
        errors = parse_full_errors_pyright(self.SAMPLE_OUTPUT, self.REPO_DIR)
        self.assertEqual(errors[0]["file"], "mypy/nodes.py")
        self.assertEqual(errors[1]["file"], "mypy/util.py")

    def test_preserves_rule_and_message(self) -> None:
        errors = parse_full_errors_pyright(self.SAMPLE_OUTPUT, self.REPO_DIR)
        self.assertEqual(errors[0]["kind"], "reportAssignmentType")
        self.assertIn("incompatible", errors[0]["message"])

    def test_preserves_severity(self) -> None:
        errors = parse_full_errors_pyright(self.SAMPLE_OUTPUT, self.REPO_DIR)
        self.assertEqual(errors[0]["severity"], "error")
        self.assertEqual(errors[1]["severity"], "warning")

    def test_empty_diagnostics(self) -> None:
        output = json.dumps({"generalDiagnostics": [], "summary": {}})
        errors = parse_full_errors_pyright(output, self.REPO_DIR)
        self.assertEqual(errors, [])

    def test_invalid_json(self) -> None:
        errors = parse_full_errors_pyright("not json", self.REPO_DIR)
        self.assertEqual(errors, [])

    def test_no_rule_field(self) -> None:
        """Some pyright diagnostics don't have a rule field."""
        output = json.dumps(
            {
                "generalDiagnostics": [
                    {
                        "file": self.REPO_DIR_REAL + "/foo.py",
                        "severity": "error",
                        "message": "Syntax error",
                        "range": {"start": {"line": 0, "character": 0}},
                    }
                ]
            }
        )
        errors = parse_full_errors_pyright(output, self.REPO_DIR)
        self.assertEqual(errors[0]["kind"], "")

    def test_path_not_under_repo_dir(self) -> None:
        """Paths outside repo_dir should be preserved as-is."""
        output = json.dumps(
            {
                "generalDiagnostics": [
                    {
                        "file": "/other/path/foo.py",
                        "severity": "error",
                        "message": "test",
                        "range": {"start": {"line": 0, "character": 0}},
                    }
                ]
            }
        )
        errors = parse_full_errors_pyright(output, self.REPO_DIR)
        self.assertEqual(errors[0]["file"], "/other/path/foo.py")


class TestWriteJsonOutput(unittest.TestCase):
    """Tests for write_json_output."""

    def _make_project(self, name: str, url: str) -> Project:
        return Project(
            location=url,
            name_override=name,
            mypy_cmd="",
            pyrefly_cmd="{pyrefly} .",
            pyright_cmd="{pyright}",
        )

    def test_writes_valid_json(self) -> None:
        projects = [self._make_project("test-proj", "https://github.com/test/proj")]
        results = [
            {
                "project": "test-proj",
                "pyrefly_errors": 5,
                "pyright_errors": 3,
                "pyrefly_time": 1.5,
                "pyright_time": 2.0,
                "pyrefly_error_list": [
                    {
                        "file": "a.py",
                        "line": 1,
                        "col": 1,
                        "kind": "bad-return",
                        "message": "test",
                        "severity": "error",
                    }
                ],
                "pyright_error_list": [
                    {
                        "file": "a.py",
                        "line": 1,
                        "col": 1,
                        "kind": "reportReturnType",
                        "message": "test",
                        "severity": "error",
                    }
                ],
                "url": "https://github.com/test/proj",
            }
        ]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            path = f.name
        try:
            write_json_output(results, projects, path)
            with open(path) as f:
                data = json.load(f)
            self.assertIn("timestamp", data)
            self.assertEqual(len(data["projects"]), 1)
            proj = data["projects"][0]
            self.assertEqual(proj["name"], "test-proj")
            self.assertEqual(proj["url"], "https://github.com/test/proj")
            self.assertEqual(proj["pyrefly"]["error_count"], 5)
            self.assertEqual(len(proj["pyrefly"]["errors"]), 1)
            self.assertEqual(proj["pyright"]["error_count"], 3)
        finally:
            os.unlink(path)

    def test_multiple_projects(self) -> None:
        projects = [
            self._make_project("proj-a", "https://github.com/a"),
            self._make_project("proj-b", "https://github.com/b"),
        ]
        results = [
            {
                "project": "proj-a",
                "pyrefly_errors": 1,
                "pyright_errors": 2,
                "pyrefly_time": 1.0,
                "pyright_time": 1.0,
                "pyrefly_error_list": [],
                "pyright_error_list": [],
            },
            {
                "project": "proj-b",
                "pyrefly_errors": 0,
                "pyright_errors": 0,
                "pyrefly_time": 0.5,
                "pyright_time": 0.5,
                "pyrefly_error_list": [],
                "pyright_error_list": [],
            },
        ]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            path = f.name
        try:
            write_json_output(results, projects, path)
            with open(path) as f:
                data = json.load(f)
            self.assertEqual(len(data["projects"]), 2)
            self.assertEqual(data["projects"][0]["name"], "proj-a")
            self.assertEqual(data["projects"][1]["name"], "proj-b")
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main()
