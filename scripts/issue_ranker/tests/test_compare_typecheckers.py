#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for compare_typecheckers.py error parsing functions."""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from compare_typecheckers import (
    parse_error_count,
    parse_error_count_mypy,
    parse_full_errors_mypy,
    parse_full_errors_pyrefly,
    parse_full_errors_pyright,
)


class TestParseErrorCount(unittest.TestCase):
    """Test parse_error_count for pyrefly/pyright output."""

    def test_pyrefly_no_errors(self):
        self.assertEqual(parse_error_count("INFO No errors"), 0)

    def test_pyrefly_single_error(self):
        self.assertEqual(parse_error_count("INFO 1 error"), 1)

    def test_pyrefly_multiple_errors(self):
        self.assertEqual(parse_error_count("INFO 4 errors (11 suppressed)"), 4)

    def test_pyrefly_comma_separated(self):
        self.assertEqual(parse_error_count("INFO 3,418 errors"), 3418)

    def test_pyright_errors(self):
        self.assertEqual(parse_error_count("2 errors, 0 warnings, 0 information"), 2)

    def test_no_match(self):
        self.assertEqual(parse_error_count("some random output"), -1)

    def test_empty_string(self):
        self.assertEqual(parse_error_count(""), -1)

    def test_multiline_takes_last(self):
        output = "some preamble\nINFO 10 errors\nINFO 5 errors"
        self.assertEqual(parse_error_count(output), 5)


class TestParseErrorCountMypy(unittest.TestCase):
    """Test parse_error_count_mypy."""

    def test_no_issues(self):
        self.assertEqual(
            parse_error_count_mypy("Success: no issues found in 50 source files"),
            0,
        )

    def test_found_errors(self):
        self.assertEqual(
            parse_error_count_mypy(
                "Found 42 errors in 10 files (checked 50 source files)"
            ),
            42,
        )

    def test_single_error(self):
        self.assertEqual(
            parse_error_count_mypy("Found 1 error in 1 file (checked 1 source file)"),
            1,
        )

    def test_no_match(self):
        self.assertEqual(parse_error_count_mypy("random output"), -1)


class TestParseFullErrorsPyrefly(unittest.TestCase):
    """Test parse_full_errors_pyrefly JSON parsing."""

    def test_basic_errors(self):
        stdout = '{"errors": [{"line": 10, "column": 3, "stop_line": 10, "stop_column": 5, "path": "foo.py", "name": "bad-return", "description": "Bad return type", "severity": "error"}]}'
        errors = parse_full_errors_pyrefly(stdout)
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0]["file"], "foo.py")
        self.assertEqual(errors[0]["line"], 10)
        self.assertEqual(errors[0]["col"], 3)
        self.assertEqual(errors[0]["kind"], "bad-return")
        self.assertEqual(errors[0]["message"], "Bad return type")
        self.assertEqual(errors[0]["severity"], "error")

    def test_empty_errors(self):
        self.assertEqual(parse_full_errors_pyrefly('{"errors": []}'), [])

    def test_invalid_json(self):
        self.assertEqual(parse_full_errors_pyrefly("not json"), [])

    def test_multiple_errors(self):
        stdout = '{"errors": [{"line": 1, "column": 1, "path": "a.py", "name": "bad-override", "description": "msg1", "severity": "error"}, {"line": 5, "column": 2, "path": "b.py", "name": "unknown-type", "description": "msg2", "severity": "error"}]}'
        errors = parse_full_errors_pyrefly(stdout)
        self.assertEqual(len(errors), 2)
        self.assertEqual(errors[0]["kind"], "bad-override")
        self.assertEqual(errors[1]["kind"], "unknown-type")

    def test_missing_fields_use_defaults(self):
        stdout = '{"errors": [{}]}'
        errors = parse_full_errors_pyrefly(stdout)
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0]["file"], "")
        self.assertEqual(errors[0]["line"], 0)
        self.assertEqual(errors[0]["col"], 0)
        self.assertEqual(errors[0]["kind"], "")
        self.assertEqual(errors[0]["message"], "")
        self.assertEqual(errors[0]["severity"], "error")


class TestParseFullErrorsPyright(unittest.TestCase):
    """Test parse_full_errors_pyright JSON parsing."""

    def test_basic_error(self):
        import tempfile

        repo_dir = tempfile.mkdtemp()
        # Use realpath to match what the parser does (macOS: /tmp -> /private/tmp)
        real_repo = os.path.realpath(repo_dir)
        file_path = os.path.join(real_repo, "foo.py")
        stdout = f'{{"generalDiagnostics": [{{"file": "{file_path}", "severity": "error", "message": "Type mismatch", "range": {{"start": {{"line": 9, "character": 4}}}}, "rule": "reportReturnType"}}]}}'
        errors = parse_full_errors_pyright(stdout, repo_dir)
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0]["file"], "foo.py")
        # Pyright uses 0-based; we normalize to 1-based
        self.assertEqual(errors[0]["line"], 10)
        self.assertEqual(errors[0]["col"], 5)
        self.assertEqual(errors[0]["kind"], "reportReturnType")
        os.rmdir(repo_dir)

    def test_empty_diagnostics(self):
        self.assertEqual(
            parse_full_errors_pyright('{"generalDiagnostics": []}', "/tmp"),
            [],
        )

    def test_invalid_json(self):
        self.assertEqual(parse_full_errors_pyright("not json", "/tmp"), [])

    def test_path_outside_repo(self):
        """Paths not under repo_dir should be left as-is."""
        stdout = '{"generalDiagnostics": [{"file": "/other/path.py", "severity": "error", "message": "msg", "range": {"start": {"line": 0, "character": 0}}, "rule": "rule"}]}'
        errors = parse_full_errors_pyright(stdout, "/tmp/repo")
        self.assertEqual(errors[0]["file"], "/other/path.py")


class TestParseFullErrorsMypy(unittest.TestCase):
    """Test parse_full_errors_mypy text output parsing."""

    def test_basic_error(self):
        output = "foo.py:10: error: Incompatible return type  [return-value]"
        errors = parse_full_errors_mypy(output)
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0]["file"], "foo.py")
        self.assertEqual(errors[0]["line"], 10)
        self.assertEqual(errors[0]["col"], 0)
        self.assertEqual(errors[0]["kind"], "return-value")
        self.assertEqual(errors[0]["message"], "Incompatible return type")
        self.assertEqual(errors[0]["severity"], "error")

    def test_error_with_column(self):
        output = "bar.py:20:5: error: Name 'x' is not defined  [name-defined]"
        errors = parse_full_errors_mypy(output)
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0]["col"], 5)

    def test_warning(self):
        output = "baz.py:3: warning: Something  [misc]"
        errors = parse_full_errors_mypy(output)
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0]["severity"], "warning")

    def test_skips_notes(self):
        output = "foo.py:10: note: Some note here"
        errors = parse_full_errors_mypy(output)
        self.assertEqual(len(errors), 0)

    def test_multiple_errors(self):
        output = (
            "a.py:1: error: Msg1  [err1]\n"
            "b.py:2: error: Msg2  [err2]\n"
            "c.py:3: warning: Msg3  [warn1]\n"
        )
        errors = parse_full_errors_mypy(output)
        self.assertEqual(len(errors), 3)

    def test_empty_output(self):
        self.assertEqual(parse_full_errors_mypy(""), [])

    def test_error_without_code(self):
        output = "foo.py:10: error: Some message without a code"
        errors = parse_full_errors_mypy(output)
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0]["kind"], "")
        self.assertEqual(errors[0]["message"], "Some message without a code")


if __name__ == "__main__":
    unittest.main()
