#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Integration tests for status_classifier LLM classification.

Requires ANTHROPIC_API_KEY or LLAMA_API_KEY in the environment.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from status_classifier import classify_status

_HAS_API_KEY = bool(
    os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("LLAMA_API_KEY")
)
_SKIP_REASON = "No API key set (need ANTHROPIC_API_KEY or LLAMA_API_KEY)"

# Valid status values returned by classify_status.
_VALID_STATUSES = {
    "false_positive",
    "false_negative",
    "already_fixed",
    "feature_request",
    "confirmed_bug",
    "unclear",
}


@unittest.skipUnless(_HAS_API_KEY, _SKIP_REASON)
class TestClassifyFalsePositive(unittest.TestCase):
    """Test classification when only pyrefly reports errors."""

    def test_pyrefly_only_errors(self):
        checker_results = {
            "snippet_count": 1,
            "pyrefly": [
                {"kind": "bad-return", "line": 5, "message": "Expected str, got int"}
            ],
            "pyright": [],
            "mypy": [],
        }
        status = classify_status(
            checker_results,
            issue_title="Pyrefly reports bad-return on valid code",
            issue_body="This code is valid but pyrefly flags it.",
        )
        self.assertIn(status, _VALID_STATUSES)


@unittest.skipUnless(_HAS_API_KEY, _SKIP_REASON)
class TestClassifyAlreadyFixed(unittest.TestCase):
    """Test classification when no checker finds errors."""

    def test_no_errors_anywhere(self):
        checker_results = {
            "snippet_count": 1,
            "pyrefly": [],
            "pyright": [],
            "mypy": [],
        }
        status = classify_status(
            checker_results,
            issue_title="Old error no longer reproduces",
            issue_body="This used to fail but seems fixed now.",
        )
        self.assertIn(status, _VALID_STATUSES)
        # With no errors from any checker, the fast path returns "already_fixed"
        self.assertEqual(status, "already_fixed")


if __name__ == "__main__":
    unittest.main()
