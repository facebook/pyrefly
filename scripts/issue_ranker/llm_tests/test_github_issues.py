#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Integration tests for github_issues.py.

Requires GITHUB_TOKEN in the environment.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from github_issues import fetch_issues

_HAS_GITHUB_TOKEN = bool(os.environ.get("GITHUB_TOKEN"))
_SKIP_REASON = "No GITHUB_TOKEN set"


@unittest.skipUnless(_HAS_GITHUB_TOKEN, _SKIP_REASON)
class TestFetchIssues(unittest.TestCase):
    """Test that fetch_issues returns well-structured issue data."""

    def test_fetch_two_issues(self):
        issues = fetch_issues(limit=2)
        self.assertIsInstance(issues, list)
        self.assertGreater(len(issues), 0)
        self.assertLessEqual(len(issues), 2)

        # Verify structure of the first issue
        issue = issues[0]
        self.assertIn("number", issue)
        self.assertIsInstance(issue["number"], int)
        self.assertIn("title", issue)
        self.assertIsInstance(issue["title"], str)
        self.assertIn("body", issue)
        self.assertIn("labels", issue)
        self.assertIsInstance(issue["labels"], list)
        self.assertIn("reactions_count", issue)
        self.assertIn("comments_count", issue)
        self.assertIn("url", issue)
        self.assertIn("milestone", issue)

    def test_fetch_with_label_filter(self):
        issues = fetch_issues(labels=["bug"], limit=2)
        self.assertIsInstance(issues, list)
        # All returned issues should have the "bug" label
        for issue in issues:
            self.assertIn("bug", issue.get("labels", []))


if __name__ == "__main__":
    unittest.main()
