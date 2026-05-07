#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Integration tests for code_extractor LLM functions.

Requires ANTHROPIC_API_KEY or LLAMA_API_KEY in the environment.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from code_extractor import extract_code_blocks, repair_snippet

_HAS_API_KEY = bool(
    os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("LLAMA_API_KEY")
)
_SKIP_REASON = "No API key set (need ANTHROPIC_API_KEY or LLAMA_API_KEY)"


@unittest.skipUnless(_HAS_API_KEY, _SKIP_REASON)
class TestExtractFromIssueBody(unittest.TestCase):
    """Test that code extraction works on a real issue body with LLM fallback."""

    def test_extract_from_fenced_block(self):
        """Fenced Python blocks should be extracted without needing the LLM."""
        body = (
            "Pyrefly reports an error on this code:\n"
            "```python\n"
            "def foo(xs: list[int]) -> list[str]:\n"
            "    return [str(x) for x in xs]\n"
            "```\n"
            "Expected no errors."
        )
        blocks = extract_code_blocks(body, use_llm=True)
        self.assertGreater(len(blocks), 0)
        self.assertIn("def foo", blocks[0])

    def test_llm_fallback_on_unfenced_code(self):
        """When no code blocks are found, use_llm=True falls back to LLM."""
        body = (
            "I have a function that takes a list of ints and returns "
            "their string representations, but pyrefly says the return type "
            "is wrong. The function uses a list comprehension with str()."
        )
        blocks = extract_code_blocks(body, use_llm=True)
        # The LLM should attempt to generate a snippet — may or may not succeed
        # so we just check it doesn't crash
        self.assertIsInstance(blocks, list)


@unittest.skipUnless(_HAS_API_KEY, _SKIP_REASON)
class TestRepairBrokenSnippet(unittest.TestCase):
    """Test that snippet repair fixes a broken code snippet."""

    def test_repair_missing_import(self):
        broken = "x: List[int] = [1, 2, 3]"
        repaired = repair_snippet(
            broken,
            issue_title="List type annotation error",
            issue_body="Pyrefly doesn't handle List without import.",
        )
        # The repaired version should be a non-empty string
        self.assertIsInstance(repaired, str)
        self.assertGreater(len(repaired.strip()), 0)


if __name__ == "__main__":
    unittest.main()
