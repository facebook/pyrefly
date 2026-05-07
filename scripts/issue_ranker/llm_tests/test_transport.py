#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Integration tests for the shared LLM transport layer.

Requires ANTHROPIC_API_KEY or LLAMA_API_KEY in the environment.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from llm_transport import call_llm, call_llm_json, LLMError

_HAS_API_KEY = bool(
    os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("LLAMA_API_KEY")
)
_SKIP_REASON = "No API key set (need ANTHROPIC_API_KEY or LLAMA_API_KEY)"


@unittest.skipUnless(_HAS_API_KEY, _SKIP_REASON)
class TestCallLlm(unittest.TestCase):
    """Test call_llm returns a non-empty string from a trivial prompt."""

    def test_returns_text(self):
        text = call_llm(
            "You are a helpful assistant.",
            "Reply with exactly the word 'hello'.",
        )
        self.assertIsInstance(text, str)
        self.assertGreater(len(text.strip()), 0)


@unittest.skipUnless(_HAS_API_KEY, _SKIP_REASON)
class TestCallLlmJson(unittest.TestCase):
    """Test call_llm_json returns a parsed dict from a JSON prompt."""

    def test_returns_dict(self):
        result = call_llm_json(
            "You are a helpful assistant. Always respond with valid JSON.",
            'Return a JSON object with a key "status" set to "ok".',
        )
        self.assertIsInstance(result, dict)
        self.assertEqual(result.get("status"), "ok")


class TestCallLlmInvalidKey(unittest.TestCase):
    """Test that a bad API key raises LLMError."""

    def test_raises_on_bad_key(self):
        # Temporarily override env to force Anthropic with a bad key
        old_llama = os.environ.pop("LLAMA_API_KEY", None)
        old_classifier = os.environ.pop("CLASSIFIER_API_KEY", None)
        old_anthropic = os.environ.pop("ANTHROPIC_API_KEY", None)
        os.environ["ANTHROPIC_API_KEY"] = "sk-ant-INVALID"

        try:
            with self.assertRaises(LLMError):
                call_llm("system", "user")
        finally:
            # Restore original env
            os.environ.pop("ANTHROPIC_API_KEY", None)
            if old_llama:
                os.environ["LLAMA_API_KEY"] = old_llama
            if old_classifier:
                os.environ["CLASSIFIER_API_KEY"] = old_classifier
            if old_anthropic:
                os.environ["ANTHROPIC_API_KEY"] = old_anthropic


if __name__ == "__main__":
    unittest.main()
