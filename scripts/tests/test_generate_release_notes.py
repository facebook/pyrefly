# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import unittest

from pyrefly.scripts.generate_release_notes import (
    batch_lines,
    condense_commit_log,
    estimate_tokens,
)


class BatchLinesTest(unittest.TestCase):
    def test_keeps_lines_in_order_within_budget(self) -> None:
        lines = [f"- {i:03d} " + "x" * 300 for i in range(10)]
        budget = 3 * (estimate_tokens(lines[0]) + 1)
        batches = batch_lines(lines, budget)
        self.assertEqual([len(b) for b in batches], [3, 3, 3, 1])
        self.assertEqual([line for b in batches for line in b], lines)

    def test_oversized_line_is_truncated_into_its_own_batch(self) -> None:
        lines = ["- small", "- " + "x" * 3000, "- small2"]
        batches = batch_lines(lines, 100)
        self.assertEqual(batches[0], ["- small"])
        self.assertEqual(len(batches[1]), 1)
        self.assertTrue(lines[1].startswith(batches[1][0]))
        self.assertLessEqual(estimate_tokens(batches[1][0]), 100)
        self.assertEqual(batches[2], ["- small2"])

    def test_empty(self) -> None:
        self.assertEqual(batch_lines([], 100), [])


class CondenseCommitLogTest(unittest.TestCase):
    def test_summarizes_every_batch_in_order(self) -> None:
        lines = [f"- {i:03d} " + "x" * 3000 for i in range(6)]
        prompts: list[str] = []

        def fake_generate(user_prompt: str, system_prompt: str, api_key: str) -> str:
            prompts.append(user_prompt)
            return f"summary {len(prompts)}"

        budget = 5000
        log = condense_commit_log(lines, budget, fake_generate, "")
        self.assertGreater(len(prompts), 1)
        for prompt in prompts:
            self.assertLessEqual(estimate_tokens(prompt), budget)
        self.assertEqual("".join(prompts).count("- 0"), len(lines))
        self.assertIn(f"condensed in {len(prompts)} batches", log)
        self.assertLess(log.index("summary 1"), log.index(f"summary {len(prompts)}"))
