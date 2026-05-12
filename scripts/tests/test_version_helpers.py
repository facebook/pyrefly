# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import io
import unittest
from contextlib import redirect_stderr, redirect_stdout

from parameterized import parameterized
from pyrefly.scripts.version_helpers import (
    is_prerelease,
    main,
    parse_semver,
    to_marketplace,
)


class ParseSemverTest(unittest.TestCase):
    @parameterized.expand(
        [
            ("bare_zero", "0.0.0", (0, 0, 0, None)),
            ("bare_minor", "1.2.0", (1, 2, 0, None)),
            ("bare_patch", "1.2.3", (1, 2, 3, None)),
            ("bare_high", "100.200.300", (100, 200, 300, None)),
            ("dev_first", "1.2.0-dev.1", (1, 2, 0, 1)),
            ("dev_high", "1.2.0-dev.999", (1, 2, 0, 999)),
            ("dev_zero", "1.2.0-dev.0", (1, 2, 0, 0)),
        ]
    )
    def test_valid(
        self,
        _name: str,
        s: str,
        expected: tuple[int, int, int, int | None],
    ) -> None:
        self.assertEqual(parse_semver(s), expected)

    @parameterized.expand(
        [
            ("empty", ""),
            ("two_components", "1.2"),
            ("dev_no_number", "1.2.0-dev"),
            ("dev_dot_no_number", "1.2.0-dev."),
            ("rc_not_supported", "1.2.0-rc.1"),
            ("alpha_not_supported", "1.2.0-alpha.1"),
            ("v_prefix", "v1.2.0"),
            ("pep440_form", "1.2.0.dev4"),
            ("undotted_devN", "1.2.0-dev4"),
            ("trailing_garbage", "1.2.0-dev.4-foo"),
            ("leading_zero_major", "01.2.3"),
            ("leading_zero_minor", "1.02.3"),
            ("leading_zero_patch", "1.2.03"),
            ("leading_zero_dev", "1.2.0-dev.04"),
        ]
    )
    def test_invalid(self, _name: str, s: str) -> None:
        with self.assertRaises(ValueError):
            parse_semver(s)


class IsPrereleaseTest(unittest.TestCase):
    def test_bare_is_not_prerelease(self) -> None:
        self.assertFalse(is_prerelease("1.2.0"))
        self.assertFalse(is_prerelease("1.2.3"))
        self.assertFalse(is_prerelease("0.0.0"))

    def test_dev_is_prerelease(self) -> None:
        self.assertTrue(is_prerelease("1.2.0-dev.1"))
        self.assertTrue(is_prerelease("1.2.0-dev.0"))
        self.assertTrue(is_prerelease("1.2.0-dev.999"))

    def test_invalid_raises(self) -> None:
        with self.assertRaises(ValueError):
            is_prerelease("garbage")


class ToMarketplaceTest(unittest.TestCase):
    @parameterized.expand(
        [
            # Stable: passes through unchanged.
            ("bare_minor", "1.2.0", "1.2.0"),
            ("bare_patch", "1.2.5", "1.2.5"),
            ("bare_zero", "0.0.0", "0.0.0"),
            # Dev releases for non-zero minor: map to (M, m-1, 9000+N).
            ("dev_first", "1.2.0-dev.1", "1.1.9001"),
            ("dev_four", "1.2.0-dev.4", "1.1.9004"),
            ("dev_zero", "1.2.0-dev.0", "1.1.9000"),
            ("dev_high", "1.2.0-dev.999", "1.1.9999"),
            ("dev_minor_one", "1.1.0-dev.5", "1.0.9005"),
            ("dev_high_minor", "1.99.0-dev.3", "1.98.9003"),
            # Major boundary: M.0.0-dev.N -> (M-1).999.(9000+N).
            ("major_boundary_2", "2.0.0-dev.1", "1.999.9001"),
            ("major_boundary_5", "5.0.0-dev.10", "4.999.9010"),
            ("major_boundary_high_dev", "2.0.0-dev.999", "1.999.9999"),
        ]
    )
    def test_mapping(self, _name: str, semver: str, marketplace: str) -> None:
        self.assertEqual(to_marketplace(semver), marketplace)

    def test_dev_at_major_zero_boundary_raises(self) -> None:
        # 0.0.0-dev.N would map to (-1).999.(9000+N) which makes no sense.
        with self.assertRaises(ValueError):
            to_marketplace("0.0.0-dev.1")

    def test_dev_counter_overflow_raises(self) -> None:
        # dev counter must stay below 1000 to avoid colliding with the next
        # stable's patch range.
        with self.assertRaises(ValueError):
            to_marketplace("1.2.0-dev.1000")
        with self.assertRaises(ValueError):
            to_marketplace("1.2.0-dev.5000")

    def test_dev_with_nonzero_patch_raises(self) -> None:
        # Dev versions must have patch == 0 (the canonical form is
        # M.m.0-dev.N). Otherwise the mapping ignores patch and produces
        # colliding marketplace versions, e.g. both 1.2.0-dev.4 and
        # 1.2.3-dev.4 would map to 1.1.9004.
        with self.assertRaises(ValueError):
            to_marketplace("1.2.3-dev.4")
        with self.assertRaises(ValueError):
            to_marketplace("1.2.1-dev.0")

    def test_strict_ordering_against_next_stable(self) -> None:
        # Marketplace uses semver ordering. The mapped dev version must be
        # strictly less than the next stable's bare version so pre-release
        # users auto-upgrade to stable when it ships.
        # 1.1.9001 < 1.2.0 (1.1.x < 1.2.0 lexically by minor).
        # We don't compare strings here — we verify the tuple ordering:
        # marketplace_dev = (1, 1, 9001, None)
        marketplace_dev = parse_semver(to_marketplace("1.2.0-dev.1"))
        # next stable from this dev cycle would be 1.2.0
        next_stable = parse_semver("1.2.0")
        # Compare as (major, minor, patch) tuples; ignore dev for ordering
        # since marketplace_dev has dev=None.
        self.assertLess(
            (marketplace_dev[0], marketplace_dev[1], marketplace_dev[2]),
            (next_stable[0], next_stable[1], next_stable[2]),
        )

    def test_invalid_raises(self) -> None:
        with self.assertRaises(ValueError):
            to_marketplace("garbage")


class CliTest(unittest.TestCase):
    def _run(
        self,
        args: list[str],
    ) -> tuple[int, str, str]:
        out = io.StringIO()
        err = io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = main(args)
        return code, out.getvalue(), err.getvalue()

    def test_to_marketplace_bare(self) -> None:
        code, out, _ = self._run(["to-marketplace", "1.2.0"])
        self.assertEqual(code, 0)
        self.assertEqual(out.strip(), "1.2.0")

    def test_to_marketplace_dev(self) -> None:
        code, out, _ = self._run(["to-marketplace", "1.2.0-dev.4"])
        self.assertEqual(code, 0)
        self.assertEqual(out.strip(), "1.1.9004")

    def test_is_prerelease_true(self) -> None:
        code, out, _ = self._run(["is-prerelease", "1.2.0-dev.4"])
        self.assertEqual(code, 0)
        self.assertEqual(out.strip(), "true")

    def test_is_prerelease_false(self) -> None:
        code, out, _ = self._run(["is-prerelease", "1.2.0"])
        self.assertEqual(code, 1)
        self.assertEqual(out.strip(), "false")

    def test_invalid_input_returns_2(self) -> None:
        code, _, err = self._run(["to-marketplace", "garbage"])
        self.assertEqual(code, 2)
        self.assertIn("Not a valid pyrefly semver", err)
