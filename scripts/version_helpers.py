# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Pyrefly version-format helpers shared by the GitHub release workflows.

`version.bzl` records canonical semver (e.g. "1.2.3" or "1.2.0-dev.4").
Maturin handles the PEP 440 conversion for PyPI automatically. The VS Code
Marketplace needs a different mapping since it doesn't accept semver
pre-release identifiers:

  - bare semver (M.m.p):         passes through unchanged
  - M.m.0-dev.N with m >= 1:     M.(m-1).(9000 + N)
  - M.0.0-dev.N with M > 1:      (M-1).999.(9000 + N)  (major boundary)

Used as a CLI from the release workflows:

    python3 scripts/version_helpers.py to-marketplace 1.2.0-dev.4
    python3 scripts/version_helpers.py is-prerelease  1.2.0-dev.4
"""

# pyre-strict

from __future__ import annotations

import argparse
import re
import sys


_SEMVER_RE: re.Pattern[str] = re.compile(
    r"^(\d+)\.(\d+)\.(\d+)(?:-dev\.(\d+))?$",
)


def parse_semver(version: str) -> tuple[int, int, int, int | None]:
    """Parse a pyrefly semver string into (major, minor, patch, dev_n).

    dev_n is None for bare semver. Raises ValueError on malformed input.
    """
    match = _SEMVER_RE.match(version)
    if not match:
        raise ValueError(f"Not a valid pyrefly semver version: {version!r}")
    for group in (match.group(1), match.group(2), match.group(3)):
        if len(group) > 1 and group[0] == "0":
            raise ValueError(
                f"Leading zero in version component {group!r} is not allowed",
            )
    dev_str = match.group(4)
    if dev_str is not None and len(dev_str) > 1 and dev_str[0] == "0":
        raise ValueError(
            f"Leading zero in dev counter {dev_str!r} is not allowed",
        )
    return (
        int(match.group(1)),
        int(match.group(2)),
        int(match.group(3)),
        int(dev_str) if dev_str is not None else None,
    )


def is_prerelease(version: str) -> bool:
    """Return True if the version is a pre-release (contains -dev.N)."""
    _, _, _, dev_n = parse_semver(version)
    return dev_n is not None


def to_marketplace(version: str) -> str:
    """Convert a pyrefly semver to its marketplace M.m.p form.

    Stable (bare semver) passes through unchanged. Dev releases for a
    non-zero minor map to `M.(m-1).(9000 + N)`. Dev releases at a major
    boundary (M.0.0-dev.N with M > 1) map to `(M-1).999.(9000 + N)`.

    >>> to_marketplace("1.2.0")
    '1.2.0'
    >>> to_marketplace("1.2.1")
    '1.2.1'
    >>> to_marketplace("1.2.0-dev.4")
    '1.1.9004'
    >>> to_marketplace("2.0.0-dev.1")
    '1.999.9001'
    """
    major, minor, patch, dev_n = parse_semver(version)
    if dev_n is None:
        return f"{major}.{minor}.{patch}"
    if patch != 0:
        # The dev form is defined as M.m.0-dev.N. Mapping ignores patch, so
        # without this check 1.2.3-dev.4 and 1.2.0-dev.4 would collide on
        # 1.1.9004.
        raise ValueError(
            f"Dev version {version!r} has nonzero patch; expected M.m.0-dev.N.",
        )
    if dev_n >= 1000:
        raise ValueError(
            f"dev counter {dev_n} is >= 1000 and would overflow the "
            "marketplace mapping. Cut a stable/patch release before the "
            "dev counter reaches 1000.",
        )
    market_patch = 9000 + dev_n
    if minor == 0:
        if major < 1:
            raise ValueError(
                f"Cannot map major-boundary dev version {version!r}: "
                "major must be >= 1.",
            )
        return f"{major - 1}.999.{market_patch}"
    return f"{major}.{minor - 1}.{market_patch}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_market = sub.add_parser(
        "to-marketplace",
        help="Print the M.m.p form for VS Code Marketplace and Open VSX.",
    )
    p_market.add_argument("version")

    p_pre = sub.add_parser(
        "is-prerelease",
        help=(
            "Print 'true' or 'false'. Exits 0 if the version is a "
            "pre-release, 1 otherwise. (Both shapes available so callers "
            "can use whichever fits.)"
        ),
    )
    p_pre.add_argument("version")

    args = parser.parse_args(argv)

    try:
        if args.command == "to-marketplace":
            print(to_marketplace(args.version))
            return 0
        if args.command == "is-prerelease":
            result = is_prerelease(args.version)
            print("true" if result else "false")
            return 0 if result else 1
    except ValueError as e:
        print(str(e), file=sys.stderr)
        return 2

    raise AssertionError(f"Unhandled command: {args.command!r}")


if __name__ == "__main__":
    sys.exit(main())
