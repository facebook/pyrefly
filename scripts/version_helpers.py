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
    python3 scripts/version_helpers.py to-pypi        1.2.0-dev.4
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
_PYPI_VERSION_RE: re.Pattern[str] = re.compile(
    r"^(\d+)\.(\d+)\.(\d+)(?:\.dev(\d+))?$",
)


def _check_no_leading_zero(component: str, name: str) -> None:
    if len(component) > 1 and component[0] == "0":
        raise ValueError(f"Leading zero in {name} {component!r} is not allowed")


def _parse_version_match(match: re.Match[str]) -> tuple[int, int, int, int | None]:
    for group in (match.group(1), match.group(2), match.group(3)):
        _check_no_leading_zero(group, "version component")
    dev_str = match.group(4)
    if dev_str is not None:
        _check_no_leading_zero(dev_str, "dev counter")
    return (
        int(match.group(1)),
        int(match.group(2)),
        int(match.group(3)),
        int(dev_str) if dev_str is not None else None,
    )


def parse_semver(version: str) -> tuple[int, int, int, int | None]:
    """Parse a pyrefly semver string into (major, minor, patch, dev_n).

    dev_n is None for bare semver. Raises ValueError on malformed input.
    """
    match = _SEMVER_RE.match(version)
    if not match:
        raise ValueError(f"Not a valid pyrefly semver version: {version!r}")
    return _parse_version_match(match)


def _parse_pypi_version(version: str) -> tuple[int, int, int, int | None]:
    """Parse a PyPI/PEP 440 version into (major, minor, patch, dev_n)."""
    match = _PYPI_VERSION_RE.match(version)
    if not match:
        raise ValueError(f"Not a valid PyPI version: {version!r}")
    return _parse_version_match(match)


def format_semver(version: tuple[int, int, int, int | None]) -> str:
    """Render a parsed version tuple as canonical semver (inverse of parse_semver)."""
    major, minor, patch, dev_n = version
    s = f"{major}.{minor}.{patch}"
    if dev_n is not None:
        s += f"-dev.{dev_n}"
    return s


def previous_minor(version: str) -> str:
    """Bare release tag the changelog for a stable minor/major cut diffs *from*.

    A minor release's notes should span the whole minor -- everything since
    the previous stable minor -- not just the last dev snapshot, so they diff
    from `M.(m-1).0`. Tags are bare semver with no `v` prefix (see
    publish_to_pypi.yml).

    Raises ValueError unless `version` is a stable `M.m.0` with m > 0. A major
    bump `M.0.0` has no previous minor in its own major line, so the caller
    must supply an explicit from-ref.
    """
    major, minor, patch, dev_n = parse_semver(version)
    if dev_n is not None or patch != 0:
        raise ValueError(
            f"previous_minor expects a stable M.m.0 version, got {version!r}"
        )
    if minor == 0:
        raise ValueError(
            f"{version!r} is a major release with no previous minor in its "
            "major line; supply an explicit from-ref"
        )
    return f"{major}.{minor - 1}.0"


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


def to_pypi(version: str) -> str:
    """Convert a pyrefly canonical version to its PyPI/PEP 440 form.

    Stable versions pass through unchanged. Canonical dev versions use
    `-dev.N`; PyPI expects `.devN`. Already-normalized PyPI dev versions also
    pass through so local packaging commands can be idempotent.

    >>> to_pypi("1.2.3")
    '1.2.3'
    >>> to_pypi("1.2.0-dev.4")
    '1.2.0.dev4'
    >>> to_pypi("1.2.0.dev4")
    '1.2.0.dev4'
    """
    try:
        major, minor, patch, dev_n = parse_semver(version)
    except ValueError:
        try:
            major, minor, patch, dev_n = _parse_pypi_version(version)
        except ValueError as pypi_error:
            raise ValueError(
                f"Not a valid pyrefly or PyPI version: {version!r}"
            ) from pypi_error
    if dev_n is None:
        return f"{major}.{minor}.{patch}"
    return f"{major}.{minor}.{patch}.dev{dev_n}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_market = sub.add_parser(
        "to-marketplace",
        help="Print the M.m.p form for VS Code Marketplace and Open VSX.",
    )
    p_market.add_argument("version")

    p_pypi = sub.add_parser(
        "to-pypi",
        help="Print the PEP 440 form for PyPI package metadata.",
    )
    p_pypi.add_argument("version")

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
        if args.command == "to-pypi":
            print(to_pypi(args.version))
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
