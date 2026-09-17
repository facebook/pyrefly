# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Translates a change scorecard into per-section and overall quality scores.

Usage:
python3 scripts/score.py D1=YES D2=YES D3=NO D4=YES D5=YES \
    I1=YES I2=NO I3=YES V1=YES V2=YES \
    C1=YES C2=YES C3=YES R1=YES R2=YES R3=YES R4=YES

Sample Output:

Design: 4/5
Implementation: 2/3
Validation: 2/2
Code Quality: 3/3
Reviewability: 4/4

Quality Score: 82%
"""

import argparse
from fractions import Fraction

SECTIONS = (
    ("Design", 40, ("D1", "D2", "D3", "D4", "D5")),
    ("Implementation", 30, ("I1", "I2", "I3")),
    ("Validation", 10, ("V1", "V2")),
    ("Code Quality", 10, ("C1", "C2", "C3")),
    ("Reviewability", 10, ("R1", "R2", "R3", "R4")),
)
DIMENSIONS = {dimension for _, _, dimensions in SECTIONS for dimension in dimensions}
REQUIRED_DIMENSIONS = {"D1", "I1", "I2"}
VERDICTS = {"YES", "NO", "UNKNOWN", "N/A"}


def parse_verdicts(arguments: list[str]) -> dict[str, str]:
    """Parse and validate DIMENSION=VERDICT command-line arguments."""
    verdicts = {}
    for argument in arguments:
        dimension, separator, verdict = argument.partition("=")
        dimension = dimension.upper()
        verdict = verdict.upper()
        if not separator:
            raise ValueError(f"expected DIMENSION=VERDICT, got {argument!r}")
        if dimension not in DIMENSIONS:
            raise ValueError(f"unexpected dimension: {dimension}")
        if verdict not in VERDICTS:
            raise ValueError(
                f"invalid verdict {verdict!r} for {dimension}; expected one of "
                f"{', '.join(sorted(VERDICTS))}"
            )
        if dimension in verdicts:
            raise ValueError(f"duplicate verdict for {dimension}")
        verdicts[dimension] = verdict

    missing = DIMENSIONS - verdicts.keys()
    if missing:
        raise ValueError(f"missing verdict(s) for {', '.join(sorted(missing))}")
    if any(verdicts[dimension] == "N/A" for dimension in REQUIRED_DIMENSIONS):
        raise ValueError(f"{', '.join(sorted(REQUIRED_DIMENSIONS))} must never be N/A")
    return verdicts


def format_percentage(value: Fraction) -> str:
    """Format a percentage with at most two decimal places."""
    return f"{float(value):.2f}".rstrip("0").rstrip(".") + "%"


def render_scorecard(verdicts: dict[str, str]) -> str:
    """Render section scores and the weighted overall score."""
    lines = []
    overall_lower = Fraction()
    overall_upper = Fraction()
    for name, weight, dimensions in SECTIONS:
        applicable = [
            verdicts[dimension]
            for dimension in dimensions
            if verdicts[dimension] != "N/A"
        ]
        if not applicable:
            score = "N/A"
            lower = upper = Fraction(1)
        else:
            yes = applicable.count("YES")
            unknown = applicable.count("UNKNOWN")
            total = len(applicable)
            lower = Fraction(yes, total)
            upper = Fraction(yes + unknown, total)
            score = f"{yes}/{total}"
            if lower != upper:
                score += f" (lower bound) - {yes + unknown}/{total} (upper bound)"
        lines.append(f"{name}: {score}")
        overall_lower += weight * lower
        overall_upper += weight * upper

    quality = format_percentage(overall_lower)
    if overall_lower != overall_upper:
        quality += f"-{format_percentage(overall_upper)}"
    return "\n".join((*lines, "", f"Quality Score: {quality}"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute a weighted score from Pyrefly review verdicts."
    )
    parser.add_argument("verdicts", nargs="+", metavar="DIMENSION=VERDICT")
    arguments = parser.parse_args()
    try:
        verdicts = parse_verdicts(arguments.verdicts)
    except ValueError as error:
        parser.error(str(error))
    print(render_scorecard(verdicts))


if __name__ == "__main__":
    main()
