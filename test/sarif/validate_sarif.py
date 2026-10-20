#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


"""
Validate Pyrefly's SARIF output against the official SARIF 2.1.0 JSON Schema.

Pyrefly builds its SARIF report from hand-written serde structs rather than a
generated object model, so nothing but this test proves the output is really
SARIF. It validates `diagnostics.expected.sarif`, the golden report that the
`--output-format sarif` end-to-end test in `test/errors.md` diffs against, so
the two stay in lockstep.

`sarif-2.1.0.json` is an unmodified copy of the schema published with the SARIF
2.1.0 specification, from
https://github.com/oasis-tcs/sarif-spec/blob/master/Schemata/sarif-schema-2.1.0.json
It is checked in rather than fetched because this test runs without network
access, and should only change if it is re-copied from upstream.

To regenerate the golden report after an intentional output change:

    pyrefly check --python-version 3.13.0 --output-format sarif \\
        --relative-to test/sarif test/sarif/diagnostics.py \\
      | jq '.runs[0].tool.driver.version = "0.0.0"' \\
      > test/sarif/diagnostics.expected.sarif

The version is pinned because the real one changes every release; the scrut
test applies the same substitution before diffing.

Requirements:
    pip install jsonschema
"""

import json
import sys
import unittest
from pathlib import Path

try:
    import jsonschema
except ImportError:
    print("Error: Required packages not installed.")
    print("Please run: pip install jsonschema")
    sys.exit(1)

SARIF_DIR = Path(__file__).parent


class TestSarifOutput(unittest.TestCase):
    def setUp(self) -> None:
        with open(SARIF_DIR / "sarif-2.1.0.json", "r") as f:
            self.schema: dict[str, object] = json.load(f)

    def test_schema_is_valid(self) -> None:
        jsonschema.Draft7Validator.check_schema(self.schema)

    def test_expected_report_is_valid_sarif(self) -> None:
        with open(SARIF_DIR / "diagnostics.expected.sarif", "r") as f:
            report = json.load(f)
        jsonschema.Draft7Validator(self.schema).validate(report)

    def test_schema_rejects_a_malformed_report(self) -> None:
        # Guards against the schema being loaded but not actually applied.
        with open(SARIF_DIR / "diagnostics.expected.sarif", "r") as f:
            report = json.load(f)
        report["runs"][0]["results"][0]["level"] = "catastrophe"
        with self.assertRaises(jsonschema.ValidationError):
            jsonschema.Draft7Validator(self.schema).validate(report)


if __name__ == "__main__":
    unittest.main()
