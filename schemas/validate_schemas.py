#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""
Schema validation tests using unittest.


This script validates the test configuration files against the JSON schemas
to ensure the schemas are correctly structured and comprehensive.
It also runs negative tests to ensure the schemas correctly reject invalid configs.

Requirements:
    pip install jsonschema toml
"""

import json
import sys
import unittest
from pathlib import Path

try:
    import jsonschema
    import toml
except ImportError:
    print("Error: Required packages not installed.")
    print("Please run: pip install jsonschema toml")
    sys.exit(1)

SCHEMAS_DIR = Path(__file__).parent


def _make_validator(schema_file: Path):
    """Create a JSON schema validator with $ref support.

    Uses RefResolver with a store mapping to handle cross-file $ref resolution.
    """
    with open(schema_file, "r") as f:
        schema = json.load(f)

    # Build a store mapping URIs to schema dicts for $ref resolution.
    store = {}

    # Register the main pyrefly.json schema so $ref from the pyproject schema works.
    main_schema_file = schema_file.parent / "pyrefly.json"
    if main_schema_file.exists() and main_schema_file != schema_file:
        with open(main_schema_file, "r") as f:
            main_schema = json.load(f)
        if "$id" in main_schema:
            store[main_schema["$id"]] = main_schema
        # Also register by relative name so "$ref": "pyrefly.json" resolves.
        store["pyrefly.json"] = main_schema

    resolver = jsonschema.RefResolver.from_schema(schema, store=store)
    validator_cls = jsonschema.validators.validator_for(schema)
    return validator_cls(schema, resolver=resolver)


class TestPositiveValidation(unittest.TestCase):
    """Positive tests: valid config files should pass schema validation."""

    def test_pyrefly_toml(self) -> None:
        toml_file = SCHEMAS_DIR / "test-pyrefly.toml"
        schema_file = SCHEMAS_DIR / "pyrefly.json"
        with open(toml_file, "r") as f:
            config = toml.load(f)
        validator = _make_validator(schema_file)
        validator.validate(config)

    def test_pyproject_toml(self) -> None:
        toml_file = SCHEMAS_DIR / "test-pyproject.toml"
        schema_file = SCHEMAS_DIR / "pyproject-tool-pyrefly.json"
        with open(toml_file, "r") as f:
            config = toml.load(f)
        config_to_validate = config["tool"]["pyrefly"]
        validator = _make_validator(schema_file)
        validator.validate(config_to_validate)


# Each entry is (test_name_suffix, toml_string). All should be rejected.
NEGATIVE_TEST_CASES: list[tuple[str, str]] = [
    # --- Wrong types ---
    (
        "project_includes_as_string",
        'project-includes = "**/*.py"',
    ),
    (
        "python_version_as_number",
        "python-version = 3.13",
    ),
    (
        "skip_interpreter_query_as_string",
        'skip-interpreter-query = "yes"',
    ),
    (
        "recursion_depth_limit_as_string",
        'recursion-depth-limit = "10"',
    ),
    (
        "recursion_depth_limit_negative",
        "recursion-depth-limit = -1",
    ),
    (
        "errors_as_string",
        'errors = "all"',
    ),
    # --- Invalid enum values ---
    (
        "untyped_def_behavior_invalid",
        'untyped-def-behavior = "always-infer"',
    ),
    (
        "recursion_overflow_handler_invalid",
        'recursion-overflow-handler = "crash"',
    ),
    (
        "enabled_ignores_invalid_tool",
        'enabled-ignores = ["type", "flake8"]',
    ),
    (
        "error_severity_invalid_string",
        '[errors]\nbad-assignment = "critical"',
    ),
    # --- Wrong structure ---
    (
        "sub_config_as_object",
        '[sub-config]\nmatches = "*.py"',
    ),
    (
        "build_system_as_string",
        'build-system = "buck"',
    ),
    # --- Missing required fields ---
    (
        "build_system_without_type",
        "[build-system]\nignore-if-build-system-missing = true",
    ),
    (
        "sub_config_entry_without_matches",
        "[[sub-config]]\npermissive-ignores = true",
    ),
    # --- Conditional build-system validation ---
    (
        "build_system_buck_with_command",
        '[build-system]\ntype = "buck"\ncommand = ["my-query"]',
    ),
    (
        "build_system_custom_without_command",
        '[build-system]\ntype = "custom"',
    ),
    (
        "build_system_custom_with_isolation_dir",
        '[build-system]\ntype = "custom"\ncommand = ["query"]\nisolation-dir = "iso"',
    ),
    # --- Pattern violations ---
    (
        "python_version_bad_pattern",
        'python-version = "3.x.1"',
    ),
    # --- minItems violation ---
    (
        "build_system_command_empty_array",
        '[build-system]\ntype = "custom"\ncommand = []',
    ),
]


def _make_negative_test_pyrefly(toml_string: str):
    """Create a test method that asserts the TOML string is rejected by pyrefly.json."""

    def test(self: unittest.TestCase) -> None:
        config = toml.loads(toml_string)
        validator = _make_validator(SCHEMAS_DIR / "pyrefly.json")
        with self.assertRaises(jsonschema.ValidationError):
            validator.validate(config)

    return test


def _make_negative_test_pyproject(toml_string: str):
    """Create a test method that asserts the config is rejected by
    pyproject-tool-pyrefly.json."""

    def test(self: unittest.TestCase) -> None:
        config = toml.loads(toml_string)
        wrapped = {"tool": {"pyrefly": config}}
        validator = _make_validator(SCHEMAS_DIR / "pyproject-tool-pyrefly.json")
        with self.assertRaises(jsonschema.ValidationError):
            validator.validate(wrapped)

    return test


class TestNegativePyrefly(unittest.TestCase):
    """Negative tests: invalid configs should be rejected by pyrefly.json."""


class TestNegativePyproject(unittest.TestCase):
    """Negative tests: invalid configs should be rejected by pyproject-tool-pyrefly.json."""


# Dynamically add a test method per negative case to each class.
for _name, _toml_string in NEGATIVE_TEST_CASES:
    setattr(
        TestNegativePyrefly,
        f"test_{_name}",
        _make_negative_test_pyrefly(_toml_string),
    )
    setattr(
        TestNegativePyproject,
        f"test_{_name}",
        _make_negative_test_pyproject(_toml_string),
    )


if __name__ == "__main__":
    unittest.main()
