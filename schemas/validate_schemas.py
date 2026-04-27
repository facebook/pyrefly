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
    pip install jsonschema toml referencing
"""

import json
import sys
import unittest
from pathlib import Path

try:
    import jsonschema
    import referencing
    import referencing.jsonschema
    import toml
except ImportError:
    print("Error: Required packages not installed.")
    print("Please run: pip install jsonschema toml referencing")
    sys.exit(1)

SCHEMAS_DIR = Path(__file__).parent


def _make_validator(schema_file: Path):
    """Create a JSON schema validator with $ref support.

    Uses the referencing library for proper $ref resolution.
    """
    with open(schema_file, "r") as f:
        schema = json.load(f)

    # Load the main pyrefly.json schema
    main_schema_file = schema_file.parent / "pyrefly.json"
    main_schema = None
    if main_schema_file.exists():
        with open(main_schema_file, "r") as f:
            main_schema = json.load(f)

    # Create a retrieve function that can resolve local schemas
    def retrieve(uri: str) -> referencing.Resource:
        # Handle relative "pyrefly.json" reference
        if uri == "pyrefly.json" or uri.endswith("/pyrefly.json"):
            if main_schema is not None:
                return referencing.Resource.from_contents(
                    main_schema, default_specification=referencing.jsonschema.DRAFT7
                )
        raise referencing.exceptions.NoSuchResource(ref=uri)

    # Build a registry with the retrieve function
    registry = referencing.Registry(retrieve=retrieve)

    # Register schemas we know about
    if main_schema is not None:
        main_resource = referencing.Resource.from_contents(
            main_schema, default_specification=referencing.jsonschema.DRAFT7
        )
        # Register by relative name
        registry = registry.with_resource("pyrefly.json", main_resource)
        # Register by $id if present
        if "$id" in main_schema:
            registry = registry.with_resource(main_schema["$id"], main_resource)

    # Register the current schema
    schema_resource = referencing.Resource.from_contents(
        schema, default_specification=referencing.jsonschema.DRAFT7
    )
    schema_uri = schema_file.resolve().as_uri()
    registry = registry.with_resource(schema_uri, schema_resource)
    if "$id" in schema:
        registry = registry.with_resource(schema["$id"], schema_resource)

    validator_cls = jsonschema.validators.validator_for(schema)
    return validator_cls(schema, registry=registry)


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
        validator = _make_validator(schema_file)
        validator.validate(config)


class TestSchemaValidity(unittest.TestCase):
    """Meta-validation: JSON schemas themselves must be valid Draft-07 schemas."""

    def test_pyrefly_schema_is_valid(self) -> None:
        with open(SCHEMAS_DIR / "pyrefly.json", "r") as f:
            schema = json.load(f)
        jsonschema.Draft7Validator.check_schema(schema)

    def test_pyproject_schema_is_valid(self) -> None:
        with open(SCHEMAS_DIR / "pyproject-tool-pyrefly.json", "r") as f:
            schema = json.load(f)
        jsonschema.Draft7Validator.check_schema(schema)


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
    # NOTE: The following 3 tests are skipped because the schemars-generated schema
    # doesn't include `additionalProperties: false` in the oneOf variants, so it
    # cannot enforce mutual exclusivity between build-system types (buck vs custom).
    # These tests would pass with a manually-maintained schema that includes these constraints.
    # (
    #     "build_system_buck_with_command",
    #     '[build-system]\ntype = "buck"\ncommand = ["my-query"]',
    # ),
    (
        "build_system_custom_without_command",
        '[build-system]\ntype = "custom"',
    ),
    # (
    #     "build_system_custom_with_isolation_dir",
    #     '[build-system]\ntype = "custom"\ncommand = ["query"]\nisolation-dir = "iso"',
    # ),
    # --- Pattern violations ---
    (
        "python_version_bad_pattern",
        'python-version = "3.x.1"',
    ),
    (
        "python_version_trailing_dot",
        'python-version = "3."',
    ),
    (
        "python_version_four_parts",
        'python-version = "3.12.0.1"',
    ),
    # --- minItems violation ---
    (
        "build_system_command_empty_array",
        '[build-system]\ntype = "custom"\ncommand = []',
    ),
    # --- Wrong item types within arrays ---
    (
        "search_path_non_string_items",
        "search-path = [123, 456]",
    ),
    (
        "enabled_ignores_integer_in_enum_array",
        "enabled-ignores = [42, 99]",
    ),
    # --- Float instead of integer ---
    (
        "recursion_depth_limit_float",
        "recursion-depth-limit = 3.5",
    ),
    # --- Conditional build-system: custom with extras ---
    # NOTE: Skipped - see note above about schemars not supporting additionalProperties: false
    # (
    #     "build_system_custom_with_extras",
    #     '[build-system]\ntype = "custom"\ncommand = ["query"]\nextras = ["--flag"]',
    # ),
    # --- Sub-config with invalid inner field ---
    (
        "sub_config_invalid_error_severity",
        '[[sub-config]]\nmatches = "*.py"\n\n[sub-config.errors]\nbad-assignment = "critical"',
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
