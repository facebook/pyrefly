#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""
Schema validation test script.


This script validates the test configuration files against the JSON schemas
to ensure the schemas are correctly structured and comprehensive.
It also runs negative tests to ensure the schemas correctly reject invalid configs.

Requirements:
    pip install jsonschema toml referencing
"""

import json
import sys
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


def _make_validator(schema_file: Path):
    """Create a JSON schema validator with $ref support."""
    with open(schema_file, "r") as f:
        schema = json.load(f)
    schema_uri = schema_file.resolve().as_uri()
    resource = referencing.Resource.from_contents(
        schema, default_specification=referencing.jsonschema.DRAFT7
    )
    registry = referencing.Registry().with_resource(schema_uri, resource)
    if "$id" in schema:
        registry = registry.with_resource(schema["$id"], resource)

    # Also register the main pyrefly.json schema so $ref from pyproject schema works.
    main_schema_file = schema_file.parent / "pyrefly.json"
    if main_schema_file.exists() and main_schema_file != schema_file:
        with open(main_schema_file, "r") as f:
            main_schema = json.load(f)
        main_resource = referencing.Resource.from_contents(
            main_schema, default_specification=referencing.jsonschema.DRAFT7
        )
        main_uri = main_schema_file.resolve().as_uri()
        registry = registry.with_resource(main_uri, main_resource)
        if "$id" in main_schema:
            registry = registry.with_resource(main_schema["$id"], main_resource)
        # Also register by relative name so "$ref": "pyrefly.json" resolves.
        registry = registry.with_resource("pyrefly.json", main_resource)

    validator_cls = jsonschema.validators.validator_for(schema)
    return validator_cls(schema, registry=registry)


def validate_toml_against_schema(toml_file: Path, schema_file: Path) -> bool:
    """Validate a TOML file against a JSON schema."""
    print(f"\n{'=' * 60}")
    print(f"Validating: {toml_file.name}")
    print(f"Schema: {schema_file.name}")
    print("=" * 60)

    try:
        with open(toml_file, "r") as f:
            config = toml.load(f)

        # For pyproject.toml, extract the tool.pyrefly section.
        if "tool" in config and "pyrefly" in config["tool"]:
            config_to_validate = config["tool"]["pyrefly"]
            print("Validating [tool.pyrefly] section")
        else:
            config_to_validate = config
            print("Validating pyrefly.toml config")

        validator = _make_validator(schema_file)
        validator.validate(config_to_validate)

        print(" Validation PASSED")
        return True

    except jsonschema.ValidationError as e:
        print("Validation FAILED")
        print(f"\nError: {e.message}")
        if e.path:
            print(f"Path: {' -> '.join(str(p) for p in e.path)}")
        if e.schema_path:
            print(f"Schema path: {' -> '.join(str(p) for p in e.schema_path)}")
        return False

    except Exception as e:
        print(f" Error during validation: {e}")
        return False


def validate_toml_should_fail(name: str, toml_string: str, schema_file: Path) -> bool:
    """Validate that a TOML string is correctly rejected by the schema.

    Returns True if the schema rejected it (expected behavior).
    """
    try:
        config = toml.loads(toml_string)
        validator = _make_validator(schema_file)
        validator.validate(config)
        # If we get here, validation passed but it should have failed.
        print(f"  FAIL (unexpectedly accepted): {name}")
        return False
    except jsonschema.ValidationError:
        print(f"  ok (correctly rejected): {name}")
        return True
    except Exception as e:
        print(f"  ERROR: {name}: {e}")
        return False


# Each entry is (name, toml_string). All should be rejected by pyrefly.json.
NEGATIVE_TEST_CASES: list[tuple[str, str]] = [
    # --- Wrong types ---
    (
        "project-includes as string instead of array",
        'project-includes = "**/*.py"',
    ),
    (
        "python-version as number instead of string",
        "python-version = 3.13",
    ),
    (
        "skip-interpreter-query as string instead of boolean",
        'skip-interpreter-query = "yes"',
    ),
    (
        "recursion-depth-limit as string instead of integer",
        'recursion-depth-limit = "10"',
    ),
    (
        "recursion-depth-limit as negative integer",
        "recursion-depth-limit = -1",
    ),
    (
        "errors as string instead of object",
        'errors = "all"',
    ),
    # --- Invalid enum values ---
    (
        "untyped-def-behavior with invalid value",
        'untyped-def-behavior = "always-infer"',
    ),
    (
        "recursion-overflow-handler with invalid value",
        'recursion-overflow-handler = "crash"',
    ),
    (
        "enabled-ignores with invalid tool name",
        'enabled-ignores = ["type", "flake8"]',
    ),
    (
        "error severity with invalid string",
        '[errors]\nbad-assignment = "critical"',
    ),
    # --- Wrong structure ---
    (
        "sub-config as object instead of array",
        '[sub-config]\nmatches = "*.py"',
    ),
    (
        "build-system as string instead of object",
        'build-system = "buck"',
    ),
    # --- Missing required fields ---
    (
        "build-system without type",
        "[build-system]\nignore-if-build-system-missing = true",
    ),
    (
        "sub-config entry without matches",
        "[[sub-config]]\npermissive-ignores = true",
    ),
    # --- Conditional build-system validation ---
    (
        "build-system buck with command present",
        '[build-system]\ntype = "buck"\ncommand = ["my-query"]',
    ),
    (
        "build-system custom without command",
        '[build-system]\ntype = "custom"',
    ),
    (
        "build-system custom with isolation-dir",
        '[build-system]\ntype = "custom"\ncommand = ["query"]\nisolation-dir = "iso"',
    ),
    # --- Pattern violations ---
    (
        "python-version not matching pattern",
        'python-version = "3.x.1"',
    ),
    # --- minItems violation ---
    (
        "build-system command as empty array",
        '[build-system]\ntype = "custom"\ncommand = []',
    ),
]


def run_negative_tests(schema_file: Path) -> tuple[int, int]:
    """Run all negative tests against a schema. Returns (passed, total)."""
    passed = 0
    total = len(NEGATIVE_TEST_CASES)
    for name, toml_string in NEGATIVE_TEST_CASES:
        if validate_toml_should_fail(name, toml_string, schema_file):
            passed += 1
    return passed, total


def validate_config_should_fail_pyproject(
    name: str, config: dict, schema_file: Path
) -> bool:
    """Validate that a config dict wrapped under tool.pyrefly is correctly
    rejected by the pyproject schema.

    Returns True if the schema rejected it (expected behavior).
    """
    wrapped = {"tool": {"pyrefly": config}}
    try:
        validator = _make_validator(schema_file)
        validator.validate(wrapped)
        print(f"  FAIL (unexpectedly accepted): {name}")
        return False
    except jsonschema.ValidationError:
        print(f"  ok (correctly rejected): {name}")
        return True
    except Exception as e:
        print(f"  ERROR: {name}: {e}")
        return False


def run_negative_tests_pyproject(
    pyrefly_schema_file: Path, pyproject_schema_file: Path
) -> tuple[int, int]:
    """Run negative tests against the pyproject schema by wrapping configs
    under [tool.pyrefly]."""
    passed = 0
    total = len(NEGATIVE_TEST_CASES)
    for name, toml_string in NEGATIVE_TEST_CASES:
        # Parse the TOML string first, then nest under tool.pyrefly.
        # Simple string wrapping doesn't work for TOML section headers.
        config = toml.loads(toml_string)
        if validate_config_should_fail_pyproject(
            f"(pyproject) {name}", config, pyproject_schema_file
        ):
            passed += 1
    return passed, total


def main():
    """Run all validation tests."""
    schemas_dir = Path(__file__).parent

    # --- Positive tests ---
    positive_tests = [
        (schemas_dir / "test-pyrefly.toml", schemas_dir / "pyrefly.json"),
        (
            schemas_dir / "test-pyproject.toml",
            schemas_dir / "pyproject-tool-pyrefly.json",
        ),
    ]

    print("Starting schema validation tests...")
    print("\n--- Positive Tests (should pass) ---")
    positive_results = []

    for toml_file, schema_file in positive_tests:
        if not toml_file.exists():
            print(f"\n  Warning: Test file not found: {toml_file}")
            continue
        if not schema_file.exists():
            print(f"\n  Warning: Schema file not found: {schema_file}")
            continue

        positive_results.append(validate_toml_against_schema(toml_file, schema_file))

    # --- Negative tests ---
    pyrefly_schema = schemas_dir / "pyrefly.json"
    pyproject_schema = schemas_dir / "pyproject-tool-pyrefly.json"

    print(f"\n{'=' * 60}")
    print("Negative Tests - pyrefly.json (should be rejected)")
    print("=" * 60)
    neg_passed, neg_total = run_negative_tests(pyrefly_schema)

    print(f"\n{'=' * 60}")
    print("Negative Tests - pyproject-tool-pyrefly.json (should be rejected)")
    print("=" * 60)
    neg_pyproject_passed, neg_pyproject_total = run_negative_tests_pyproject(
        pyrefly_schema, pyproject_schema
    )

    # --- Summary ---
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print("=" * 60)

    pos_passed = sum(positive_results)
    pos_total = len(positive_results)
    print(f"Positive tests passed: {pos_passed}/{pos_total}")
    print(f"Negative tests passed (pyrefly.json): {neg_passed}/{neg_total}")
    print(
        f"Negative tests passed (pyproject-tool-pyrefly.json): "
        f"{neg_pyproject_passed}/{neg_pyproject_total}"
    )

    all_passed = (
        pos_passed == pos_total
        and neg_passed == neg_total
        and neg_pyproject_passed == neg_pyproject_total
    )

    if all_passed:
        print("\nAll tests passed!")
        return 0
    else:
        total_failures = (
            (pos_total - pos_passed)
            + (neg_total - neg_passed)
            + (neg_pyproject_total - neg_pyproject_passed)
        )
        print(f"\n{total_failures} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
