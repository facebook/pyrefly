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


def validate_toml_against_schema(toml_file: Path, schema_file: Path) -> bool:
    """Validate a TOML file against a JSON schema."""
    print(f"\n{'=' * 60}")
    print(f"Validating: {toml_file.name}")
    print(f"Schema: {schema_file.name}")
    print("=" * 60)

    try:
        # Load TOML config
        with open(toml_file, "r") as f:
            config = toml.load(f)

        # Load JSON schema
        with open(schema_file, "r") as f:
            schema = json.load(f)

        # For pyproject.toml, extract the tool.pyrefly section
        if "tool" in config and "pyrefly" in config["tool"]:
            config_to_validate = config["tool"]["pyrefly"]
            print("Validating [tool.pyrefly] section")
        else:
            config_to_validate = config
            print("Validating pyrefly.toml config")

        # Validate using a resolver for $ref support
        schema_uri = Path(schema_file).resolve().as_uri()
        resource = referencing.Resource.from_contents(
            schema, default_specification=referencing.jsonschema.DRAFT7
        )
        registry = referencing.Registry().with_resource(schema_uri, resource)
        if "$id" in schema:
            registry = registry.with_resource(schema["$id"], resource)
        validator_cls = jsonschema.validators.validator_for(schema)
        validator = validator_cls(schema, registry=registry)
        validator.validate(config_to_validate)

        print(" Validation PASSED")
        return True

    except jsonschema.ValidationError as e:
        print("❌ Validation FAILED")
        print(f"\nError: {e.message}")
        if e.path:
            print(f"Path: {' -> '.join(str(p) for p in e.path)}")
        if e.schema_path:
            print(f"Schema path: {' -> '.join(str(p) for p in e.schema_path)}")
        return False

    except Exception as e:
        print(f" Error during validation: {e}")
        return False


def main():
    """Run all validation tests."""
    schemas_dir = Path(__file__).parent

    tests = [
        (schemas_dir / "test-pyrefly.toml", schemas_dir / "pyrefly.json"),
        (
            schemas_dir / "test-pyproject.toml",
            schemas_dir / "pyproject-tool-pyrefly.json",
        ),
    ]

    print("Starting schema validation tests...")
    results = []

    for toml_file, schema_file in tests:
        if not toml_file.exists():
            print(f"\n  Warning: Test file not found: {toml_file}")
            continue
        if not schema_file.exists():
            print(f"\n  Warning: Schema file not found: {schema_file}")
            continue

        results.append(validate_toml_against_schema(toml_file, schema_file))

    # Summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print("=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"Tests passed: {passed}/{total}")

    if passed == total:
        print("\n All tests passed!")
        return 0
    else:
        print(f"\n {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
