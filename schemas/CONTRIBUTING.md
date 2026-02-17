# Contributing to Pyrefly Schemas

Thank you for your interest in improving Pyrefly's JSON Schema definitions!

## Overview

The schemas in this directory provide validation, autocomplete, and documentation for Pyrefly configuration files. When updating the schemas, it's important to keep them in sync with the actual configuration implementation.

## Schema Files

- **pyrefly.json**: Schema for standalone `pyrefly.toml` files
- **pyproject-tool-pyrefly.json**: Schema for the `[tool.pyrefly]` section in `pyproject.toml`

Both schemas should have identical configuration options, just wrapped differently.

## When to Update Schemas

Update the schemas whenever:

1. **New configuration options are added** to Pyrefly
2. **Configuration options are modified** (renamed, type changed, etc.)
3. **Configuration options are deprecated or removed**
4. **Default values change**
5. **Enum values are added or removed**

## How to Update Schemas

### 1. Check the Source of Truth

The authoritative source for configuration options is:
- `crates/pyrefly_config/src/config.rs` - Main config structure
- `crates/pyrefly_config/src/base.rs` - Base config options
- `website/docs/configuration.mdx` - User-facing documentation

Always verify your changes against these sources.

### 2. Update Both Schema Files

When adding a new configuration option, update `pyrefly.json`. `pyproject-tool-pyrefly.json` contains the definition for the `[tool.pyrefly]` section in a `pyproject.toml` file, so it likely won't need to be updated.

### 3. Follow JSON Schema Best Practices

- **Use descriptive descriptions**: Each property should have a clear description
- **Set appropriate types**: Use `string`, `boolean`, `array`, `object`, etc.
- **Add default values**: Include the actual default value from the Rust code
- **Use enums for fixed values**: For options like `python-platform` or `untyped-def-behavior`
- **Add patterns for simple validation**: For example, version strings should match `^\d+(\.\d+)?(\.\d+)?$`
- **Mark required properties**: Use `required` array for mandatory fields

### 4. Add Test Cases

Add examples of the new configuration option to the test files:
- `test-pyrefly.toml` - Examples for `pyrefly.toml`
- `test-pyproject.toml` - Examples for `pyproject.toml`

### 5. Validate Your Changes

Run the validation script to ensure the schemas are correct:

```bash
python schemas/validate_schemas.py
```

This script validates the test configuration files against the schemas.

### 6. Update Documentation

If you're adding a new configuration option, also update:
- `schemas/README.md` - If the change affects how users interact with schemas
- `website/docs/configuration.mdx` - User-facing documentation (if not already done)

## Schema Structure

### Common Properties Format

```json
"property-name": {
  "description": "Clear description of what this does",
  "type": "string|boolean|array|object|number",
  "default": <default-value>,
  "enum": ["value1", "value2"],  // For fixed set of values
  "pattern": "regex",            // For string validation
  "items": {...},                // For array element types
  "properties": {...}            // For object properties
}
```

### Naming Conventions

- Use kebab-case for property names: `python-version`, not `python_version`
- Match the exact names used in the TOML configuration
- For deprecated options, add a note in the description

### Example: Adding a New Boolean Option

```json
"new-option-name": {
  "description": "Whether to enable the new feature. Default is false.",
  "type": "boolean",
  "default": false
}
```

### Example: Adding a New Enum Option

```json
"new-mode": {
  "description": "The mode to use for the new feature.",
  "type": "string",
  "enum": ["strict", "lenient", "off"],
  "default": "lenient"
}
```

### Example: Adding a New Array Option

```json
"new-patterns": {
  "description": "List of glob patterns for the new feature.",
  "type": "array",
  "items": {
    "type": "string"
  },
  "default": []
}
```

## Testing

### Manual Testing

1. Create a test configuration file with the new option
2. Open it in VS Code with the "Even Better TOML" extension
3. Verify that:
   - Autocomplete suggests the new option
   - Hover shows the correct description
   - Invalid values are highlighted
   - Default value is documented

### Automated Testing

The `validate_schemas.py` script validates test files against schemas:

```bash
# Install dependencies (if not already installed)
pip install jsonschema toml

# Run validation
python schemas/validate_schemas.py
```

Add test cases to `test-pyrefly.toml` and `test-pyproject.toml` that exercise your new configuration option.

## Questions?

If you have questions about updating the schemas:
- Check existing schema definitions for similar patterns
- Review the [JSON Schema documentation](https://json-schema.org/)
- Open an issue or ask on [Discord](https://discord.com/invite/Cf7mFQtW7W)
