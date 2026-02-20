# Pyrefly JSON Schemas

This directory contains JSON Schema definitions for Pyrefly configuration files.

## Available Schemas

### pyrefly.json
JSON Schema for `pyrefly.toml` configuration files. This schema provides validation, autocomplete, and documentation for all Pyrefly configuration options when editing `pyrefly.toml` files.

**Schema URL:** `https://pyrefly.org/schemas/pyrefly.json`

### pyproject-tool-pyrefly.json
JSON Schema for the `[tool.pyrefly]` section in `pyproject.toml` files. This schema provides the same validation and autocomplete features as the pyrefly.json schema, but formatted for use within pyproject.toml files.

**Schema URL:** `https://pyrefly.org/schemas/pyproject-tool-pyrefly.json`

## Editor Integration

### VS Code

VS Code will automatically detect and use these schemas when you have the appropriate extensions installed:

1. **For pyrefly.toml files:**
   - Install the "Even Better TOML" extension
   - Add the following to your `settings.json`:
   ```json
   {
     "evenBetterToml.schema.associations": {
       "pyrefly.toml": "https://pyrefly.org/schemas/pyrefly.json"
     }
   }
   ```

2. **For pyproject.toml files:**
   - The "Even Better TOML" extension automatically supports `[tool.pyrefly]` sections
   - No additional configuration needed

### PyCharm / IntelliJ

PyCharm automatically recognizes TOML schemas. You can configure schema mappings in:
- Settings → Languages & Frameworks → Schemas and DTDs → JSON Schema Mappings

### Other Editors

Most modern editors with TOML support can use JSON Schemas. Consult your editor's documentation for configuring custom schema mappings.

## Using Schemas Locally

During development, you can reference the local schema files:

**For pyrefly.toml:**
Add a comment at the top of your `pyrefly.toml`:
```toml
# yaml-language-server: $schema=./schemas/pyrefly.json
```

**For pyproject.toml:**
Add a comment in the `[tool.pyrefly]` section:
```toml
[tool.pyrefly]
# yaml-language-server: $schema=./schemas/pyproject-tool-pyrefly.json
```

## Schema Features

The schemas provide:

- **Validation**: Ensures your configuration follows the correct structure and types
- **Autocomplete**: Suggests available configuration options as you type
- **Documentation**: Shows descriptions and default values for each configuration option
- **Type Checking**: Validates that values match expected types (strings, arrays, booleans, etc.)
- **Enum Validation**: Ensures enum values (like `python-platform` or `untyped-def-behavior`) are valid

## Contributing to Schemas

When adding new configuration options to Pyrefly:

1. Update the schema files in this directory
2. Ensure the schema matches the actual config structure in `crates/pyrefly_config/`
3. Update the configuration documentation in `website/docs/configuration.mdx`
4. Test the schema with sample configs

## Submitting to SchemaStore

These schemas can be submitted to [SchemaStore](https://github.com/SchemaStore/schemastore) to provide automatic validation in all compatible editors without manual configuration.

To submit:
1. Fork the SchemaStore repository
2. Add the schema files to the `src/schemas/json/` directory
3. Update the `src/api/json/catalog.json` to register the schemas
4. Submit a pull request

For more information, see the [SchemaStore contribution guide](https://github.com/SchemaStore/schemastore/blob/master/CONTRIBUTING.md).
