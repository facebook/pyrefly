# SchemaStore Submission Guide

This guide explains how to submit Pyrefly's JSON schemas to [SchemaStore](https://github.com/SchemaStore/schemastore), which will enable automatic schema validation in all compatible editors without manual configuration.

## What is SchemaStore?

SchemaStore is a community-driven repository of JSON schemas for configuration files. When a schema is added to SchemaStore, most major editors (VS Code, JetBrains IDEs, Neovim, etc.) automatically provide validation, autocomplete, and documentation for the associated file types.

## Prerequisites

Before submitting to SchemaStore, ensure:

1. ✅ The schemas are well-tested with sample configurations
2. ✅ The schemas follow JSON Schema Draft-07 specification
3. ✅ Documentation is complete and accurate
4. ✅ The schemas are hosted at a stable URL (e.g., `https://pyrefly.org/schemas/`)

## Submission Steps

### 1. Host the Schemas

The schemas need to be publicly accessible at stable URLs. The recommended approach is to host them on the Pyrefly website:

- `https://pyrefly.org/schemas/pyrefly.json`
- `https://pyrefly.org/schemas/pyproject-tool-pyrefly.json`

### 2. Fork SchemaStore

```bash
git clone https://github.com/SchemaStore/schemastore.git
cd schemastore
git checkout -b add-pyrefly-schemas
```

### 3. Add Schemas to Catalog

Edit `src/api/json/catalog.json` and add entries for both schemas:

```json
{
  "name": "Pyrefly Configuration",
  "description": "Configuration file for Pyrefly, a fast Python language server and type checker",
  "fileMatch": ["pyrefly.toml", ".pyrefly.toml"],
  "url": "https://pyrefly.org/schemas/pyrefly.json"
},
{
  "name": "Pyrefly Configuration (pyproject.toml)",
  "description": "Pyrefly configuration in pyproject.toml under [tool.pyrefly]",
  "fileMatch": ["pyproject.toml"],
  "url": "https://pyrefly.org/schemas/pyproject-tool-pyrefly.json"
}
```

### 4. (Optional) Add Local Copies

While not strictly required, SchemaStore appreciates having local copies of schemas for backup and faster access:

```bash
cp schemas/pyrefly.json schemastore/src/schemas/json/pyrefly.json
cp schemas/pyproject-tool-pyrefly.json schemastore/src/schemas/json/pyproject-tool-pyrefly.json
```

If adding local copies, update the catalog entries to use local paths:

```json
{
  "name": "Pyrefly Configuration",
  "description": "Configuration file for Pyrefly, a fast Python language server and type checker",
  "fileMatch": ["pyrefly.toml", ".pyrefly.toml"],
  "url": "https://json.schemastore.org/pyrefly.json"
}
```

### 5. Test the Schemas

SchemaStore has automated tests. Run them to ensure your schemas are valid:

```bash
npm install
npm test
```

### 6. Create Pull Request

Commit your changes and create a pull request:

```bash
git add .
git commit -m "Add Pyrefly configuration schemas"
git push origin add-pyrefly-schemas
```

Then create a PR on GitHub with:

- **Title**: "Add Pyrefly configuration schemas"
- **Description**: Explain what Pyrefly is and link to the documentation
- **Checklist**: Confirm you've followed all SchemaStore contribution guidelines

### 7. Respond to Feedback

SchemaStore maintainers may request changes. Common feedback includes:

- Improving descriptions
- Adding more specific validation patterns
- Ensuring enum values are complete
- Fixing schema validation errors

## Maintenance

Once accepted, you'll need to keep the schemas updated:

1. When adding new config options, update both the local schemas and the hosted versions
2. Create a new PR to SchemaStore if the schema URL changes
3. Monitor issues on the SchemaStore repo for user-reported problems

## Alternative: TOML-Specific Editors

Some TOML-specific editors (like the "Even Better TOML" VS Code extension) support schema associations without SchemaStore. These are configured via:

**VS Code settings.json:**
```json
{
  "evenBetterToml.schema.associations": {
    "pyrefly.toml": "https://pyrefly.org/schemas/pyrefly.json",
    "pyproject.toml": "https://pyrefly.org/schemas/pyproject-tool-pyrefly.json"
  }
}
```

## References

- [SchemaStore Repository](https://github.com/SchemaStore/schemastore)
- [SchemaStore Contributing Guide](https://github.com/SchemaStore/schemastore/blob/master/CONTRIBUTING.md)
- [JSON Schema Specification](https://json-schema.org/specification.html)
