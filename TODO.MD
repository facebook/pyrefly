# TODO

## Documentation Updates

- [ ] **Update website documentation for the new query-script build system feature**
  - Document the new `query-script` build system type that was added in this branch
  - The feature allows using custom scripts to query build information instead of being limited to Buck builds
  - Add configuration examples showing how to set up `pyrefly.toml` with:
    ```toml
    build-system = { type = "query-script", command = "python3", args = ["tools/pyrefly_bazel_query.py", "--"] }
    ```

- [ ] **Include examples for the JSON format expected by query scripts**
  - Reference the sample implementation: https://github.com/lolpack/pyrefly_bazel_sample
  - Document the expected JSON schema for script output:
    - `ScriptManifestDatabase` structure with `db` and `root` fields
    - `ScriptManifest` structure with `deps`, `srcs`, `python_version`, `python_platform`, and optional `buildfile_path`
  - Provide example JSON output format
  - Document command line interface expectations (scripts should accept `--file` arguments)

## Implementation Notes

Current branch (`add_build_system_support`) adds:
- New `QueryScript` build system type in `crates/pyrefly_build/src/lib.rs`
- Complete implementation in `crates/pyrefly_build/src/script.rs`
- Configuration support in `crates/pyrefly_config/src/config.rs`
- Test coverage for the new functionality

The query script approach provides a flexible foundation that could be used to implement support for many of the build systems listed above.