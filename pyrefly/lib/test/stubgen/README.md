# Stubgen Test Fixtures

Each subdirectory is a snapshot test case containing:

- `input.py` — Python source fed to stubgen
- `expected.pyi` — expected stub output

## Adding a new test case

1. Create a new subdirectory (e.g. `my_feature/`).
2. Add an `input.py` with the Python code to test.
3. Add a `#[test]` function in `stubgen/mod.rs` that calls
   `assert_stubgen_snapshot("my_feature")`.
4. Run with `STUBGEN_UPDATE_SNAPSHOTS=1` (see below) to generate
   `expected.pyi`, then review the output.

**Important:** adding the fixture files alone is not enough — you must
also add the corresponding test function, or the fixture will never be
exercised.

## Updating snapshots

When stubgen output changes intentionally, regenerate the expected files:

```bash
# With Buck (internal):
STUBGEN_UPDATE_SNAPSHOTS=1 buck test pyrefly:pyrefly_library -- test_stubgen

# With Cargo (external):
STUBGEN_UPDATE_SNAPSHOTS=1 cargo test test_stubgen
```

Review the diffs in `expected.pyi` files before committing.
