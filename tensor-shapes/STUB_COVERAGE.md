# Stub overlay coverage

The NumPy, JAX, and Torch shape packages are partial PEP 561 stub packages. A
module supplied by one of these packages replaces that module's normal typing
surface, so every public name omitted from the overlay becomes unavailable to
users of the shape stubs.

`stub_coverage.py` compares each shipped stub module with the corresponding
runtime module. It parses declarations from the `.pyi` file and uses the union
of public `dir()` entries and names in `__all__` for the runtime library. A
computed stub `__all__` falls back to declaration discovery rather than
preventing inspection. The tool also compares explicitly configured classes
such as `torch.Tensor` and `numpy.ndarray`.

Class comparisons use every runtime-visible member and declarations made
directly on the configured stub class. A future target that inherits stub
members may therefore over-report those inherited names as gaps.

Each package has a `stub_coverage.toml` configuration. Overlay-only modules and
incidental runtime names can be excluded there.

## Configuration

The two required top-level keys are `runtime_package` and `stub_directory`.
All collection keys are optional:

```toml
runtime_package = "package"
stub_directory = "package-stubs"
skip_modules = ["package.overlay_only"]

[module_excludes]
"package.module" = ["incidental_runtime_name"]

[member_excludes]
"package.module.Class" = ["incidental_runtime_member"]

[[member_targets]]
stub_module = "package.module"
stub_class = "Class"
runtime_module = "package.module"
runtime_class = "Class"
```

`runtime_package` is the importable package being inspected. `stub_directory`
is the overlay tree relative to the configuration file. `skip_modules` lists
overlay modules that have no corresponding runtime module. `module_excludes`
maps fully qualified runtime modules to public names that should not count as
gaps. `member_excludes` maps a composite `<stub_module>.<stub_class>` key to
runtime-visible members that should not count as gaps. Dotted keys in both
tables must be quoted as shown above.

Each `member_targets` table compares one class. Its four required fields name
the module and class in the overlay (`stub_module`, `stub_class`) and at runtime
(`runtime_module`, `runtime_class`). Unknown top-level keys and unknown member
target fields are rejected.

Run the inspection with the shared test virtualenv:

```bash
~/.tensor-shapes-venv/bin/python tensor-shapes/stub_coverage.py \
  tensor-shapes/pyrefly-torch-stubs/stub_coverage.toml
```

The default output gives counts by module and class. Add `--show-names` for a
human-readable inventory or `--json` for machine-readable analysis. Replace
`torch` with `numpy` or `jax` for the other packages.

This is currently an inspection tool rather than a CI ratchet: it does not
compare against checked-in snapshots or fail based on the number of gaps. The
checker reports the installed library version but does not enforce it; the
shared virtualenv is currently kept stable by `test-requirements.txt`.
