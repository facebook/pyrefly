*Release date: June 10, 2026*

> **About dev releases**
> Dev releases (versions like `X.Y.Z-dev.N`) are non-stable snapshots cut periodically from trunk. They give early adopters a chance to try in-progress features and surface issues before the next stable release, but they don't carry the same stability or compatibility guarantees as a stable release — don't pin production projects to a dev version.

Pyrefly v1.1.0-dev.2 bundles **250 commits** from **37 contributors**.

---

## ✨ New & Improved

### Type Checking

- Improved handling of defaults for function parameters annotated with type variables. Parameters like `x: T = 0` now correctly validate their defaults against the type variable's constraints or bounds, and the default value is properly used when solving type variables in function calls.
- Enhanced type narrowing for constrained type variables. Operations like `isinstance` checks and binary operations now preserve the type variable while correctly narrowing to specific constraints, fixing several false positives and negatives.
- Improved handling of methods on type variables. When calling methods on a type variable (e.g., `x.lower()` where `x: T` with `T` constrained to `bytes | str`), the return type now correctly preserves the type variable when the method returns `self`'s type.
- Better support for shaped arrays across libraries. The tensor shape machinery now works with any array class that opts in via `shape_extensions.shaped_array`, including `numpy.ndarray` and `jax.Array`, not just PyTorch tensors.
- Improved literal promotion and handling in function parameter defaults, making type inference more predictable when defaults include literal values.
- Fixed handling of binary operations and comparisons on constrained type variables, ensuring the type variable is preserved when appropriate rather than degrading to a union of constraints.

### Language Server

- Literal value completions now work in more contexts. When assigning to an annotated variable, returning from a function, or setting an attribute with a `Literal` type, the editor now suggests the valid literal values.
- New TSP endpoint `typeServer/getExpectedType` returns the contextually expected type at a cursor position (e.g., the parameter type for a call argument, the declared type for an assignment), enabling richer IDE features.
- Improved "move module member" refactoring. When moving a symbol to another module, Pyrefly now rewrites consumer imports automatically, updating `from source import foo` to `from target import foo` across the codebase.
- New "move symbol to new file" refactoring creates a new file for a top-level symbol, moves the definition, and updates all consumer imports in one action.
- Enhanced pytest fixture support. Go-to-definition and find-references now work for pytest fixtures used as function parameters, navigating to the fixture definition.
- Better handling of non-Python modules in build systems. Go-to-definition on imports from `.thrift` files and other non-Python modules now navigates to the module file when the specific symbol can't be resolved.

### Coverage Reporting

- New `pyrefly coverage check` command provides a user-friendly way to verify type coverage meets a threshold, with configurable output formats and fail-under percentage.
- Coverage reports now exclude symbols that can't be imported due to shadowed namespace packages, providing more accurate coverage metrics.
- Improved handling of stub re-exports. When a stub re-exports a class, the class's members are now correctly excluded from the coverage report to avoid inconsistent results.
- Coverage reports now respect `__all__` in stubs more accurately, only treating imports as public re-exports when they appear in `__all__` or use the `as` alias syntax.

### Error Reporting

- Error output is now significantly faster (up to 75% reduction in printing time for large error counts) through optimized rendering and buffering.
- New `--output-format junit-xml` option emits JUnit XML test reports, making it easier to integrate Pyrefly into CI dashboards.
- Improved error messages for `None`-related type mismatches. When passing `T | None` where `T` is expected, the error now includes an actionable hint suggesting narrowing with `is not None` or widening the expected type.
- Better error messages for missing overload matches, now including details about arity mismatches and specific argument type errors.

---

## 🐛 Bug fixes

We closed **24** bug issues this release 👏

- **#3554:** Fixed false-positive `unknown-name` errors after short-circuit boolean operations. Code like `if SOMETHING_DEFINITELY_FALSE and A:` no longer reports `A` as unknown, since it's unreachable.
- **#3300:** Fixed `unknown-name` false positive when a name is introduced via a walrus operator in a decorator. Names defined with `:=` in decorator expressions are now visible after the decorated definition.
- **#1496, #3418, #3623:** Fixed handling of defaults for function parameters with type variable annotations. Defaults like `x: T = 0` are now validated correctly and used when solving type variables in calls.
- **#3059:** Fixed `__getitem__` resolution on type variables with constraints. Subscripting a type variable bounded by `TypedDict` or other subscriptable types now works correctly instead of falling back to `object.__getitem__`.
- **#3603:** Fixed type narrowing for discriminated `TypedDict` unions when using membership tests. Expressions like `event["ph"] in ("X", "C")` now correctly narrow the union to matching members.
- **#3079, #3082:** Fixed false-positive key errors after `key in dict` checks. When a `TypedDict` key is tested with `in`, subsequent subscript access no longer reports the key as possibly absent.
- **#3607:** Fixed import resolution when using a config file. Bare imports like `from bar import func` that work at runtime now resolve correctly even when a `pyrefly.toml` is present, via the new `enable-fallback-search-path` option.
- **#357:** Fixed unquoted forward reference handling in Python ≤3.13. Runtime-evaluated annotations now correctly report uninitialized forward names unless `from __future__ import annotations` is active.
- **#3536:** Fixed `invalid-variance` error location for overloaded methods. The error now appears on the offending overload signature rather than always pointing to the first overload.
- **#3671:** Fixed stack overflow when a descriptor's `__get__` is itself a descriptor. Self-referential descriptor patterns now terminate gracefully instead of crashing.
- And more! #3235, #3673, #3541, #3684, #3647, #3632, #3626, #3640, #3641, #3639, #3645, #3701, #3664, #3598

Thank-you to all our contributors who found these bugs and reported them! Did you know this is one of the most helpful contributions you can make to an open-source project? If you find any bugs in Pyrefly we want to know about them! Please open a bug report issue [here](https://github.com/facebook/pyrefly/issues).

---

## 📦 Upgrade

```bash
pip install --upgrade pyrefly==1.1.0-dev.2
```

### How to safely upgrade your codebase

Upgrading the version of Pyrefly you're using or a third-party library you depend on can reveal new type errors in your code. Fixing them all at once is often unrealistic. We've written scripts to help you temporarily silence them. After upgrading, follow these steps:

1. `pyrefly check --suppress-errors`
2. Run your code formatter of choice
3. `pyrefly check --remove-unused-ignores`
4. Repeat until you achieve a clean formatting run and a clean type check.

This will add `# pyrefly: ignore` comments to your code, enabling you to silence errors and return to fix them later. This can make the process of upgrading a large codebase much more manageable.

Read more about error suppressions in the [Pyrefly documentation](https://pyrefly.org/en/docs/error-suppressions/).

---

## 🖊️ Contributors this release

@rchen152, @stroxler, @kinto0, @yangdanny97, @connernilsen, @grievejia, @NathanTempest, @jorenham, @samwgoldman, @asukaminato0721, David Tolnay, @lolpack, @mikeleppane, @maggiemoss, @MarcoGorelli, @JakobDegen, @arthaud, Anqi Wu, @terror, @bigfootjon, @aahanaggarwal, @rchiodo, @magic-akari, @SamB, @nitishagar, @ndmitchell, @NarxPal, Gregory Carlin, @javabster, @thatfunkymunki, @shobhitmehro, @hrolfurgylfa

---

*Please note: These release notes summarize major updates and features. For brevity, not all individual commits are listed.*
