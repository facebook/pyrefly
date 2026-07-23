*Release date: May 27, 2026*

> **About dev releases**
> Dev releases (versions like `X.Y.Z-dev.N`) are non-stable snapshots cut periodically from trunk. They give early adopters a chance to try in-progress features and surface issues before the next stable release, but they don't carry the same stability or compatibility guarantees as a stable release — don't pin production projects to a dev version.

Pyrefly v1.1.0-dev.1 bundles **250 commits** from **40 contributors**.

---

## ✨ New & Improved

### Type Checking

- New `incompatible-comparison` diagnostic catches comparisons like `int == str` where operand types can't overlap.
- Classes whose metaclass extends `ABCMeta` are now correctly treated as abstract.
- Frozen dataclasses reject manual `__setattr__`/`__delattr__` overrides.
- Annotation-only descriptor fields are recognized as descriptors — fixes false positives for SQLAlchemy `Mapped` and similar asymmetric descriptor types.

### Language Server

- Code actions can convert dict literals to TypedDict, dataclass, or Pydantic model — with automatic imports.
- Sphinx cross-references (`:meth:`, `:class:`, `:func:`) in docstrings now render as clickable links in hover tooltips.
- Inlay hints for class attributes defined in `__init__` without explicit annotations.
- Supports `python.analysis.diagnosticMode`, `python.analysis.importFormat`, and `python.analysis.inlayHints` settings — full interop with the Python VS Code ecosystem.

### Error Reporting

- New `explicit-any` error kind (ignored by default) warns when `Any` appears in annotations or type alias bodies; includes migration support for mypy/pyright codebases.
- Diagnostic paths in `baseline.json` are now relative and always use forward slashes — fixes baseline matching when CI checkout paths differ from local dev environments and across OSes.
- The `annotation-mismatch` error kind has been removed (it was never emitted).

### Configuration

- New `all` preset enables every error kind at error severity — for codebases that want comprehensive checking. Available in both the CLI and the VS Code `python.pyrefly.typeCheckingMode` setting.
- New `pytorch-efficiency-lints` flag turns on PyTorch efficiency checks at warn severity: catches `.item()` GPU sync, redundant `.to(device)`, deprecated `.cuda()`, and `print(tensor)`.

### Tensor Shapes

- Tensor shape operations now live in user-space stubs (`torch/_shapes.pyi`) via the new `@uses_shape_dsl` decorator.
- DSL compile errors and shape-DSL type errors now emit proper diagnostics with precise source spans instead of panicking.
- Recursive `@shape_dsl_function` definitions and invalid call signatures are caught at compile time.

### Stubgen

- Stubgen now emits synthesized `__init__` for dataclasses, preserving constructor signatures in generated stubs.

### Build & CI

- **Pyrefly now compiles on stable Rust** — no more nightly toolchain dependency. GitHub CI workflows updated to match.
- musllinux wheels are now built for Alpine and other musl-based distros.
- Conformance test sources updated from upstream `python/typing`.

---

## 🐛 Bug fixes

We closed **34** bug issues this release 👏

- **#2798:** Fixed false positive `not-iterable` error when iterating over a list stored in a dict that also contains non-iterable values. When a dict was built inside a loop with heterogeneous inner dicts (e.g., `{"start": date, "tasks": []}`), Pyrefly incorrectly inferred the inner dict's value type as `date` instead of `date | list[...]`, causing spurious errors when accessing `val["tasks"]`.
- **#3188:** Fixed error suppression for multi-line implicit string concatenations. When a `# pyrefly: ignore` comment appeared above a multi-line implicit string concatenation, only the first error was covered. Pyrefly now reparses the AST to respect ignore comments for the entirety of the construct.
- **#3385:** Fixed false positive `missing-override-decorator` when assigning to inherited properties. If a child class assigned to a property (with a setter) defined in a base class, Pyrefly incorrectly flagged it as an implicit override.
- **#3208:** Fixed stubgen missing instance variables defined in `__init__`. Classes in generated stubs were missing attributes if those attributes were not explicitly annotated at the class level. Stubgen now emits synthesized `__init__` for dataclasses, preserving constructor signatures.
- **#3237:** Fixed cross-file references lost when CRLF/LF line endings differ. When files on disk used CRLF but the editor sent LF-normalized content via `textDocument/didOpen`, byte offsets diverged, causing exact byte-range comparisons to fail. Pyrefly now uses a line-number fallback to match definitions correctly.
- **#3378:** Fixed exhaustive Enum variants not narrowing the type. When a `Final` Enum variant was checked with `is not`, the type was not narrowed out of the union, causing false positives in exhaustiveness checks.
- **#3413:** Fixed strict mode insisting on `@override` for standard dunders. Pyrefly now skips `missing-override-decorator` for dunders inherited from `object`, as adding `@override` to `__repr__`, `__eq__`, etc., is pure noise.
- **#3455:** Fixed crash when overload residual capture included `PartialQuantified` vars. The solver now correctly handles partial vars in overload residuals, preventing panics.
- **#3392:** Fixed `Self@[Enum]` not being properly narrowed. For an exhaustive `match` on an enum type within an instance method, the `self` argument wasn't being narrowed as expected. Enum member subtraction now preserves and handles `SelfType`, allowing exhaustive enum match self fallbacks to narrow to `Never`.
- **#3299:** Fixed incorrect type inference after `__setitem__` / `__getitem__`. When a class's `__setitem__` and `__getitem__` signatures were asymmetric, Pyrefly incorrectly narrowed the assigned value. Subscript-assignment narrowing is now gated by symmetry checks, preventing false positives with custom container types.
- And more! #3446, #3448, #3396, #1492, #2451, #3431, #3210, #3381, #2547, #3147, #3524, #3506, #3519, #3343, #3505, #3369, #3344, #3400, #3394, #3221, #3514, #3520, #3544, #3458

Thank-you to all our contributors who found these bugs and reported them! Did you know this is one of the most helpful contributions you can make to an open-source project? If you find any bugs in Pyrefly we want to know about them! Please open a bug report issue [here](https://github.com/facebook/pyrefly/issues).

---

## 📦 Upgrade

```bash
pip install --upgrade pyrefly==1.1.0-dev.1
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

@stroxler, @rchen152, @ndmitchell, generatedunixname2066905484085733, @yangdanny97, @kinto0, @jorenham, @asukaminato0721, @migeed-z, @samwgoldman, @lolpack, @grievejia, David Tolnay, @connernilsen, @arthaud, @maggiemoss, @NathanTempest, @f1sherFM, @mfish33, @lordhaa123, Anqi Wu, @javabster, @rexledesma, @MackDing, Brian Rosenfeld, @fylux, @YBJ0000, @arpitjain099, @EpicEric, @NarxPal, Philippe Bidinger, @knQzx, @yeetypete, @GHX5T-SOL, Tejesh Mehta, @fangyi-zhou, @LeSingh1, @P-r-e-m-i-u-m, @Arths17, @sjh9714

---

*Please note: These release notes summarize major updates and features. For brevity, not all individual commits are listed.*
