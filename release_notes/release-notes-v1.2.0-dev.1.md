*Release date: July 1, 2026*

> **About dev releases**
> Dev releases (versions like `X.Y.Z-dev.N`) are non-stable snapshots cut periodically from trunk. They give early adopters a chance to try in-progress features and surface issues before the next stable release, but they don't carry the same stability or compatibility guarantees as a stable release — don't pin production projects to a dev version.

Pyrefly v1.2.0-dev.1 bundles **235 commits** from **38 contributors**.

---

## ✨ New & Improved

### Type Checking

- Attrs classes are now fully supported with comprehensive field synthesis, validation, and special-method generation. Pyrefly recognizes `@attr.s`, `@define`, `@frozen`, and their variants, handling field specifiers (`attr.ib()`, `field()`), converters, validators, defaults, and private-field aliasing. See the new [attrs documentation](https://pyrefly.org/en/docs/attrs/) for details.
- `isinstance` narrowing now consumes dynamic uncertainty before intersecting with the target type, so `x: Any` narrowed by `isinstance(x, C)` refines to `C` instead of preserving the `Any`. This closes a type-safety hole while sacrificing the gradual guarantee in exchange for more precise runtime-evidence-based narrowing.
- TypedDict `.get()` and `.pop()` methods with literal defaults now preserve the field type when the default is assignable to it, so `.get("x", "b")` on a `Literal["a", "b"]` field returns `Literal["a", "b"]` instead of `Literal["a", "b"] | str`.
- Unannotated class attributes are now inferred by unioning the types of all assignments in the constructor, instead of taking the type of the first assignment.

### Language Server

- Baselined errors (stored in a separate baseline file) now appear as hints in the IDE instead of errors, making it easier to distinguish new issues from known technical debt.
- Document symbol search now works in editors like Helix that don't support hierarchical symbols, returning a flat `SymbolInformation` list with dotted container names.
- Semantic tokens are now emitted for names bound by `with ... as` and `except ... as`, so those variables are syntax-highlighted consistently with other local bindings.
- Go-to-definition on non-Python source files (e.g., `.thrift`) now navigates directly to the symbol definition via text search instead of the top of the file, and works for nested attribute/enum member access.
- The unused-type-ignore rule detects `# type: ignore` comments that suppress no errors, helping clean up stale suppressions (off by default; enable with `unused-type-ignore = "error"` in `pyrefly.toml`).
- Cross-file diagnostics now refresh on save in strict-spec LSP clients like Zed, which previously needed a language-server restart to pick up changes in imported files.

### Stubgen

- Multi-line parenthesized annotations, values, and return types are now re-wrapped in `(...)` when emitted, preventing `IndentationError` in the generated stubs.
- Callable-typed values with differing return types across overloads are now rendered as `Callable[..., Incomplete]` instead of bare `Incomplete`, preserving callability information.
- Implicit class variables (bare assignments in a class body) are now annotated as `ClassVar[...]` in emitted stubs, correctly distinguishing them from instance attributes.
- `__all__` is now preserved verbatim when it's a static list/tuple literal, so re-exports survive in the stub.
- Async generators are emitted as plain `def` with `AsyncGenerator` return annotations instead of `async def`, following the typeshed convention and fixing coroutine-vs-generator confusion.

### Tensor Shape Types

- The `pyrefly-shape-extensions` and `pyrefly-torch-stubs` packages are now published to PyPI alongside Pyrefly releases, providing runtime helpers and shape-aware PyTorch stubs for tensor shape type checking.
- The shape DSL now supports `D[N]` or `D(N)` runtime wrappers for dimension expressions, allowing evaluated annotations to survive Python runtime while preserving symbolic arithmetic.
- `assert_shape` runtime helper validates tensor shapes at runtime, falling back to rank-only checks for symbolic shapes.

---

## 🐛 Bug fixes

We closed **20** bug issues this release 👏

- **#3867:** Fixed a regression where `isinstance` narrowing silently stopped working after a sibling branch in the same if/elif/else chain narrowed a different variable and terminated. The lazy-builtins optimization incorrectly lifted `isinstance` into the fork base, producing a degenerate Phi that resolved to `Never` and broke downstream narrowing.
- **#3893:** Fixed a stack overflow crash when checking self-referential protocols with overloaded methods whose `self:` annotations reference the protocol itself. A new per-thread cycle guard prevents infinite recursion during overload filtering.
- **#3841:** Unpacking a TypeVar bounded by a tuple (e.g., `Z: tuple[str, int]`) now preserves positional element types instead of collapsing to a union. `u, v = x` where `x: Z` now reveals `u: str, v: int` instead of `u: int | str, v: int | str`.
- **#3787:** TypedDict `.get(key, "literal")` on a non-required key now returns the field type when the literal default is assignable to it, instead of widening to `field_type | str`.
- **#3561:** Functions decorated with `@no_type_check` no longer emit `unannotated-return` diagnostics, since the decorator explicitly opts the function out of type checking.
- **#3900:** Dataclass and Pydantic BaseModel fields named `"self"` are no longer incorrectly flagged with `bad-keyword-argument` and `bad-argument-type` on construction. The instance receiver is now named `__dataclass_self__` when a field named `self` exists.
- **#3881:** SQLAlchemy declarative base classes with `kw_only=True` now correctly apply the keyword-only constraint to subclass fields, fixing false "field without default may not follow field with default" errors.
- **#2858:** Match subjects without a real narrowing name (e.g., `match f(x):`) now carry direct fallthrough narrows across cases, and the final capture forwards to the already-evaluated subject narrowed by previous cases.
- **#3445:** Packages with a PEP 561 `py.typed` marker no longer trigger `untyped-import` recommendations, even when imported as submodules.
- **#3954:** The `typings/` directory under the config root is now auto-discovered on the CLI path (not just in the IDE), fixing a divergence where CLI and IDE had different stub search paths for the same config.
- And more! #3879, #3891, #3293, #3928, #3688, #3890, #3912, #3945, #3926, #3924

Thank-you to all our contributors who found these bugs and reported them! Did you know this is one of the most helpful contributions you can make to an open-source project? If you find any bugs in Pyrefly we want to know about them! Please open a bug report issue [here](https://github.com/facebook/pyrefly/issues).

---

## 📦 Upgrade

```bash
pip install --upgrade pyrefly==1.2.0-dev.1
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

@stroxler, @shobhitmehro, @yangdanny97, @jorenham, @grievejia, generatedunixname2066905484085733, @connernilsen, @asukaminato0721, @samwgoldman, generatedunixname949130641157030, @kinto0, @NathanTempest, @ericzhonghou, @rchen152, @javabster, @QEDady, @mikeleppane, @markselby9, @aodihis, @arthaud, @DibbayajyotiRoy, Myles Matz, @ndmitchell, David Tolnay, @Kartikey077, pratved64, @maggiemoss, @goutamadwant, @kz357, @nitishagar, @hrolfurgylfa, @olekuhlmann, @JakobDegen, @lesbass, @fangyi-zhou, @ibraheemshaikh5, @alexander-beedie, @xaskii

---

*Please note: These release notes summarize major updates and features. For brevity, not all individual commits are listed.*
