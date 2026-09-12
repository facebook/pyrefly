*Release date: July 10, 2026*

> **About dev releases**
> Dev releases (versions like `X.Y.Z-dev.N`) are non-stable snapshots cut periodically from trunk. They give early adopters a chance to try in-progress features and surface issues before the next stable release, but they don't carry the same stability or compatibility guarantees as a stable release — don't pin production projects to a dev version.

Pyrefly v1.2.0-dev.2 bundles **158 commits** from **23 contributors**.

---

## ✨ New & Improved

### Type Checking

- Functions decorated with `functools.singledispatch` now type-check calls using the signature of the function you decorated, catching argument errors like `f("bad")` when that function expects `int`. Registered implementations are also checked against that function's first parameter, so their dispatch types must be compatible.
- Generic `singledispatch` functions now infer their generic types from call arguments instead of reporting `Unknown`.
- Pyrefly can now warn when a function declared to return a concrete type returns `Any`, with separate diagnostics for explicit and inferred `Any` returns. These warnings are off by default for backward compatibility; you can enable them explicitly, or let Pyrefly migrate the setting from mypy when `warn_return_any` is enabled.
- `isinstance(x, type)` now preserves type arguments when narrowing unions, so `type[int] | str` narrows to `type[int]` instead of bare `type`.
- Class decorators annotated as returning `type[Any]` now make the decorated class behave dynamically, matching decorators that can substantially change the class at runtime.
- Class methods can now access generic class attributes through `cls`, fixing false `missing-attribute` errors on abstract base classes.
- Membership checks with `Any`/unknown items now invoke the container's `__contains__` method and return `bool` instead of `Unknown`.
- `copy.replace` is now type-checked like dataclass replacement, so Pyrefly validates immutable updates more accurately.

### Language Server

- Hover tooltips for union methods now look at each union member, so `x: A | B; x.foo()` correctly shows `foo` as a method when both `A.foo` and `B.foo` are methods.
- Hovering over augmented assignment operators like `+=` now shows the `__iadd__` definition and type, or the right-hand side literal type when that is the best available information.
- Attribute hovers like `c.x` now show where the attribute's type was narrowed instead of unrelated first-use info.
- Hover and type lookups now work correctly in notebook cells after the first.
- Inlay hints, document symbols, references, and diagnostic grouping now work correctly in notebook code cells that follow markdown cells.
- Auto-import completions now respect the `python.analysis.autoImportCompletions` setting, skipping auto-imports when disabled.
- Value completions are now skipped when typing a keyword-argument name (e.g., `func(foo=1, ba|)`), and statement keywords like `try`/`while` are no longer offered in expression contexts.
- Deprecated stdlib typing aliases (e.g., `typing.List`) are now ranked below their `collections.abc` equivalents in auto-import completions.

### Bazel Integration

- A new, experimental `bazel-check` command adds Bazel-focused type-checking for projects that provide source information through JSON manifests. It is intended to collaborate with an in-progress `rules_pyrefly` Bazel rule implementation.

### Tensor Shape Types

- Shape-aware stubs can now express methods whose types follow another method on the same object. This lets `nn.Module.__call__` type-check like the module's `forward` method.
- `SizeTuple` is now the public way to write shape parameters, with support for compact syntax like `ndarray[[2, 3], dtype]` and unpacking with `*Elements[SizeTuple]`.
- Torch and NumPy shape stubs now use `SizeTuple`, making it easier to model APIs involving multiple tensors with different shapes.
- Symbolic shape variables now use `SymVar`, written as `N: SymVar` in PEP 695 syntax. Shape arithmetic is now limited to these symbolic shape variables.

---

## 🐛 Bug fixes

We closed **26** bug reports this release 👏

- **#897:** Fixed an issue where attribute narrowing with optional values failed to preserve the narrowed type across conditional branches, causing false positives when accessing attributes after a null check.
- **#3585:** Fixed `isinstance(value, type)` narrowing for `type[int] | str` unions, which previously collapsed to bare `type` instead of preserving `type[int]`.
- **#3584:** Fixed `isinstance(value, type)` narrowing for `TypeForm[object]`, which previously produced `type & TypeForm[object]` instead of the correct `type[int]` after an `issubclass` check.
- **#2689:** Fixed subsequent `isinstance(cls, type)` and `issubclass(cls, upper_bound)` checks erasing `TypeVar` information, so `type[F]` is preserved through assertion chains instead of widening to the bound.
- **#3806:** Added `no-any-return` error kinds to detect returning `Any` from functions declared to return concrete types, with explicit and implicit variants and support for migrating mypy's `warn_return_any` setting.
- **#4024:** Fixed `pyrefly coverage check --strict` incorrectly labeling fully-`Any` symbols (e.g., `def f(x: Any) -> Any`) as `coverage-partial` instead of `coverage-missing`.
- **#4062:** Fixed `pytorch-efficiency-lints` not recognizing `Tensor` in common stub layouts.
- **#4083:** Fixed `tuple(xs)` losing track of tuple length when `xs` is already a tuple, now preserving known shapes like `tuple[int, int]` instead of widening to `tuple[int, ...]`.
- **#4093:** Fixed an internal error when accessing `__class__.__setattr__` on a class object, caused by missing metaclass handling.
- **#4073:** Fixed unpacked `TypeVarTuple` arguments like `tuple[int, *Ts]` not being assignable to varargs type `tuple[int, *Ts]` due to an internal mismatch in how Pyrefly represented the tuple types.
- And more! #4025, #4019, #4020, #4021, #3730, #3732, #3997, #4018, #4023, #3949, #4022, #3920, #4092, #3078, #3960, #4064

Thank you to all our contributors who found these bugs and reported them! Did you know this is one of the most helpful contributions you can make to an open-source project? If you find any bugs in Pyrefly, we want to know about them! Please open a bug report issue [here](https://github.com/facebook/pyrefly/issues).

---

## 📦 Upgrade

```bash
pip install --upgrade pyrefly==1.2.0-dev.2
```

### How to safely upgrade your codebase

Upgrading the version of Pyrefly you're using or a third-party library you depend on can reveal new type errors in your code. Fixing them all at once is often unrealistic. We've written scripts to help you temporarily silence them. After upgrading, follow these steps:

1. `pyrefly check --suppress-errors`
2. Run your code formatter of choice
3. `pyrefly check --remove-unused-ignores`
4. Repeat until you achieve a clean formatting run and a clean type check.

This will add `# pyrefly: ignore` comments to your code, enabling you to silence errors and return to them later. This can make the process of upgrading a large codebase much more manageable.

Read more about error suppressions in the [Pyrefly documentation](https://pyrefly.org/en/docs/error-suppressions/).

---

## 🖊️ Contributors this release

@stroxler, @grievejia, @shobhitmehro, @asukaminato0721, @jorenham, @kinto0, @yangdanny97, David Tolnay, @ericzhonghou, @connernilsen, @qnox, @mikeleppane, @dillydill123, @arthaud, @alexander-beedie, @NathanTempest, @fhoehle, @WilliamK112, @xaskii, @durvesh1992, @lennardwalter, @kz357, @nitishagar

---

*Please note: These release notes summarize major updates and features. For brevity, not all individual commits are listed.*
