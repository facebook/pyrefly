*Release date: July 24, 2026*

> **About dev releases**
> Dev releases (versions like `X.Y.Z-dev.N`) are non-stable snapshots cut periodically from trunk. They give early adopters a chance to try in-progress features and surface issues before the next stable release, but they don't carry the same stability or compatibility guarantees as a stable release — don't pin production projects to a dev version.

Pyrefly v1.2.0-dev.3 bundles **250 commits** from **29 contributors**.

---

## ✨ New & Improved

### Type Checking

- TypedDict classes now synthesize `__required_keys__` and `__optional_keys__` attributes during attribute lookup, so protocols requiring these attributes (like those in langgraph) work correctly.
- Attrs fields can now use the `@<field>.converter` decorator (added in attrs 26.2.0) to define converters, and the converter's input type is correctly used for the `__init__` parameter.
- `attrs.converters.pipe` now types the `__init__` parameter as the first converter's input type instead of `Any`, so invalid constructor arguments are caught.
- `attrs.converters.default_if_none` now types the `__init__` parameter as `<field type> | None` instead of `Any`, matching the converter's runtime behavior.
- Generic function converters in attrs fields (like `copy.deepcopy`) now solve their type variables against the field's declared type, so the `__init__` parameter is correctly typed.
- Bare generic class converters in attrs fields now infer their element type from the field annotation, so `converter=tuple` on a `Sequence[int]` field types the parameter as `Iterable[int]`.
- Polars DataFrames constructed from dictionary literals now infer a schema tracking each column's name and type, so later column access is type-checked.
- Polars `.select()`, `.drop()`, `.rename()`, and `.with_columns()` operations now preserve and transform the DataFrame schema, keeping column information through method chains.
- Polars `.filter()`, `.sort()`, and `.fill_null()` operations now preserve the DataFrame schema unchanged, since they only affect rows.
- Pandas DataFrames are now tracked with partial schemas (marked with `...` in display), so known columns are type-checked while avoiding false positives from dynamic column operations.
- `functools.partial` now type-checks bound arguments and synthesizes a precise residual signature for the remaining parameters, catching errors at construction and call sites.
- `functools.partial` over generic functions now preserves type variables, so bound arguments can solve them and the residual stays generic.
- `functools.partial` over overloaded functions now resolves to the matching overload branches, so later calls still benefit from overload resolution.
- `functools.partial` over class constructors and bound methods now works correctly, checking arguments and producing the expected instance type.
- `functools.partial` results now expose `.func`, `.args`, and `.keywords` attributes and narrow correctly in `isinstance` checks.
- `functools.partial` over `**Unpack[TypedDict]` splats now expands the TypedDict fields and checks the residual parameters correctly.
- Enum `.value` on an enum type (not a specific member) now infers as the union of member literal values instead of widening to the mixin type.
- Positional class patterns now narrow matched attributes correctly, resolving the attribute name from `__match_args__` at solve time.
- Sequence pattern element captures now read from the narrowed subject type, so sibling element constraints refine the captured types.
- `isinstance` checks on facets (like tuple elements or union members) now filter the parent union to matching members.
- Unpacking variadic tuples with fixed prefix/suffix now types starred captures precisely instead of smearing the fixed elements into the star.

### Language Server

- Misplaced file-level `# pyrefly: ignore-errors` directives (placed after code) now emit a `misplaced-ignore` warning pointing users to move them to the top.
- Module auto-import completions no longer suggest duplicate entries for modules already imported with regular import statements.
- Overloaded class methods now associate the implementation's docstring with the overload chain, so hover shows documentation at call sites.
- Import alias names (`import x as y`, `from x import y as z`) are now classified as definition contexts, so completions don't suggest module paths for them.
- Completions for import aliases now preserve the original exported name and generate correct `from model import MyModel as MyModelAlias` statements.
- Inlay hints now debounce server-side (default 150ms) to prevent width jitter while typing, settling only when editing pauses.
- Keyword-argument hover now resolves parameter types from the defining module and falls back to the selected call signature when refinement is unavailable.
- Hover on `and`/`or` operators now returns the enclosing boolean expression range for better context.
- Renaming a Protocol class now renames the class symbol instead of incorrectly renaming `__init__`.
- Selection ranges (`textDocument/selectionRange`) now follow AST nesting from expression to statement to scope to document, with support for `#region`/`#endregion` markers.
- `#region`/`#endregion` folding markers are now recognized (including nested regions and spaced `# region`), independently of comment-section folding.
- TSP type queries (`getComputedType`, `getDeclaredType`, `getExpectedType`) now resolve correctly for files that aren't open in the editor.
- Pyrefly client settings (`pyrefly.*`) are now ignored when running as a type server (TSP mode), so stale extension settings don't interfere with Pylance.

### Error Reporting

- A new `unknown-column` error kind flags reads of DataFrame columns that don't exist in the known schema.
- A new `column-type-mismatch` error kind flags Polars column literals whose elements cannot share one dtype.
- A new `implicit-reexport` error kind (default: ignore) flags imports of names that are only available through plain `import`/`from ... import` (not explicit re-exports).
- A new `invalid-type-checking-constant` error kind flags user-defined `TYPE_CHECKING` constants that aren't assignable to `bool`.
- A new `missing-super-call` error kind (default: ignore) flags `__init__`, `__new__`, or `__init_subclass__` methods that override a non-object parent without calling the parent method.
- A new `empty-body` error kind (default: ignore) flags ellipsis-only function bodies where `None` isn't assignable to the annotated return type.
- A new `unused-call-result` error kind (default: ignore) flags discarded function/method call results whose type is informative (not `None`, `Any`, or `Never`).
- A new `invalid-abstract-method` error kind (default: ignore) flags `@abstractmethod` decorators in non-abstract classes.
- Error summary output is now sorted by count then name, so ties break deterministically and the output is stable across runs.
- The `full-text-with-github` output format now emits readable diagnostics followed by GitHub workflow commands, so source locations remain visible in raw logs.

### Tensor Shape Typing

- Many improvements. Still early alpha state.

### Polars & Pandas Support

- Polars `DataFrame` construction from dictionary literals now infers a complete schema with column names and types.
- Polars column reads by string literal now error on unknown columns, catching typos before runtime.
- Polars list subscripts (like `df[["a", "b"]]`) now narrow the schema to the selected columns in the requested order.
- Polars `.select()` narrows the schema to the named columns in the order given.
- Polars `.drop()` narrows the schema by removing the named columns while preserving the original order.
- Polars `.rename()` relabels columns in the schema, handling swaps correctly and falling back on ambiguous cases.
- Polars `.with_columns()` adds or overwrites columns in the schema, appending new names and preserving untouched columns.
- Polars selector strings (like `"*"` and regex patterns) are now recognized, so `.select("*")` and `.drop("^col.*$")` fall back instead of erroring.
- Pandas DataFrames are tracked with partial schemas (kind: Pandas, completeness: Partial), so known columns are checked without false positives from dynamic operations.
- Polars column literals now type each column from its first element and error on incompatible mixes, matching Polars runtime behavior.

### Configuration & Tooling

- `pyrefly coverage check --public-only` now keeps public re-exports from excluded files, so symbols are only excluded when all public re-export locations are excluded.
- `pyrefly coverage` commands now accept a `[coverage]` config table with `includes` and `excludes` that override project globs, so coverage can be measured on a different file set.
- `pyrefly coverage check --fail-under` now always prints findings regardless of whether the threshold is met, so users see what's missing even when passing.
- `pyrefly coverage` now excludes privately-named classes (like `_Foo`) unless exported via `__all__`, consistent with other symbol kinds.
- Config files without a `[tool.pyrefly]` section are now only considered root markers, so `pyproject.toml` files don't shadow parent configs unless they contain Pyrefly settings.
- TOML parse errors now point at the offending value instead of the start of the file, so users can find the mistake quickly.
- The `--project-excludes` flag now takes precedence over `coverage.excludes`, and positional `FILES...` args also override coverage globs.
- Module handles for files without a config now apply the `fallback_search_path`, so `__unknown__` files count under `--public-only`.

---

## 🐛 Bug fixes

We closed **48** bug issues this release 👏

- **#4071:** Fixed incorrect `bad-override` error when a child class generic over a `TypeVarTuple` overrides a parent method with the same `*args: *Ts` signature.
- **#3745:** TypedDict classes now synthesize `__required_keys__` and `__optional_keys__` attributes, so protocols requiring these attributes (like those in langgraph) no longer fail structural checks.
- **#4109:** Module auto-import completions no longer suggest duplicate entries for modules already imported with regular import statements (like `import json`).
- **#3619:** Overloaded class methods now associate the implementation's docstring with the overload chain's first declaration, so hover shows documentation at call sites.
- **#3437:** Fixed stack overflow when checking recursive enum classes by removing the recursive Flag subtype check and moving Flag detection downstream.
- **#3354:** Fixed panic with fuzzed NamedTuple code by adding a distinct binding key for invalid assignment targets, preventing collisions with synthesized functional NamedTuple bindings.
- **#3240:** Constructor hover types are now resolved before rendering, preventing internal `@NNN` variables from leaking into tooltips.
- **#1323:** `enumerate()` over tuple literals now preserves element types instead of widening to a generic tuple, so `for i, x in enumerate(("a", "b"))` types `x` as `Literal["a"] | Literal["b"]`.
- **#3305:** `#region`/`#endregion` folding markers are now recognized (including nested regions and spaced `# region`), independently of comment-section folding.
- **#4155:** Generic function converters in attrs fields (like `copy.deepcopy`) now solve their type variables against the field's declared type, so the `__init__` parameter is correctly typed instead of leaking a bare `TypeVar`.
- And more! #4123, #4125, #4127, #4128, #3330, #3638, #3546, #4107, #3329, #2495, #4034, #4082, #3900, #3435, #3701, #3056, #633, #3857, #3814, #3154, #3085, #2817, #1770, #1689, #1510, #964, #1866, #1084, #4164, #3821, #3727, #3669, #3368, #2504, #2235, #1681, #1663, #1373

Thank-you to all our contributors who found these bugs and reported them! Did you know this is one of the most helpful contributions you can make to an open-source project? If you find any bugs in Pyrefly we want to know about them! Please open a bug report issue [here](https://github.com/facebook/pyrefly/issues).

---

## 📦 Upgrade

```bash
pip install --upgrade pyrefly==1.2.0-dev.3
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

@stroxler, @shobhitmehro, generatedunixname2066905484085733, @yangdanny97, @asukaminato0721, @kinto0, @anushamukka-dev, @samwgoldman, @rchen152, @NathanTempest, @jorenham, @aodihis, @tobyh-canva, @brosenfeld, @nitishagar, @goutamadwant, @grievejia, @roian6, @paranoa233, @kz357, @teerthsharma, generatedunixname949130641157030, @coclique, @SarahMKosowsky, @markselby9, @oprypin, @knQzx, @alexander-beedie, @ak4-sh

---

*Please note: These release notes summarize major updates and features. For brevity, not all individual commits are listed.*
