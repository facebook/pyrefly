*Release date: August 06, 2026*

> **About dev releases**
> Dev releases (versions like `X.Y.Z-dev.N`) are non-stable snapshots cut periodically from trunk. They give early adopters a chance to try in-progress features and surface issues before the next stable release, but they don't carry the same stability or compatibility guarantees as a stable release — don't pin production projects to a dev version.

Pyrefly v1.3.0-dev.1 bundles **162 commits** from **23 contributors**.

---

## ✨ New & Improved

### Type Checking

- You can now use `Protocol` as an instance of `_ProtocolMeta` in type checks, and Pyrefly correctly models Protocol using typeshed's `_ProtocolMeta`, making it assignable to `ABCMeta`.
- Classes that inherit from `Sequence` or other abstract base classes now correctly report when abstract methods remain unimplemented, even when the class doesn't directly extend `ABC`.
- Attribute access on `type[type[X]]` now uses the metaclass for more precise results instead of triggering internal errors.
- TypeVars explicitly bounded by `Any` now preserve dynamic attribute access instead of converting to `object`, fixing false missing-attribute errors.
- Legacy TypeVars declared in an enclosing class are now correctly flagged as out-of-scope when referenced inside nested classes, and type aliases that capture enclosing-scope TypeVars are also flagged.
- Out-of-scope legacy TypeVars in subscripted generics (e.g., `list[T]()` where `T` is unbound) now emit errors at the call site.
- Metaclass conflict detection is now more lenient, choosing a legal metaclass when multiple are available through inheritance, and redundant `metaclass=type` declarations are now allowed.

### Language Server

- Inlay hints now include the imports they depend on, so accepting a type hint also inserts the necessary `import` statements automatically.
- Type annotations inserted from inlay hints now render `Unknown` as `typing.Any` to produce valid Python source.
- Inlay hints use `from foo import X` style imports by default when possible, falling back to `import foo` only when the module is already imported that way.
- Hover on a `case _` wildcard now resolves to the match subject expression's type, not just the first token of the subject.
- Method-name completion no longer returns duplicate entries while defining methods.
- Positional inlay hints now display correctly even after keyword arguments.
- Keyword completion now works for `**kwargs: Unpack[TypedDict]` parameters, exposing each TypedDict field as a completable keyword argument.
- Completion suggestions now filter out TypedDict keys that aren't valid Python identifiers or are reserved keywords.
- Parameter-name hints now appear for callable instances by coercing them to their bound `__call__` signature.
- Bare generic classes now preserve class-scoped type parameters when converting `__new__` to a callable, fixing constructor argument inference.
- Inlay hints are now clickable, allowing you to navigate to class and type definitions directly from hint overlays.

### Polars and pandas DataFrame Support

- `df.with_columns(...)` now infers each added or overwritten column's dtype from its Polars expression instead of leaving it `Unknown`.
- `df.select(...)` produces a precise column schema when its arguments are Polars expressions, not only plain column-name strings.
- `df.group_by(...).agg(...)` calls now infer a precise column schema, including grouping keys and aggregation outputs.
- `pl.Series("a", [1])` now reveals `Series[Int64]` instead of the opaque `Series` class, carrying element dtypes forward.
- `df.get_column(name)` and `df.to_series(index)` now return typed `Series[dtype]` instead of an opaque `Series`.
- Polars DataFrames built with a `schema=` whose column set doesn't match the data now report a clear `column-schema-mismatch` error.
- Column names, element dtypes, construction flags, and join strategies are now resolved from their inferred types (e.g., `Final` or `Literal[str]`) instead of requiring bare literals.
- DataFrame columns built from variables or call results now infer their dtype from the value's type, not just from literals.
- DataFrames built from a `TypedDict`-typed value now infer columns from the TypedDict fields.
- `df.lazy()` and `df.collect()` now preserve the column schema across the conversion.
- `pl.DataFrame[Schema]` annotations now read the schema class's columns, so annotated parameters carry their schema into the function body.
- `pl.read_csv` and `pl.scan_csv` with inline `schema=` dictionaries now infer complete CSV schemas, and reader options like projections and overrides are applied.
- `df.fill_null` now models integer and float widening according to Polars runtime behavior.
- `df[[]]` (empty list subscript) now preserves the original schema instead of producing an empty schema.
- Duplicate Polars output columns are now reported with a new `duplicate-column` error.
- pandas DataFrames built with `columns=` now project the data onto the given column set and order.
- Column inference never reports a false error on correct Polars or pandas code.

### Django Support

Pyrefly now understands reverse relationships between Django models *in the same file*:
- Reverse managers are added to the target of each `ForeignKey`, with the default `<source>_set` attribute or a custom `related_name`.
- Reverse `OneToOneField` relationships expose the source model directly on the reverse side.
- Reverse `ManyToManyField` relationships add a manager with the correct model type.
- Django `related_name` placeholders (class and app-label) are filled in, and disabled or malformed names create no attribute.

### Error Suppressions and Diagnostics

- `dataclass_transform` now rejects unsupported keyword parameters, as required by the typing specification.
- `mock.patch` string targets are now validated, reporting missing imports or attributes at the string literal location.
- Missing attributes from `mock.patch` targets are reported as warnings under a new `missing-attribute-patch-target` error kind.
- Unused-ignore diagnostics now underline the suppression comment's `#` character, ensuring a valid UTF-8 boundary even after multibyte characters.

### CLI and Configuration

- You can now treat `ALL_CAPS` variables as `Final` by enabling the `treat-all-caps-as-final` configuration option, which reports reassignment errors.
- The `--typeshed-path` option now works correctly when checking typeshed's own stubs, deriving module names from `<typeshed_path>/stdlib`.

---

## 🐛 Bug fixes

We closed **14** bug issues this release 👏

- **#4077:** Fixed an issue where attributes available only through `__getattr__` or `__getattribute__` were incorrectly rejected in protocol subtyping. Pyrefly now allows these fallback methods to satisfy protocol members, with special handling for class objects and certain dunders.
- **#4126:** Method-name completion was combining dedicated class/magic-method candidates with ordinary value completions, causing existing methods to appear twice. Method definitions now stay on the dedicated completion paths.
- **#2797:** Classes inheriting from `Sequence` or other abstract base classes now correctly report unimplemented abstract methods, even when the class doesn't directly extend `ABC`.
- **#4437:** Fixed a stack overflow crash when checking code with a recursive `__new__` method. Pyrefly now detects direct recursive `__new__` targets.
- **#4459:** Fixed a panic (`Key Anon already exists`) when a `NamedTuple` was defined inline inside a `match` statement. A dedicated `MatchSubject` binding key now prevents collisions.
- **#385:** Pyrefly now checks whether a context manager suppresses exceptions when doing type narrowing, so a `raise` inside a context manager that returns `True` from `__exit__` no longer incorrectly narrows the type.
- **#4453:** Fixed a panic when a class named `classproperty` or `cached_classproperty` had no type arguments. The match arm now checks for type arguments before unwrapping.
- **#4317:** Redundant boolean conditions (e.g., `A and True`) no longer introduce placeholder narrowing operations, preserving the original narrowing behavior.
- **#4444:** Fixed a panic when writing output reports for files containing multibyte characters followed by `# type: ignore` comments. Unused-ignore diagnostics now underline the ASCII `#` character.
- **#4377:** Callable instances now show parameter names in inlay hints by coercing them to their bound `__call__` signature.
- **#4379:** `@total_ordering` now recognizes non-synthesized rich comparison methods inherited through the class MRO, fixing false positives when `__lt__` is defined in a base class.
- **#3653:** Fixed exponential blowup for nested calls containing container literals. Nested expressions are now inferred to a `Type` once up front, collapsing shared subtrees.
- **#4405:** TypeVars explicitly bounded by `Any` now preserve dynamic attribute access instead of converting to `object`, fixing false missing-attribute errors.
- **#4267:** Bare generic classes now preserve class-scoped type parameters when converting `__new__` to a callable, fixing constructor argument inference.

And more! #4430, #4417, #4387, #4315, #3678, #3334, #4380, #2313, #1130, #4388, #4386, #4390, #4399, #4443, #4455, #4457, #4464, #4447, #4436, #4441, #4442, #4451, #4452, #4456, #4458, #4460, #317, #1576, #4331, #4304, #4393, #4361

Thank-you to all our contributors who found these bugs and reported them! Did you know this is one of the most helpful contributions you can make to an open-source project? If you find any bugs in Pyrefly we want to know about them! Please open a bug report issue [here](https://github.com/facebook/pyrefly/issues).

---

## 📦 Upgrade

```bash
pip install --upgrade pyrefly==1.3.0-dev.1
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

@shobhitmehro, @rchen152, @asukaminato0721, generatedunixname2066905484085733, @yangdanny97, @grievejia, @kinto0, generatedunixname949130641157030, @lyydsheep, @ndmitchell, @stroxler, generatedunixname89002005232357, @connernilsen, @jcarreiro, @patrickswedish, @randolf-scholz, @vincevannoort, @renz011tzar, @ak4-sh, @tobyh-canva, @KotlinIsland, @Sanjays2402, @alexander-beedie

---

*Please note: These release notes summarize major updates and features. For brevity, not all individual commits are listed.*
