*Release date: September 10, 2026*

Pyrefly v1.3.0 bundles **934 commits** from **71 contributors**.

---

## Release Highlights

### Type Checking

- **New diagnostics and more accurate type inference.** Pyrefly now catches invalid literal regular expressions, invalid `mock.patch` targets, unsupported `dataclass_transform` arguments, incompatible `Protocol.__call__` overrides, and unimplemented abstract methods. Pattern-match exhaustiveness, overload resolution, narrowing, and generic inference are also more accurate.
- **Better support for major Python frameworks.** Pyrefly understands same-file Django reverse relationships, checks SQLAlchemy updates against model fields, and recognizes attributes registered on PyTorch modules.

### Language Server

- **Search the whole workspace for methods and nested definitions.** Workspace symbol search now covers methods, nested classes and functions, and class attributes even in unopened files.
- **New editor refactorings and quick fixes.** Change Signature updates a function and its call sites together, while new quick fixes remove unused imports and insert `assert x is not None`. Inlay hints can also insert required imports and link to definitions.

### Configuration & CLI

- **Target individual Pyrefly errors with standard suppression comments.** `# type: ignore[pyrefly:<code>]` suppresses a specific Pyrefly diagnostic without hiding unrelated errors on the same line.
- **Choose how Pyrefly handles untyped dependencies.** The new `--replace-untyped-imports-with-any` option replaces selected third-party packages that lack stubs or a `py.typed` marker with `Any`, and `pyrefly init` translates mypy's `follow_untyped_imports` setting automatically.
- **Baseline files are easier to maintain and review.** Baselines can match by concise description instead of source position, use a compact format, show existing errors at reduced severity, remove stale entries with `--prune-baseline`, and reject them in CI with `--error-stale-baseline`.

### Experimental Extensions

- **Much broader shape-aware JAX and NumPy support.** New JAX stubs cover array creation, manipulation, and linear algebra, while the new `pyrefly-numpy-stubs` package brings shape checking to NumPy. These stubs use a new type-level shape DSL, which replaces the old `@shaped_array` API.
- **Expanded Polars and pandas DataFrame schema support.** Pyrefly tracks Polars schemas through common DataFrame transformations, with support for typed `Series` and schema annotations. pandas `columns=` projections now preserve the requested schema as well.

---

## ✨ New & Improved

### Type Checking

- Pattern matching gained stronger exhaustiveness checking, including tuple subjects and open types such as unions. Open-type exhaustiveness now has its own configurable `non-exhaustive-match-open-type` error kind.
- Overload selection now follows the latest typing specification more closely, producing a safe common return type for gradual arguments and reducing false positives in complex overloads.
- Type inference and narrowing are more precise for membership tests, equality checks, `hasattr`, wide `Literal` unions, callable values, reverse tuple slices, and values assigned from `Any`.
- Dataclass fields backed by descriptors are now checked for incompatible read and write types under the new `bad-dataclass-descriptor` error kind.
- String targets passed to `unittest.mock.patch` are validated, with nonexistent attributes reported as `missing-attribute-patch-target` warnings.
- Literal regular expressions are checked for invalid patterns and capturing groups under the new `regex` error kind.
- Django support now understands reverse `ForeignKey`, `OneToOneField`, and `ManyToManyField` relationships in the same file, while Django REST Framework serializers avoid false override errors for common `Meta` and field patterns.
- SQLAlchemy `update().values()` checks values against mapped model fields.
- PyTorch modules recognize attributes registered through `register_buffer` and `register_parameter`.

### Language Server

- Workspace symbol search now includes methods, nested classes, nested functions, and class attributes.
- Cross-file call hierarchy, type hierarchy, and find-references now work without first opening every relevant file.
- A new change-signature refactoring updates a function's parameters and its call sites together.
- New quick fixes remove unused imports and insert `assert x is not None` when an optional value needs narrowing.
- Inlay hints can add required imports, are clickable for navigation, and now handle callable objects, instance methods, positional arguments, and `**kwargs: Unpack[TypedDict]` more accurately.
- Rename and navigation are more reliable for relative imports, decorated functions, named dataclass and Pydantic arguments, operators, module paths in strings, and legacy type-parameter declarations.
- Custom `pyrefly.lspPath` values now work correctly with Windows, home-relative, and workspace-relative paths.
- LSP clients can provide `extraSearchPaths` and `extraProjectExcludes` during initialization, improving integration with editors that manage their own import paths and excluded directories.
- Editors can display custom build-system activity and failures through the new `pyrefly/typeErrorDisplayStatusChanged` notification.

### Performance

- TSP now reuses analysis for unopened files instead of solving the same module for every request. In a captured Pylance session, 22,294 unopened-file `getComputedType` requests fell from 668 seconds to 3.4 seconds, while total request time fell from 671 seconds to 6.4 seconds.
- Unknown-name suggestions are dramatically faster in files with very large scopes. Pyrefly now skips suggestion work for diagnostics that will be discarded and rejects unlikely candidates before computing their full edit distance. A pathological internal case fell from 96.8 seconds to 0.62 seconds, while focused benchmarks improved by 15–1,390x.

### Configuration & CLI

- `# type: ignore[pyrefly:<code>]` comments suppress only the named Pyrefly diagnostic without hiding unrelated errors on the same line.
- The new `type-ignore-unknown-tag-behavior` option controls how tags belonging to other tools affect Pyrefly diagnostics: they can have no effect, downgrade diagnostics to warnings, or retain the legacy blanket-suppression behavior.
- The new `replace-untyped-imports-with-any` option replaces selected installed third-party packages with `Any` when they provide neither stubs nor a `py.typed` marker.
- `pyrefly init` automatically translates mypy's global and per-module `follow_untyped_imports` settings to the new option.
- `baseline-matching-mode` can match diagnostics by their column or concise description, reducing churn when unrelated code moves.
- `baseline-format` can write full metadata or a minimal representation containing only the fields needed for matching.
- `baseline-error-level` can expose matched diagnostics at `info`, `warn`, or `error` severity instead of hiding them, with baseline provenance included in text, JSON, SARIF, and summary output.
- `--prune-baseline` removes stale entries without recording new errors, while `--error-stale-baseline` lets CI reject a baseline containing obsolete entries.
- The opt-in `treat-all-caps-as-final` option treats reassignment of `ALL_CAPS` names as a `bad-assignment`.
- Projects can require a compatible Pyrefly version from `pyrefly.toml` using a PEP 440 version constraint.
- The new `python-interpreter-find-command` setting supports custom interpreter discovery.
- Multiple `--output` destinations with different formats can be specified in one CLI invocation.
- New documentation explains how to integrate Pyrefly with Pants through the `pants-pyrefly` plugin.

### Experimental Extensions

- A composable type-level DSL now describes tensor shape transformations, and the JAX, NumPy, and PyTorch stubs have migrated from the decorator-based V1 shape system to direct V2 signatures.
- Shape-aware JAX stubs now cover array creation, indexing, reductions, searching and sorting, FFT, linear algebra, einsum and other contractions, and most of `jax.lax`.
- The new `pyrefly-numpy-stubs` package brings shape-aware checking to NumPy alongside the existing PyTorch support.
- The legacy `@shaped_array` API has been removed; custom shape annotations must migrate to `IntTuple`-generic classes and the V2 DSL.
- Polars DataFrame analysis now tracks schemas through construction, `select`, `with_columns`, `group_by().agg()`, joins, CSV readers, and lazy/eager conversion, with typed `Series`, nested and owned dtypes, and schema information from variables, calls, and `TypedDict`s.
- PEP 593 `Annotated[DataFrame, Schema(...)]` supports exact and open schema contracts, while the new `column-schema-mismatch` and `duplicate-column` errors catch invalid schemas and conflicting output columns before runtime.
- pandas DataFrames built with `columns=` now project their inferred schema onto the requested column set and order.

---

## 🐛 Bug fixes

We closed **88** bug issues this release 👏

- **#3653:** Eliminated exponential work when nested calls contain container literals by inferring shared expression subtrees once.
- **#4678:** Fixed pathological check times exceeding a thousand seconds on small files containing very long non-ASCII lines.
- **#4437:** Fixed a stack overflow crash when checking code with a recursive `__new__` method. Pyrefly now detects direct recursive `__new__` targets.
- **#4459:** Fixed a panic when a `NamedTuple` was defined inline inside a `match` statement. A dedicated `MatchSubject` binding key now prevents collisions.
- **#4130:** Star imports from `py.typed` packages installed in `site-packages` now recognize the package's public names as re-exports.

Thank you to all our contributors who found these bugs and reported them! Did you know this is one of the most helpful contributions you can make to an open-source project? If you find any bugs in Pyrefly we want to know about them! Please open a bug report issue [here](https://github.com/facebook/pyrefly/issues).

---

## 📦 Upgrade

```bash
pip install --upgrade pyrefly==1.3.0
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

@shobhitmehro, @rchen152, @asukaminato0721, generatedunixname2066905484085733, @yangdanny97, @grievejia, @kinto0, generatedunixname949130641157030, @lyydsheep, @ndmitchell, @stroxler, generatedunixname89002005232357, @connernilsen, @jcarreiro, @patrickswedish, @randolf-scholz, @vincevannoort, @renz011tzar, @ak4-sh, @tobyh-canva, @KotlinIsland, @Sanjays2402, @alexander-beedie, David Tolnay, @javabster, @samwgoldman, @IBlackVoid, @nitishagar, @WilliamK112, @NathanTempest, @ytausch, @xaskii, generatedunixname1431085361989520, @a7or, @markselby9, @heejaechang, Willem Kokke, @dillydill123, @kakolla, generatedunixname1699489071355949, @ting-hong-shieh, @kavix, @Vishwaspatel2401, @danielgaskins, generatedunixname89002005307016, @fangyi-zhou, @MarcoGorelli, @thomaspolasek, @tague, @paranoa233, @d34db3ff, @AMR5210, @DarkNightForge, @ternaus, @austin3dickey, @cakeni, @lolpack, @jakevdp, @Pager-dot, @tkim602, @anishfyi, @mangeshraut712, @jorenham, @Pyxelate, @DetachHead, @kavyansh18, @rootkiller6788, @auscompgeek, Khan Mohammed, @KSAGlory, @devteamaegis

---

*Please note: These release notes summarize major updates and features. For brevity, not all individual commits are listed.*
