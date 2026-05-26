# Pyrefly v0.64.0
**Status : BETA**
*Release date: May 05, 2026*

Pyrefly v0.64.0 bundles **190 commits** from **20 contributors**.

---

## ✨ New & Improved

| Area | What's new |
|------|------------|
| **Type Checking** | - You can now pass generic or overloaded callables to higher-order functions and Pyrefly will preserve their structure in the return type. For example, `identity(identity)` now correctly returns a generic callable instead of degrading to `Unknown`. <br><br>- Overload structure is now preserved when passing overloaded functions to decorators or higher-order functions, with automatic pruning of incompatible branches based on solved type constraints. <br><br>- Same-scope class rebinds (like `Real = Dummy` after `class Real`) are now checked against the original class as if it were an implicit `type[Real]` annotation, preventing silent type changes and fixing spurious constructor-call errors. <br><br>- Generic classes with missing type arguments in lax mode now degrade to `Any` instead of raising variance errors, improving consistency with how we handle other incomplete types. <br><br>- Pydantic `field_validator` decorators with `mode='before'` and `mode='plain'` are now supported, allowing validators to accept broader input types before coercion. <br><br>- Spurious unpack diagnostics are no longer emitted when the right-hand side involves `Never` (e.g. `a, b, c = never()` or `a, b = (never(), 1)`). The unpack solver is now `Never`-aware, recognizing that the producing expression cannot complete and any error message at the unpack site would be misleading. <br><br>- `assert` statements now check that `__bool__` is callable on the test expression, matching the behavior already in place for `if`, `while`, and ternary expressions (and aligning with mypy and pyright). |
| **Language Server** | - The language server now advertises both `source.fixAll` and `source.fixAll.pyrefly` code action kinds, enabling selective fix-on-save configuration across editors that implement the LSP protocol. <br><br>- Document highlights now correctly distinguish between read and write references, setting `DocumentHighlightKind::WRITE` for assignments and declarations. <br><br>- Go-to-definition on relative imports in site-packages files now correctly resolves to the package source instead of returning null when a `pyproject.toml` exists at the project root. <br><br>- Notebook cell index resolution has been fixed to prevent mismatches between code cells and markdown cells, eliminating panics and incorrect byte offset calculations in Jupyter notebooks. <br><br>- Cross-module "find references" (external references) is now enabled by default, returning references across the entire project rather than just the current file. <br><br>- A new quick fix turns the existing "Did you mean `Foo.BAR`?" diagnostic note for missing enum members into a code action that replaces the offending string literal with the proper enum member access. <br><br>- A new `# pyrefly: ignore` quick fix inserts a suppression comment for the diagnostic at the cursor, automatically merging into an existing pyrefly-ignore directive on the same line or on a comment-only line above when present. <br><br>- Numeric parameter defaults now preserve their source spelling (e.g. `0o777`, `0xFF`, `0b101`) in hover and signature display rather than being normalized to decimal. <br><br>- Code actions documentation has been added to the IDE Supported Features page, covering quick fixes and `source.fixAll.pyrefly` configuration. |
| **Onboarding & VS Code Extension** | - A redesigned unconfigured-project experience: when no `pyrefly.toml` is found, Pyrefly auto-detects nearby `mypy.ini`, `pyrightconfig.json`, or `[tool.mypy]`/`[tool.pyright]` sections in `pyproject.toml` and synthesizes an in-memory configuration migrated from those settings (using the `legacy` or `default` preset respectively). With no detectable configuration, the new `basic` preset is used. <br><br>- A new `python.pyrefly.typeCheckingMode` workspace setting (auto / off / basic / legacy / default / strict, default `auto`) lets users pick a preset for files not covered by an explicit Pyrefly configuration, directly from the VS Code settings UI. The legacy `python.pyrefly.displayTypeErrors` setting is now deprecated, with values transparently mapped to the new model. <br><br>- A new `python.pyrefly.disableTypeErrors` workspace setting provides a clean per-workspace kill switch for diagnostics, independent of the type-checking mode. <br><br>- The VS Code status bar has been redesigned: it now shows the active preset (e.g. "Pyrefly (Legacy)", "Pyrefly (Basic)") and the tooltip explains why that preset was chosen and links to the relevant docs. <br><br>- After a `pyrefly check` on an unconfigured project, the CLI now prints a short upsell to **stderr** explaining what configuration was synthesized and pointing at `pyrefly init`. The message is routed to stderr so machine-readable stdout formats (e.g. `--output-format json`) remain untouched. |
| **Configuration** | - Configuration presets (`off`, `basic`, `legacy`, `default`, `strict`) are now available via the `preset` option, providing named collections of error severities and behavior settings as a base configuration that user settings can override. <br><br>- The `legacy` preset is now used by `pyrefly init` for mypy migration, disabling checks mypy doesn't have and setting looser inference defaults. <br><br>- The `implicit-any` error code has been split into sub-kinds (`implicit-any-attribute`, `implicit-any-empty-container`, `implicit-any-parameter`, `implicit-any-type-argument`) with `implicit-any` as the parent, allowing finer-grained control over where implicit `Any` is flagged. <br><br>- The `unbound-name` error is now disabled in the `legacy` preset to match mypy's default behavior, which does not flag possibly-undefined variables. |
| **Error Reporting** | - A new `incompatible-overload-residual` error kind has been introduced for cases where all branches of an overloaded callable are pruned during higher-order function analysis, making it easier to configure these errors independently. <br><br>- Error messages for all-pruned overload residuals now describe the incompatibility in terms of "solved type variables" rather than "solved type constraints" for better clarity. <br><br>- The `pyrefly suppress` command now correctly handles removal of unused ignores via the `--remove-unused` flag, which was previously broken. |
| **Factory Boy Support** | - Pyrefly now infers the correct model return types for `create()`, `build()`, `create_batch()`, and `build_batch()` methods on `DjangoModelFactory` subclasses by reading the inner `Meta.model` attribute. <br><br>- False-positive `bad-override` errors on the inner `Meta` class in factory-boy factories are now suppressed, matching how we handle Django and Marshmallow. |
| **Reporting** | - The `pyrefly report` JSON output now includes a `path` field on each `ModuleReport`, for compatibility with typestats and similar tooling. |
| **Performance** | - Deeply-nested dict literals no longer cause exponential memory growth during type inference. A depth-25 dict literal that previously consumed ~7.7 GB now uses ~239 MB by computing the union of field types on demand instead of storing it redundantly. <br><br>- Callable residual finalization has been optimized to avoid redundant type cloning and traversals, reducing memory churn in attribute-heavy code. <br><br>- Eliminated some bugs that caused Pyrefly to unnecessarily analyze dependencies, improving latency and memory use, especially in the IDE. |

---

## 🐛 bug fixes

We closed **15** bug issues this release 👏

- #3057: Fixed an issue where string concatenation with the `+` operator was incorrectly flagging `str` as not assignable to `LiteralString` attributes. Pyrefly now preserves `LiteralString` style when adding two explicit string literals and uses implicit style otherwise.
- #105: Fixed premature type pinning in function calls where arguments were incorrectly narrowed before all constraints were solved. For example, `foo(x, y)` with `x: int | None` and `y: int | None` no longer incorrectly narrows `x` to `None` when passed to a generic `foo[T](a: T, b: T)`.
- #3198: Fixed `pyrefly suppress --remove-unused` which was not actually removing unused error suppressions. The command now correctly processes the `--remove-unused` flag.
- #3024: The language server now advertises `source.fixAll.pyrefly` in addition to `source.fixAll`, allowing users to selectively enable or disable Pyrefly's fix-all actions in editors that support LSP code action kinds.
- #2819: Fixed incorrect variance errors when using generic classes like Pydantic's `RootModel` in lax mode. Missing type arguments now degrade to `Any` instead of raising errors, matching our handling of other incomplete types.
- #3000: Fixed "find references" failures in Cursor and other editors caused by relative imports in site-packages not resolving correctly when a `pyproject.toml` existed at the project root.
- #2563: Fixed go-to-definition on relative imports in virtual environment site-packages, which was returning null because the project root's import path was matching before the more specific site-package prefix.
- #3193: Fixed an error where `list["A|B"]` was incorrectly rejected as `not-a-type`. Type argument subscripts are now bound as type expressions even in value context, allowing forward-ref strings to be parsed.
- #3286: Fixed exponential memory blowup when type-checking deeply-nested dict literals, which could cause VSCode to be killed by the OS. Memory usage for a depth-25 dict dropped from ~7.7 GB to ~239 MB.
- #3261: Fixed a false positive `bad-class-definition` when a dataclass field was assigned inside a `@classmethod` or `__init_subclass__`. Pyrefly was incorrectly extracting these as dataclass fields, even though Python's `dataclasses.dataclass` ignores them at runtime.
- #2914: `assert` statements now flag a non-callable `__bool__` on the test expression, closing a gap that previously only caught the issue inside `if`, `while`, and ternary expressions.
- #2867: Fixed `urlunparse` being inferred as returning `Literal[b'']` instead of `str`. The fix reworks `as_superclass` so tuple-like `NamedTuple` subclasses are upcast through their erased tuple element types, which stops `ParseResult` from spuriously matching `Iterable[None]` and selecting the bytes overload.
- #3266: Added a quick fix for the existing "Did you mean `Foo.BAR`?" diagnostic note for missing enum members, turning the suggestion into a code action that rewrites the surrounding string literal.
- #3230: Numeric parameter defaults now preserve their original spelling (e.g. `0o777`) in hover and signature display rather than being normalized to a decimal value.
- #3302: Added a `path` field to the `pyrefly report` JSON `ModuleReport`, restoring compatibility with typestats.

Thank-you to all our contributors who found these bugs and reported them! Did you know this is one of the most helpful contributions you can make to an open-source project? If you find any bugs in Pyrefly we want to know about them! Please open a bug report issue [here](https://github.com/facebook/pyrefly/issues)

---


## 📦 Upgrade

```bash
pip install --upgrade pyrefly==0.64.0
```

### How to safely upgrade your codebase

Upgrading the version of Pyrefly you're using or a third-party library you depend on can reveal new type errors in your code. Fixing them all at once is often unrealistic. We've written scripts to help you temporarily silence them. After upgrading, follow these steps:

1\. `pyrefly check --suppress-errors`
2\. run your code formatter of choice
3\. `pyrefly check --remove-unused-ignores`
4\. Repeat until you achieve a clean formatting run and a clean type check.

This will add  `# pyrefly: ignore` comments to your code, enabling you to silence errors and return to fix them later. This can make the process of upgrading a large codebase much more manageable.

Read more about error suppressions in the \[Pyefly documentation\](https://pyrefly.org/en/docs/error-suppressions/)

---

## 🖊️ Contributors this release
@stroxler, @rchen152, @migeed-z, @grievejia, @samwgoldman, @kinto0, @asukaminato0721, @jvansch1, @NathanTempest, @avikchaudhuri, @connernilsen, Carlos Fernandez, David Tolnay, @jorenham, @rexledesma, @arthaud, @pawlowskialex, @QuantumManiac, @knQzx, @QEDady

---

Please note: These release notes summarize major updates and features. For brevity, not all individual commits are listed. Highlights from patch release changes that were shipped after the previous minor release are incorporated here as well.
