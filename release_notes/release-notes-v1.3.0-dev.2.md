*Release date: August 17, 2026*

> **About dev releases**
> Dev releases (versions like `X.Y.Z-dev.N`) are non-stable snapshots cut periodically from trunk. They give early adopters a chance to try in-progress features and surface issues before the next stable release, but they don't carry the same stability or compatibility guarantees as a stable release — don't pin production projects to a dev version.

Pyrefly v1.3.0-dev.2 bundles **126 commits** from **29 contributors**.

---

## ✨ New & Improved

### Type Checking

- Protocol methods named `__call__` are now checked for override consistency like any other method, catching incompatible signatures that would fail at runtime.
- Scoped type aliases can now match `TypeForm`, fixing cases where valid type expressions were incorrectly rejected.
- `no-any-return` no longer fires on functions declared to return `object`, since all runtime values satisfy `object`.
- Non-callable `__init_subclass__` attributes are now detected, catching runtime errors when a class inherits from a parent with a non-callable `__init_subclass__`.
- Type parameter defaults now correctly fall back to the appropriate type (`object` for TypeVars, `tuple[()]` for TypeVarTuples, `...` for ParamSpecs) rather than always using `Any`.
- Unmatched type parameters at the end of a type parameter list now take into account previously added defaults when computing their own default values.

### Language Server

- Diagnostics for untitled files now appear as squiggles in the editor, not just in the Problems panel.
- The language server now restarts atomically when configuration changes, preventing race conditions where file watcher updates could arrive while the client was down.
- Relative imports are now correctly updated when renaming files within a package, and imports are converted to absolute paths when a file is moved outside its package.

### CLI & Configuration

- The Bazel integration is now documented, with setup instructions for the Bzlmod toolchain and aspect workflow.
- Mypy and Pyright configuration migration now preserves more settings, including strict mode, platform, untyped-body behavior, and follow_imports, with warnings when error codes cannot be mapped.
- Configuration documentation has been updated to reflect the `preset=auto` default behavior.
- Multiple `--output` destinations with different formats can now be specified on the command line, enabling simultaneous output to multiple formats.

### Baseline Files

- Baseline files now omit line numbers and store only concise descriptions, reducing churn when unrelated code shifts and shrinking baseline file size.
- New `--error-unused` flag exits non-zero when the baseline contains unused entries, ensuring CI fails until the baseline is refreshed.
- New `--remove-unused` flag rewrites the baseline with only entries that still match, allowing the set of suppressed errors to shrink without recording new ones.
- Baselined errors can now be shown in output at a reduced severity using the `baseline-error-level` configuration option, with provenance markers indicating which results matched the baseline.

---

## 🐛 Bug fixes

We closed **16** bug issues this release 👏

- **#4411:** Annotation-only class attributes in stub files are now resolvable within the same class body, fixing cases where `x: int` followed by `y = x` incorrectly reported that `x` could not be found.
- **#4342:** `datetime.datetime` is now correctly recognized as a subtype of `datetime.date`, fixing false positives when dates and datetimes were used interchangeably.
- **#4482:** Functions with `*args: *tuple[*Ts, Suffix]` annotations now correctly track the variadic tuple shape during call matching, accepting arguments that match the prefix, middle, and suffix elements.
- **#4471:** Descriptor semantics are no longer incorrectly applied to annotated instance attributes that are initialized in `__init__`, fixing false errors when an annotation's type implements `__get__` but no class-level descriptor exists.
- **#4493:** Scoped type aliases that resolve to `Literal` types can now match `TypeForm`, fixing cases where valid type expressions were rejected.
- **#4055:** Relative imports are now correctly adjusted when renaming files within a package, and converted to absolute imports when a file is moved outside its package.
- **#4497:** Diagnostics for untitled files are now published on the correct URI, so squiggles appear in the editor rather than only in the Problems panel.
- **#4424:** Explicit re-exports in `__all__` are now preserved even when a later dynamic append or extend cannot be resolved, fixing false `implicit-reexport` errors for names like `torch.Tensor`.
- **#4541:** Field specifiers for descriptor-typed fields now correctly preserve `init=False` and required/optional status, fixing false errors when dataclass-transform field specifiers return descriptors.
- **#4378:** Variadic positional parameter hints now preserve the `*` marker, so `args=` becomes `*args=` in signature help.
- And more! #3987, #4401, #3706, #3447, #4521, #3183

Thank-you to all our contributors who found these bugs and reported them! Did you know this is one of the most helpful contributions you can make to an open-source project? If you find any bugs in Pyrefly we want to know about them! Please open a bug report issue [here](https://github.com/facebook/pyrefly/issues).

---

## 📦 Upgrade

```bash
pip install --upgrade pyrefly==1.3.0-dev.2
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

@rchen152, generatedunixname2066905484085733, @yangdanny97, David Tolnay, generatedunixname949130641157030, @javabster, @shobhitmehro, @asukaminato0721, @samwgoldman, @kinto0, @alexander-beedie, @grievejia, @connernilsen, @tobyh-canva, @IBlackVoid, @nitishagar, @WilliamK112, @lyydsheep, @NathanTempest, @ytausch, @patrickswedish, @xaskii, generatedunixname1431085361989520, @a7or, @markselby9, @heejaechang, Willem Kokke, @dillydill123, @kakolla

---

*Please note: These release notes summarize major updates and features. For brevity, not all individual commits are listed.*
