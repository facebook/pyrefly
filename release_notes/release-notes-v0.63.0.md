# Pyrefly v0.63.0
**Status : BETA**
*Release date: April 27, 2026*

Pyrefly v0.63.0 bundles **129 commits** from **26 contributors**.

---

## ✨ New & Improved

| Area | What's new |
|------|------------|
| **Type Checking** | - Enum member types are preserved even when the metaclass conflicts with `EnumMeta`, reducing noise in projects using custom metaclasses with enums. <br><br>- Constrained `TypeVar`s no longer get pinned to a specific constraint when matched against `Any`, preventing false positives. <br><br>- Self/cls annotations on all methods and classmethods are validated to ensure they reference the defining class or a superclass, catching more annotation errors. |
| **Language Server** | - The LSP now reports `unused-ignore` diagnostics when configured to do so, helping you clean up stale suppression comments. <br><br>- Completions for attribute override definitions are available in class bodies, surfacing base-class members filtered by fuzzy match. <br><br>- The LSP server no longer crashes on Jupyter notebook cell URIs (`vscode-notebook-cell:`), with full support for resolving notebook cell paths and position offsets. <br><br>- Workspace symbol search uses the correct location for re-exported symbols, preventing panics on multi-byte UTF-8 characters. <br><br>- Inlay hints are clickable for built-in types like `tuple`, `dict`, and `str`, enabling go-to-definition directly from hint overlays. |
| **Error Messages** | - A new `unnecessary-type-conversion` lint warns when `str()`, `int()`, or `float()` is called on an argument that is already of that exact type. |
| **Reporting & Coverage** | - Public symbol filtering is available via `pyrefly report --public-only`, using cross-module tracing to report only public symbols. |
| **Performance** | - TypedDict subset checks are now cached on the Solver, reducing CPU time by ~5.3x and wall time by ~6.7x on pydantic (from 9.5s to 1.4s). |
| **Configuration & Initialization** | - `pyrefly init` supports `--dry-run` for safe previews without writing files, and `--print-config` for machine-readable TOML output. |

---

## 🐛 bug fixes

We closed **9** bug issues this release 👏

- #3099: Fixed an issue where property setters and deleters inflated typable counts in `pyrefly report` by incorrectly counting their trivial `-> None` return types.
- #3098: Fixed an issue where overloads in `pyrefly report` were not deduplicated, causing parameters and callable signatures to be counted multiple times and inflate coverage metrics.
- #3067: Fixed an issue where the type display path was dropping the unpack marker (`*`) for direct `TypeVarTuple` arguments, causing `Shape` to render bare instead of `*Shape`.
- #3040: Fixed an issue where properties on metaclasses were not taking precedence over properties on the class during class-level attribute access, causing false `bad-assignment` and `bad-return` errors.
- #3150: Fixed an issue where type aliases were inflating type coverage in `pyrefly report` by being counted as typable entities.
- #3041: Fixed a panic during workspace/symbol requests on re-exported symbols with multi-byte UTF-8 characters, caused by using the canonical module's byte offset against the re-exporting file's buffer.
- #3109: Added a new `unnecessary-type-conversion` lint that warns when `str()`, `int()`, or `float()` is called on an argument that is already of that exact type, making the conversion redundant.
- #3187: Fixed a panic in `pyrefly report` when `@no_type_check` decorator was used, caused by a missing key lookup for skipped parameter annotations.
- #3090: Improved the unused-coroutine error message when an `await` expression already has `await` but produces a coroutine due to an incorrect return type annotation on the function definition.


Thank-you to all our contributors who found these bugs and reported them! Did you know this is one of the most helpful contributions you can make to an open-source project? If you find any bugs in Pyrefly we want to know about them! Please open a bug report issue [here](https://github.com/facebook/pyrefly/issues)

---


## 📦 Upgrade

```bash
pip install --upgrade pyrefly==0.63.0
```

### How to safely upgrade your codebase

Upgrading the version of Pyrefly you're using or a third-party library you depend on can reveal new type errors in your code. Fixing them all at once is often unrealistic. We've written scripts to help you temporarily silence them. After upgrading, follow these steps:

1\. `pyrefly check --suppress-errors`
2\. run your code formatter of choice
3\. `pyrefly check --remove-unused-ignores`
4\. Repeat until you achieve a clean formatting run and a clean type check.

This will add  `# pyrefly: ignore` comments to your code, enabling you to silence errors and return to fix them later. This can make the process of upgrading a large codebase much more manageable.

Read more about error suppressions in the [Pyrefly documentation](https://pyrefly.org/en/docs/error-suppressions/)

---

## 🖊️ Contributors this release
@rchen152, @migeed-z, @avikchaudhuri, @grievejia, @kinto0, @jorenham, @jvansch1, generatedunixname2066905484085733, @stroxler, @tejasreddyvepala, David Tolnay, @fangyi-zhou, @asukaminato0721, @lolpack, @NathanTempest, @connernilsen, @zbowling, @rubmary, @rexledesma, Anass Al-Wohoush, @javabster, @ABohra3, generatedunixname89002005232357, @tkaleas, @knQzx, generatedunixname89002005307016

---

Please note: These release notes summarize major updates and features. For brevity, not all individual commits are listed. Highlights from patch release changes that were shipped after the previous minor release are incorporated here as well.
