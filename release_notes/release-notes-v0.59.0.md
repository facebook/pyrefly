# Pyrefly v0.59.0
**Status : Beta**
***Release date:** March 30, 2026*

Pyrefly v0.59.0 bundles **153 commits** from **20 contributors**.

---

## ✨ New & Improved

| Area | What’s new |
| :---- | :---- |
| **Type Checking** | - You can now use `while...else` statements with returns in the `else` clause without triggering a false positive `missing-explicit-return` error.  <br><br>- Pyrefly now correctly handles type inference for nested empty dictionaries when constructing TypedDict instances, avoiding `implicit-any` errors. <br><br>- Error messages now highlight related code with inline labels; for example, an unsupported * operation will show the types of both operands directly in the source snippet  |
| **Language Server** |  \- LSP hover information for classes now displays constructor signature and docstring. <br><br>- Support additional LSP functionality for notebooks, including find-references and rename. |
| **Performance** | - Faster typechecking in large pythonc codebases, up to 2x faster on recent benchmarks on real world projects <br><br>- Reduced CPU usage through smarter caching of module resolution results <br><br>- Improved performance of the LSP server by reducing redundant workspace diagnostic publishes. |

---

## 🐛 bug fixes

We closed 16 bug issues this release 👏

- \#2026: Fixed an issue where recursive bounded generics were incorrectly reported as `object`, ensuring accurate type checking.
- \#2812: Resolved a false positive `invalid-type-var` error when persisting the `get` method of a fully-annotated `dict`.
- \#2804: Fixed an `implicit-any` false positive that occurred with TypedDict items, improving code readability.
- \#2868: Pyrefly now correctly recognizes `while...else` statements with returns in the `else` clause as exhaustive.
- \#2814: Enhanced hover information for `datetime.datetime` imports to display constructor signatures and docstrings.
- \#2896: Fixed a `bad-argument-type` error that occurred when using double-underscore arguments.
- \#2893: Pyrefly now correctly handles dict Literal key types as subtypes of str key types.
- \#2865: Resolved an issue where tuple subclasses with overridden `__getitem__` were not recognized.
- \#2871: Fixed a false positive error when using `isinstance` with `type | X`.
- And more\! \#2444, \#1270, \#2900,  \#2862, \#2853

Thank-you to all our contributors who found these bugs and reported them\! Did you know this is one of the most helpful contributions you can make to an open-source project? If you find any bugs in Pyrefly we want to know about them\! Please open a bug report issue [here](https://github.com/facebook/pyrefly/issues)

---

## 📦 Upgrade

```shell
pip install --upgrade pyrefly==0.59.0
```

### How to safely upgrade your codebase

Upgrading the version of Pyrefly you're using or a third-party library you depend on can reveal new type errors in your code. Fixing them all at once is often unrealistic. We've written scripts to help you temporarily silence them. After upgrading, follow these steps:

1. `pyrefly check --suppress-errors`
2. run your code formatter of choice
3. `pyrefly check --remove-unused-ignores`
4. Repeat until you achieve a clean formatting run and a clean type check.

This will add  `# pyrefly: ignore` comments to your code, enabling you to silence errors and return to fix them later. This can make the process of upgrading a large codebase much more manageable.

Read more about error suppressions in the [Pyrefly documentation](https://pyrefly.org/en/docs/error-suppressions/)

---

## 🖊️ Contributors this release

@rchen152, @lolpack, @yangdanny97, @stroxler, @samwgoldman, @jvansch1, @kinto0, @connernilsen, @asukaminato0721, @migeed-z, @arthaud, @grievejia, @rubmary, @Adist319, David Tolnay, @yslim-030622, @tejasreddyvepala, @mvanhorn, @MountainGod2

---

Please note: These release notes summarize major updates and features. For brevity, not all individual commits are listed. Highlights from patch release changes that were shipped after the previous minor release are incorporated here as well.
