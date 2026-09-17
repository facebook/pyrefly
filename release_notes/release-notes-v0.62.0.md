# Pyrefly v0.62.0
**Status : BETA**
*Release date: April 20, 2026*

Pyrefly v0.62.0 bundles **87 commits** from **23 contributors**.

---

## ✨ New & Improved

| Area | What's new |
|------|------------|
| **Type Checking** | - `TypeVarTuple` inference has been changed to be consistent with `TypeVar`, per a recent change to the typing spec. <br><br>- Errors logged during speculative union checks and overload calls are now reverted, eliminating a source of confusing false positives. <br><br>- Union-typed decorators that return fully unknown types (either `Unknown` or callables with all-unknown signatures) preserve the original function signature instead of replacing it with `Unknown`, reducing false positives by ~23% on TensorFlow. |
| **Language Server** | - Semantic tokens and completions work for `inmemory://` documents on Windows. <br><br>- LSP server crashes from out-of-range line numbers in client requests are prevented by clamping positions to the buffer's valid range. |
| **Error Reporting** | - Error kinds can now have sub-kinds that can be disabled using their shared prefix. <br><br>- Invariance checks for mutable attributes (corresponding to mypy's `mutable-override` opt-in behavior) have been moved to a new `bad-override-mutable-attribute` error code that is a sub-kind of `bad-override`. <br><br>- The `bad-param-name-override` error has been renamed to `bad-override-param-name` and made a sub-kind of `bad-override`. <br><br>- Sub-configs that define `[errors]` inherit the root config's error severity overrides for any codes they don't explicitly set. |
| **Configuration** | - When migrating from mypy via `pyrefly init`, `bad-override-mutable-attribute` is disabled by default to match mypy's behavior. <br><br>- Project excludes (e.g., `project-excludes = ["**/*.ipynb"]`) no longer block discovery of `.py` files when the default `project-includes` contains both `**/*.py*` and `**/*.ipynb`. |

---

## 🐛 bug fixes

We closed 12 bug issues this release 👏

- #3118: Fixed incorrect stub package recommendations for typeshed third-party libraries. Pyrefly now suggests the correct package name (e.g., `types-python-dateutil` for the `dateutil` module, not `types-dateutil`) by extracting the module→package mapping from the bundled typeshed archive, preventing potential typosquatting.
- #3081: Fixed NewType wrappers with NoneType bases being incorrectly rejected or treated inconsistently. `NewType("NewNoneType", NoneType)` is now accepted as a valid nominal type declaration, and plain `None` is correctly rejected where `NewNoneType` is required.
- #3052: Fixed false positive `unexpected-keyword` errors for named parameters before `*args: P.args`. Functions like `call_with_retry(f, max_attempts=10, *args: P.args, **kwargs: P.kwargs)` now correctly allow `max_attempts` to be passed as a keyword argument, matching mypy and pyright behavior.
- #3110: Fixed LSP server crashes when the client sends a position with a line number beyond the end of the buffer (e.g., after a `DidChangeTextDocument` race where the file was truncated). Out-of-range positions now map to EOF instead of panicking.
- #2912: Fixed false positive `bad-argument-type` for `list(null_values.items())` when the return type hint is a union like `Sequence[str] | list[tuple[str, str]]`. Pyrefly now tries constructing the class with each union member independently and unions the results, ensuring the inferred type is assignable to the hint.
- #2644: Fixed false positive `bad-argument-type` when calling a method with `AnyStr`. Placeholder variables used during overload resolution are now saved and restored around overload calls, preventing `AnyStr` from being incorrectly specialized to `str` and polluting subsequent checks.
- #2872: Fixed false positive `invalid-type-var` for generic functions captured as closure default arguments. The `Visit` implementation for `DefaultValue` now calls `visit` instead of `recurse`, ensuring type-level visitors see the `Type` node stored in the default value.
- #3159: Fixed incorrect type inference for `.value` on enum members with non-data-type mixins. Mixins that don't define `__new__` (e.g., `class Meta: pass`) are no longer treated as data type mixins, so `Foo.bar.value` correctly returns `Literal[1]` instead of `Meta`.
- #3161: Fixed false positive `bad-argument-type` for overloaded functions with vararg unpacking (e.g., `*args: *tuple[int, str]`). Type check errors for unpacked varargs are now sent to `call_errors` instead of `arg_errors`, so they don't cause the overload to be incorrectly rejected.
- #3047: Fixed false positive `bad-specialization` when matching a type variable against a union like `N | Iterable[N]`. Pyrefly now uses snapshot-based rollback when trying each union member, ensuring specialization errors from one branch don't leak into the final result if another branch succeeds without errors.
- And more! #3122, #3080, #3074

Thank-you to all our contributors who found these bugs and reported them! Did you know this is one of the most helpful contributions you can make to an open-source project? If you find any bugs in Pyrefly we want to know about them! Please open a bug report issue [here](https://github.com/facebook/pyrefly/issues)

---


## 📦 Upgrade

```bash
pip install --upgrade pyrefly==0.62.0
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
@rchen152, @stroxler, @migeed-z, @jorenham, @samwgoldman, @connernilsen, @NathanTempest, @kinto0, @jvansch1, @fangyi-zhou, @tejasreddyvepala, David Tolnay, generatedunixname2066905484085733, @mstykow, @AHA705, @Arths17, @grievejia, Claudionor Santos, @austin3dickey, @robertoaloi, @salvatorebenedetto, @iamPulakesh, generatedunixname89002005307016

---

Please note: These release notes summarize major updates and features. For brevity, not all individual commits are listed. Highlights from patch release changes that were shipped after the previous minor release are incorporated here as well.
