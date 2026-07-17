# Pyrefly v0.61.0
**Status : BETA**
*Release date: April 13, 2026*

Pyrefly v0.61.0 bundles **85 commits** from **21 contributors**.

---

## ✨ New & Improved

| Area | What's new |
|------|------------|
| **Type Checking** | - Division, floor division, and modulo operations with a literal zero divisor (e.g., `x / 0`, `y // 0`, `z % 0`) are flagged as errors, catching runtime `ZeroDivisionError` before execution. <br><br>- Multiple inheritance with conflicting `__slots__` definitions is detected and reported as an error, matching CPython's runtime behavior and preventing layout conflicts. <br><br>- Protocol members assigned a value without an explicit type annotation (e.g., `x = None` in a `Protocol` class body) are flagged as errors, ensuring protocol members have declared types as required by the typing specification. |
| **Language Server** | - Variables used exclusively within f-string format specifiers (e.g., `f"{key:<{max_len}}"`) are correctly recognized as used, eliminating false positive unused-variable warnings. <br><br>- The VS Code extension explicitly declares workspace trust capabilities, requiring trusted workspaces to run and allowing machine-overridable scope for `lspPath` and `lspArguments` settings for improved security. |
| **Coverage Reporting** | - The `pyrefly report` command now excludes some dunder methods and typing-only constructs from coverage metrics. <br><br>- Per-module JSON output includes entity counts (n_functions, n_methods, n_function_params, n_method_params, n_classes, n_attrs, n_properties, n_type_ignores) for downstream consumers. <br><br>- A new `--module <name>` CLI flag allows overriding the module name in JSON output, supporting callers that need canonical package names instead of filesystem-derived names. |
| **Pydantic** | - Pydantic lax conversion special-cases regex patterns, fixing false positives when passing compiled patterns to Pydantic models. |
| **Performance** | - Fixed a bug in overload evaluation that caused exponential memory consumption and indefinite hangs on code with many overloaded calls. |

---

## 🐛 bug fixes

We closed **9** bug issues this release 👏

- #3031: Fixed a crash in mypy_primer caused by a variable leak in `LitEnum` — types are now deep-forced before storage to prevent leaking vars into the solver.
- #2915: Division, floor division, and modulo by literal `0` are now flagged as errors, catching `ZeroDivisionError` at static analysis time instead of runtime.
- #3009: Fixed false positive unused-variable warnings for variables used exclusively within f-string format specifiers (e.g., `f"{key:<{max_len}}"`). The AST visitor now correctly descends into `format_spec` nodes.
- #2799: Fixed false positive `[missing-attribute]` errors for `dict.setdefault(key, []).append(val)` on unannotated dicts. Overload resolution now creates fresh partial variables for each overload, preventing incorrect pinning.
- #2991: Fixed Pydantic lax-mode rewriting `re.Pattern[str]` to `Pattern[LaxStr]` and rejecting `re.Pattern[str]`. Regex patterns now expand to `re.Pattern[T] | T` instead of recursively widening the inner type.
- #2916: Fixed runtime `TypeError` from multiple inheritance with conflicting `__slots__` (same slot names). Pyrefly now detects and reports this layout conflict during class metadata computation.
- #2917: Fixed runtime `TypeError` from multiple inheritance with conflicting `__slots__` (different slot names). Pyrefly now detects non-empty `__slots__` in multiple bases and reports the conflict.
- #3064: Fixed false positive when using `issubclass()` after `isinstance()` narrowing with custom metaclasses (e.g., Django's `ModelBase`). Metaclass instances are now correctly accepted as valid class objects.
- #3030: Fixed false positive `LiteralString` type error in `map(str.strip, ...)`. Overloads with narrower `self`-type annotations are now filtered out during unbound method resolution.

Thank-you to all our contributors who found these bugs and reported them! Did you know this is one of the most helpful contributions you can make to an open-source project? If you find any bugs in Pyrefly we want to know about them! Please open a bug report issue [here](https://github.com/facebook/pyrefly/issues)

---


## 📦 Upgrade

```bash
pip install --upgrade pyrefly==0.61.0
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
@rchen152, @migeed-z, @javabster, generatedunixname2066905484085733, @rubmary, @kinto0, @lolpack, @asukaminato0721, @knQzx, David Tolnay, @yangdanny97, @Arths17, @tejasreddyvepala, @salvatorebenedetto, @rchiodo, @stroxler, @samwgoldman, @arthaud, @fangyi-zhou, @NathanTempest, generatedunixname89002005307016

---

Please note: These release notes summarize major updates and features. For brevity, not all individual commits are listed. Highlights from patch release changes that were shipped after the previous minor release are incorporated here as well.
