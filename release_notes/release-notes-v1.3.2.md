*Release date: September 28, 2026*

Pyrefly v1.3.2 is a patch release that fixes regressions in how Pyrefly infers the type of a variable after it is assigned a value of type `Any`.

---

## 🐛 Bug fixes

- **#4846:** Fixed several regressions from v1.3.0 in type inference after a variable is reassigned to `Any`:
  - Assigning an implicit `Any` (a value with no known type, such as the result of an unannotated function or an unresolved import) to an annotated variable once again produces `Unknown`. It no longer turns into an explicit `Any`.
  - When a variable annotated as `Optional[T]` or `T | None` is narrowed to `None` and then reassigned inside that branch, Pyrefly no longer adds the annotation's `None` back after the branch. For example, `x: Any | None` followed by `if x is None: x = get_any()` now gives `Any` instead of `Any | None`.
  - When a variable's annotation is a union that contains `Any`, assigning `Any` to it now narrows the variable to `Any`. For example, `x: Any | None` followed by `x = x or get_any()` now gives `Any` instead of `Any | None`, so later attribute access no longer reports a false `missing-attribute` error on `None`.

Thank-you to all our contributors who found these bugs and reported them! Did you know this is one of the most helpful contributions you can make to an open-source project? If you find any bugs in Pyrefly we want to know about them! Please open a bug report issue [here](https://github.com/facebook/pyrefly/issues).

---

## 📦 Upgrade

```bash
pip install --upgrade pyrefly==1.3.2
```

### How to safely upgrade your codebase

Upgrading the version of Pyrefly you're using or a third-party library you depend on can reveal new type errors in your code. Fixing them all at once is often unrealistic. We've written scripts to help you temporarily silence them. After upgrading, follow these steps:

1. `pyrefly check --suppress-errors`
2. Run your code formatter of choice
3. `pyrefly check --remove-unused-ignores`
4. Repeat until you achieve a clean formatting run and a clean type check.

This will add `# pyrefly: ignore` comments to your code, enabling you to silence errors and return to fix them later. This can make the process of upgrading a large codebase much more manageable.

Read more about error suppressions in the [Pyrefly documentation](https://pyrefly.org/en/docs/error-suppressions/).
