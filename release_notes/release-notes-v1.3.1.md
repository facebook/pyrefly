*Release date: September 14, 2026*

Pyrefly v1.3.1 is a patch release with a single bug fix.

---

## 🐛 Bug fixes

- **#4909:** Fixed a critical issue where Pyrefly would immediately crash on startup with a thread panic when empty or relative base paths were encountered during configuration. The server now safely resolves these paths against the working directory, preventing the crash and allowing normal operation.

Thank-you to all our contributors who found these bugs and reported them! Did you know this is one of the most helpful contributions you can make to an open-source project? If you find any bugs in Pyrefly we want to know about them! Please open a bug report issue [here](https://github.com/facebook/pyrefly/issues).

---

## 📦 Upgrade

```bash
pip install --upgrade pyrefly==1.3.1
```

### How to safely upgrade your codebase

Upgrading the version of Pyrefly you're using or a third-party library you depend on can reveal new type errors in your code. Fixing them all at once is often unrealistic. We've written scripts to help you temporarily silence them. After upgrading, follow these steps:

1. `pyrefly check --suppress-errors`
2. Run your code formatter of choice
3. `pyrefly check --remove-unused-ignores`
4. Repeat until you achieve a clean formatting run and a clean type check.

This will add `# pyrefly: ignore` comments to your code, enabling you to silence errors and return to fix them later. This can make the process of upgrading a large codebase much more manageable.

Read more about error suppressions in the [Pyrefly documentation](https://pyrefly.org/en/docs/error-suppressions/).
