*Release date: October 1, 2026*

> **About dev releases**
> Dev releases (versions like `X.Y.Z-dev.N`) are non-stable snapshots cut periodically from trunk. They give early adopters a chance to try in-progress features and surface issues before the next stable release, but they don't carry the same stability or compatibility guarantees as a stable release — don't pin production projects to a dev version.

Pyrefly v1.4.0-dev.3 bundles **86 commits** from **19 contributors**.

This is a short snapshot — three days after v1.4.0-dev.2 — so it is a small release focused on unreachable-code reporting and language-server fixes.

---

## ✨ New & Improved

### Type Checking

- Unreachable-code reporting continues to expand: Pyrefly now reports `except` clauses that can never be entered, redundant exception classes inside a reachable `except` clause, and dead code after a call that never returns.

### Language Server

- Pyrefly migrated from its `lsp-types` fork to `gen-lsp-types` 0.11.0.
- Baselines now have the option to automatically remove fixed baselined errors on file save.

### Other

- Build-system-supplied configuration is better documented, and module finder options moved into a struct.
- Agent invocation metadata is now shared and used for error telemetry.

---

## 🐛 Bug fixes

- **#4825:** Fixed a false `bad-return` for a generator function annotated `Iterator[X] | T`.
- **#5073:** Fixed a series of unreachability bugs that made it into the last release.
- **#4668:** Extension-less Python scripts, such as a shebang script named `script` rather than `script.py`, got no diagnostics in the editor. Their errors were computed and then dropped before publishing, because the default `project-includes` globs cannot match a path with no extension. These files are now reported, and they respect `project-includes` by matching as though they ended in `.py`.
- An unsaved file is now checked as the language the editor reports it to be, rather than always being assumed to be Python.
- An assignment to a field inherited from a frozen Pydantic model is now checked. The assigned value was skipped, so a type mismatch went unreported and the attribute narrowed to `Never`.
- The right-hand side of an attribute assignment is now always inferred. It was sometimes skipped, which hid errors in the assigned expression.

Thank-you to all our contributors who found these bugs and reported them! Did you know this is one of the most helpful contributions you can make to an open-source project? If you find any bugs in Pyrefly we want to know about them! Please open a bug report issue [here](https://github.com/facebook/pyrefly/issues).

---

## 📦 Upgrade

```bash
pip install --upgrade pyrefly==1.4.0-dev.3
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

@stroxler, @connernilsen, @rchen152, @kinto0, @asukaminato0721, @samwgoldman, @grievejia, @shobhitmehro, @jakevdp, @nitishagar, @shahhamdihassan, @arthaud, @Trighap52, @maggiemoss, @dantrapp, @tonyyuyiding, @MarcoGorelli, @igorsugak, @NarxPal

---

## 🔬 Tensor Shape Support

> **JAX stub support is still under development and is not yet available on PyPI.** The JAX improvements below describe ongoing work in the repository rather than an installable stub package in this release.

- JAX `svd` shape typing is improved.
- `jnp.cross` no longer accepts 2D vectors.

---

*Please note: These release notes summarize major updates and features. For brevity, not all individual commits are listed.*
