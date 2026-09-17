*Release date: September 18, 2026*

> **About dev releases**
> Dev releases (versions like `X.Y.Z-dev.N`) are non-stable snapshots cut periodically from trunk. They give early adopters a chance to try in-progress features and surface issues before the next stable release, but they don't carry the same stability or compatibility guarantees as a stable release — don't pin production projects to a dev version.

Pyrefly v1.4.0-dev.1 bundles **98 commits** from **13 contributors**.

---

## ✨ New & Improved

### Type Checking

- A new opt-in `uninitialized-instance-variable` check reports instance attributes that may be read before assignment, including fields on dataclasses declared with `init=False`.
- Overload inference now handles generic call targets and callable fallbacks more precisely, including `functools.partial`, independently pruned arguments, and writable callback variance.
- Property getters, setters, and deleters with extra required parameters now report invalid signatures when the decorator is applied.
- Unaliased directory imports now bind the imported name correctly, and source files nested beyond the configured analysis depth are rejected cleanly.
- Django `annotate()` calls now add their declared attributes to the resulting queryset type.

### Language Server

- Hover information now prefers interface docstrings when available and falls back consistently to implementation docstrings across imports.
- Parameter-name inlay hints now align correctly for bound methods whose receiver is named `__self`.

---

## 🐛 Bug fixes

We closed **3** bug issues this release 👏

- **#4918:** Property decorators now reject getters, setters, and deleters with extra required parameters instead of silently accepting invalid signatures.
- **#4936:** Fixed shifted parameter-name inlay hints for bound methods whose receiver is named `__self`, such as methods on `loguru.Logger`.
- **#1610:** SQLAlchemy dataclass fields now reject mutable `dict`, `list`, and `set` literals passed to `mapped_column(default=...)` and recommend `default_factory` instead.

Thank-you to all our contributors who found these bugs and reported them! Did you know this is one of the most helpful contributions you can make to an open-source project? If you find any bugs in Pyrefly we want to know about them! Please open a bug report issue [here](https://github.com/facebook/pyrefly/issues).

---

## 📦 Upgrade

```bash
pip install --upgrade pyrefly==1.4.0-dev.1
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

@rchen152, David Tolnay, @dependabot, @kylei, @dedsec-terminal, Oleh Prypin, Gábor Varga, @jakevdp, @jasonnathanieln, @stroxler, @ndmitchell, @connernilsen, @asukaminato0721

---

## 🔬 Tensor Shape Support

> **JAX stub support is still under development and is not yet available on PyPI.** The JAX improvements below describe ongoing work in the repository rather than an installable stub package in this release.
>
> The current `einops` stubs are limited to Torch tensors. We are still exploring how to support multiple array packages.

- JAX NumPy, linear algebra, and FFT APIs now accept `ArrayLike` inputs more broadly, with improved shape inference for `linspace`, `logspace`, and `geomspace`.
- JAX, NumPy, and PyTorch constructors now preserve shapes derived from nested list literals, including zero-sized dimensions.
- Torch shape stubs now expose public submodules as attributes and support the single-argument and keyword forms of `torch.where` and bare `@torch.no_grad`.
- Shape-aware einops stubs now cover the remaining public APIs for Torch tensors.
- `pyrefly-shape-extensions` now supports Python 3.10 and provides runtime markers for regular nested lists.

---

*Please note: These release notes summarize major updates and features. For brevity, not all individual commits are listed.*
