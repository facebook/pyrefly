*Release date: September 28, 2026*

> **About dev releases**
> Dev releases (versions like `X.Y.Z-dev.N`) are non-stable snapshots cut periodically from trunk. They give early adopters a chance to try in-progress features and surface issues before the next stable release, but they don't carry the same stability or compatibility guarantees as a stable release — don't pin production projects to a dev version.

Pyrefly v1.4.0-dev.2 bundles **493 commits** from **20 contributors**.

---

## ⚠️ Notable change: `X | None` annotations with an unknown `X`

Pyrefly now decides whether `|` builds a type union from the type-form context of the expression, instead of from the types of its operands. Before this change, an annotation such as `UnresolvedType | None` (for example, when `UnresolvedType` comes from a missing import) was evaluated as plain `Unknown`, and the `| None` was silently dropped. Pyrefly now evaluates it as `Unknown | None`.

This is more correct, but it can surface new errors, such as `missing-attribute` on `None`, in code that uses unresolved types in `Optional`-style annotations. Please try this dev release and [report](https://github.com/facebook/pyrefly/issues) any false positives. (#4846)

---

## ✨ New & Improved

### Type Checking

- Pyrefly now respects the `stdlib/VERSIONS` file of a custom typeshed, and it notices when that file changes.
- Build systems can now supply a default configuration.
- Unreachable-code reporting is more precise: Pyrefly reports dead code after a run of `with` statements, branches that are decided by the type of their test, and `except` clauses that can never match.
- Pydantic `Strict` types, such as `StrictInt`, now enable strict mode for the field.
- `TypeVarTuple` variance is now checked.
- `NewType` over an abstract class no longer produces a false-positive `bad-instantiation` error.

### Language Server

- The status indicator refreshes when a `pyrefly.toml` file is added or removed, and Pyrefly rewatches files when an explicitly configured config file changes.
- The TSP `initialize` response now includes `serverInfo`.
- In a `match` value pattern, attribute completion now ranks enum members that earlier `case` arms already cover lower.
- Baseline matching supports a new `column-ordered` mode, and the language server matches baselines in batches.

### Other

- Pyrefly now has an experimental programmatic API for Monty.
- When some warnings are hidden, the message now explains how to show them.
- Memory usage and commit latency in the incremental state are improved.

---

## 🐛 Bug fixes

- **#4846:** Fixed regressions from v1.3.0 in type inference after a variable is reassigned to `Any`. See also the notable change above.
- **#4858, #4618:** `Enum.value` no longer widens a literal value to its base type (for example, to `str`).
- **#4400:** Pydantic private attributes (`PrivateAttr`) are no longer treated as frozen.
- **#4995:** Django models now give `pk` the correct type when the primary key is a `ForeignKey(..., primary_key=True)`.
- **#4991:** `pyrefly suppress` now writes `# pyrefly: ignore[code]` without a space before the bracket, which matches the documentation.
- **#4986:** Hover now works on attribute assignments in `__init__`.
- **#4813:** Pyrefly no longer rejects a duck-typed `Mapping` in a `**` dict unpacking.
- **#4650:** `stubgen` no longer keeps the full analysis of the whole import closure in memory.
- **#4592:** A `__get__` overload with a self type of `Concatenate[ObjT, P1]` and a sibling `TypeVar` now matches correctly.

Thank-you to all our contributors who found these bugs and reported them! Did you know this is one of the most helpful contributions you can make to an open-source project? If you find any bugs in Pyrefly we want to know about them! Please open a bug report issue [here](https://github.com/facebook/pyrefly/issues).

---

## 📦 Upgrade

```bash
pip install --upgrade pyrefly==1.4.0-dev.2
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

@stroxler, @connernilsen, @rchen152, @kinto0, @asukaminato0721, @samwgoldman, @jakevdp, @yeetypete, @MarcoGorelli, @grievejia, Abby Mitchell, @Pager-dot, @arthaud, @digvijaysai29, David Tolnay, @yangdanny97, @Aniketsy, @QEDady, @alexander-beedie, Alan Kurusingal

---

## 🔬 Tensor Shape Support

> **JAX stub support is still under development and is not yet available on PyPI.** The JAX improvements below describe ongoing work in the repository rather than an installable stub package in this release.

- Torch shape stubs cover much more of the public API, including more pointwise, reduction, indexing, FFT, pooling, convolution, and attention operations, and they report unknown top-level Torch APIs.
- A new `IntVar[N]` runtime wrapper supports dimension arithmetic. Bare `*tuple[...]` and `*IntTuple` splats are now accepted in shapes.
- JAX and NumPy stubs have improved shape inference for `random`, `lax`, `pad`, `split`, `expand_dims`, and other operations, and the top-level `jax` namespace now has stubs.

---

*Please note: These release notes summarize major updates and features. For brevity, not all individual commits are listed.*
