# Pyrefly v0.58.0
**Status : BETA**
*Release date: March 24, 2026*

Pyrefly v0.58.0 bundles **190 commits** from **24 contributors**.

---

## ✨ New & Improved

| Area | What's new |
|------|------------|
| **Type Checking** | - Binary operations like `<` and `+` now work correctly on identical constrained TypeVars (e.g., `T: int \| float`), eliminating false positives when comparing or adding values of the same type variable. <br><br>- Type variables now accumulate lower bounds instead of pinning immediately to the first encountered type, fixing issues with generic function calls and improving type inference accuracy. <br><br>- Overload resolution has been improved to eliminate overloads with incompatible parameter counts and to apply return type hints more intelligently, giving you more precise return types and better error messages. <br><br>- A bug in Pyrefly's variance inference algorithm that caused it to hang indefinitely on self-referential generic classes has been fixed. |
| **Language Server** | - Go-to-definition on constructor calls jumps directly to `__init__` or `__new__` methods instead of the class definition, and callable instances navigate to `__call__`. This makes it much easier to find the actual implementation you're looking for. <br><br>- Hover tooltips display float default parameter values instead of `...`, making function signatures more informative. <br><br>- The language server handles generic metaclasses with unbound type variables gracefully, preventing confusing cascading errors when attribute lookups resolve to bare type variables. <br><br>- Memory usage in watch mode has been significantly reduced by garbage-collecting stale loader entries when configurations are reloaded. <br><br>- Workspace diagnostic mode no longer shows false "memory path not found" errors after closing files, and warnings are correctly filtered out for non-open files. |
| **Exhaustiveness Checking** | - Exhaustiveness checking has been simplified and made more powerful. You can now use `isinstance` checks and multi-subject narrowing for exhaustiveness, and nested if/elif chains are recognized as exhaustive when all branches are covered. <br><br>- Pattern matching with faceted subjects (e.g., `match obj.attr`) correctly narrows types in negative cases, fixing false positives where fallback branches were incorrectly typed as `Never`. |
| **Error Reporting** | - A new `--min-severity` flag (and corresponding config option) lets you control the minimum severity level for displayed errors. By default, only errors are shown; warnings and info-level diagnostics are hidden unless you explicitly lower the threshold. <br><br>- `reveal_type` output is treated as a directive rather than a normal diagnostic, so it won't be affected by baseline exclusion, suppression commands, or severity thresholds. <br><br>- The `--remove-unused` flag on `pyrefly suppress` now also removes unused `# pyre-fixme` comments when Pyre is enabled in your configuration. |
| **Performance** | - Recursive return type inference limits both depth and inner union width, preventing exponential memory usage and dramatically improving performance on complex mutual recursion (e.g., the `weasyprint` library now type-checks in seconds instead of minutes). <br><br>- Several performance optimizations have been made to error collection, path lookups, and calculation caching, resulting in faster type checking across the board. |

---

## 🐛 bug fixes

We closed **20** bug issues this release 👏

- #2452: Fixed a memory leak where the language server's memory usage climbed by ~3GB on every `pyproject.toml` save. Stale loader entries are now garbage-collected at commit time, keeping memory usage stable across configuration reloads.
- #2826: Fixed an issue where pattern matching on an attribute (e.g., `match c.items: case [_]:`) incorrectly narrowed the base variable to `Never`. Sequence pattern narrowing operations now correctly propagate facet subject information.
- #2520: Fixed false positive "variable may be uninitialized" errors in nested if/elif branches. Exhaustive if/elif chains are now recognized as terminating for uninitialized local checks, just like match statements.
- #1518: Fixed an issue where exhaustive if/elif chains over enum variants still reported "unbound-name" errors. The exhaustiveness key is now passed through non-exhaustive fork merging so the synthetic fallthrough path is recognized as impossible.
- #1286: Fixed exhaustiveness checking in match statements to work with isinstance-based narrowing and mixed is/isinstance patterns. Narrowing to `Never` is now always sound for flow analysis regardless of the type being narrowed.
- #2833: Fixed an issue where `dict.get(k)` with an incorrect argument type returned `Unknown` instead of the correct return type. Overloads are now eliminated by arity before applying return type hints, giving you precise return types even when calls have errors.
- #823: Fixed ParamSpec forwarding between generic helpers. When one generic helper forwards `*args: P.args, **kwargs: P.kwargs` to another, the solver now correctly validates the forwarding pattern instead of producing bogus "Expected `P` to be a ParamSpec value" errors.
- #2309: Fixed false positive `bad-override` errors when subclassing `str` and adding optional parameters to overridden methods. Inapplicable parent overloads (e.g., `LiteralString` overloads when the subclass is not `LiteralString`) are now filtered out during override checking.
- #1083: Fixed false positive "No matching overload" errors when accessing descriptors with overloaded `__get__` through `type[T]` where `T` is bounded. The descriptor base now preserves the `ClassBase` wrapper to correctly produce `type[T]` instead of `type[A]`.
- #2007: Fixed false positive unused variable warnings when reassigning parameters in loops or when assigning to global/nonlocal variables. The unused variable analysis now correctly tracks these assignment patterns.
- And more! #2419, #1252, #2043, #2218, #650, #2164, #1635, #2168, #1005, #2655

Thank-you to all our contributors who found these bugs and reported them! Did you know this is one of the most helpful contributions you can make to an open-source project? If you find any bugs in Pyrefly we want to know about them! Please open a bug report issue [here](https://github.com/facebook/pyrefly/issues)

---


## 📦 Upgrade

```bash
pip install --upgrade pyrefly==0.58.0
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
@stroxler, @rchen152, @yangdanny97, @grievejia, @kinto0, @migeed-z, @jvansch1, @ndmitchell, @samwgoldman, @maggiemoss, @arthaud, @connernilsen, @fangyi-zhou, @avikchaudhuri, @asukaminato0721, @jackulau, @PhilHem, @Rayahhhmed, @oyarsa, generatedunixname2066905484085733, David Tolnay, Brian Rosenfeld, @rubmary, @rchiodo

---

Please note: These release notes summarize major updates and features. For brevity, not all individual commits are listed. Highlights from patch release changes that were shipped after the previous minor release are incorporated here as well.
