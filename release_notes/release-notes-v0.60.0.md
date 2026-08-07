# Pyrefly v0.60.0
**Status : BETA**
*Release date: April 06, 2026*

Pyrefly v0.60.0 bundles **168 commits** from **28 contributors**.

---

## ✨ New & Improved

| Area | What's new |
|------|------------|
| **Type Checking** | - Partial support for `TypeForm` (PEP 747). <br><br>- Enum bug fixes: member types are preserved as literals instead of being promoted away, and the `auto()` function infers its type from `_generate_next_value_` and mixed-in data types, rather than always defaulting to `int`. <br><br>- Annotated variables preserve their declared types over inferred `Any` types, preventing accidental loss of type information. |
| **Language Server** |  - Pyrefly now shows warnings for `InvalidAnnotation`, `MissingImport`, and `UnknownName` errors even when no configuration file is present, aligning with pyright's default diagnostic behavior. <br><br>- IDE features like hover, go-to-definition, and autocomplete work inside unannotated function bodies when `check-unannotated-defs = false`. Previously, these functions were completely skipped during analysis, leaving you without any IDE support. <br><br>- The `provide-type` endpoint now returns correct types for operator expressions and properly qualifies type alias names with their module. <br><br>- Code lens "Run" and "Test" commands are available for `if __name__ == "__main__"` blocks and pytest/unittest test methods, enabling one-click execution from the editor. |
| **Error Messages** | - The `untyped-import` error has been upgraded from "ignore" to "warn" by default. <br><br>- The summary line in `pyrefly check` output now shows how many warnings or info messages were hidden when using `--min-severity`, so you'll know if diagnostics were filtered out (e.g., "INFO 0 errors (12 warnings not shown)"). <br><br>- You can now place `# pyrefly: ignore-errors` directives after module docstrings, not just at the very top of the file. <br><br>- Suppression comments work correctly with backslash line continuations. The suppression above the first line of a backslash-continued expression applies to the entire expression, matching the behavior for multi-line strings. <br><br>- The `pyrefly report` command recognizes suppression comments from more tools (`# type: ignore`, `# pyrefly: ignore`, `# pyright: ignore`, `# mypy: type: ignore`, `# pyre-fixme`, `# ty: ignore`, and `# zuban: ignore`), making the report more comprehensive when analyzing codebases that use multiple type checkers. |
| **Performance** | - Reduced wall time by 55% on a colour-science/colour benchmark by caching protocol subset checks and avoiding expensive clones. |

---

## 🐛 bug fixes

We closed **27** bug issues this release 👏

- #2976: Fixed an issue where `yield from` with a union return annotation (e.g., `Iterator[tuple[Any, ...]] | Generator[dict[str, Any], None, Unknown]`) incorrectly reported `invalid-yield` because the union members were not being decomposed before checking assignability.
- #2968: Fixed a false positive `bad-argument-type` error when using metaclasses — `type[Banana | Grape]` is now correctly assignable to `FruitMeta` when `Banana` and `Grape` inherit from `Fruit(metaclass=FrustMeta)`.
- #2402: Fixed an issue where Pyrefly excluded all files when the project lived inside a hidden directory (e.g., `~/.codex/worktrees/XXXX/project/`) because the `**/.[!/.]*/**` glob pattern matched hidden directory components anywhere in the absolute path, including ancestors above the project root.
- #2863: Fixed an issue where bounded type variables in match subjects resolved to their bound instead of the actual narrowed type — tuple match subjects like `match x, y:` now correctly narrow individual variables by their corresponding sub-patterns.
- #2980: Fixed an issue where `namedtuple` with a field named `cls` caused type warnings because the field collided with the synthesized `__new__` method's first parameter — the internal parameter is now `_cls`.
- #2922: Fixed an issue where applying `@dataclass` to an `Enum` subclass was not rejected — this is now correctly reported as `BadClassDefinition` because Python's `dataclasses` module does not support Enum subclasses (runtime `TypeError`).
- #2923: Fixed an issue where applying `@dataclass` to a `TypedDict` subclass was not rejected — this is now correctly reported in the canonical `class_metadata_of` path, and `dataclass_metadata` is cleared after reporting the error.
- #2921: Fixed an issue where applying `@dataclass` to a `Protocol` subclass was not rejected — this is now correctly reported because Protocol classes define structural interfaces, not data containers.
- #2975: Fixed an issue where the `provide-type` endpoint returned `Unknown` for type aliases — operator expressions now return result types instead of dunder method signatures, and type alias names in function signatures are now module-qualified.
- #2982: Fixed a stack overflow crash when checking fuzzed code with mutually recursive type aliases like `type T = U; type U = T` — when a cycle is detected, `wrap_type_alias` now returns an error type instead of the cyclic body.
- And more! #2987, #2610, #2874, #2857, #1950, #2741, #2983, #2919, #2979, #2973, #1982, #2910, #2950, #2978, #2875, #2986, #2972, #2852

Thank-you to all our contributors who found these bugs and reported them! Did you know this is one of the most helpful contributions you can make to an open-source project? If you find any bugs in Pyrefly we want to know about them! Please open a bug report issue [here](https://github.com/facebook/pyrefly/issues)

---

## 📦 Upgrade

```bash
pip install --upgrade pyrefly==0.60.0
```

### How to safely upgrade your codebase

Upgrading the version of Pyrefly you're using or a third-party library you depend on can reveal new type errors in your code. Fixing them all at once is often unrealistic. We've written scripts to help you temporarily silence them. After upgrading, follow these steps:

1. `pyrefly check --suppress-errors`
2. run your code formatter of choice
3. `pyrefly check --remove-unused-ignores`
4. Repeat until you achieve a clean formatting run and a clean type check.

This will add `# pyrefly: ignore` comments to your code, enabling you to silence errors and return to fix them later. This can make the process of upgrading a large codebase much more manageable.

Read more about error suppressions in the [Pyrefly documentation](https://pyrefly.org/en/docs/error-suppressions/)

---

## 🖊️ Contributors this release

@stroxler, @yangdanny97, @migeed-z, @rchen152, @javabster, @asukaminato0721, @kinto0, @avikchaudhuri, @arthaud, @tejasreddyvepala, @lolpack, @fangyi-zhou, @Louisvranderick, @runlevel5, @yeetypete, @NathanTempest, @stanleyshen2003, Morgan Bartholomew, @connernilsen, @salvatorebenedetto, @kshitijgetsac, @Raf-Hs, Mick Killianey, David Tolnay, @dhleong, @Arths17

---

Please note: These release notes summarize major updates and features. For brevity, not all individual commits are listed. Highlights from patch release changes that were shipped after the previous minor release are incorporated here as well.
