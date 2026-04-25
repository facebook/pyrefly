# Pyrefly Release Notes (v0.28.0 - v0.57.0)

Combined release notes for all minor versions of Pyrefly from v0.28.0 to v0.57.0.

## Table of Contents

- [v0.28.0](#pyrefly-v0.28.0)
- [v0.29.0](#pyrefly-v0.29.0)
- [v0.30.0](#pyrefly-v0.30.0)
- [v0.31.0](#pyrefly-v0.31.0)
- [v0.32.0](#pyrefly-v0.32.0)
- [v0.33.0](#pyrefly-v0.33.0)
- [v0.34.0](#pyrefly-v0.34.0)
- [v0.35.0](#pyrefly-v0.35.0)
- [v0.36.0](#pyrefly-v0.36.0)
- [v0.37.0](#pyrefly-v0.37.0)
- [v0.40.0](#pyrefly-v0.40.0)
- [v0.41.0](#pyrefly-v0.41.0)
- [v0.42.0](#pyrefly-v0.42.0)
- [v0.43.0](#pyrefly-v0.43.0)
- [v0.44.0](#pyrefly-v0.44.0)
- [v0.45.0](#pyrefly-v0.45.0)
- [v0.46.0](#pyrefly-v0.46.0)
- [v0.47.0](#pyrefly-v0.47.0)
- [v0.49.0](#pyrefly-v0.49.0)
- [v0.50.0](#pyrefly-v0.50.0)
- [v0.51.0](#pyrefly-v0.51.0)
- [v0.52.0](#pyrefly-v0.52.0)
- [v0.53.0](#pyrefly-v0.53.0)
- [v0.54.0](#pyrefly-v0.54.0)
- [v0.55.0](#pyrefly-v0.55.0)
- [v0.56.0](#pyrefly-v0.56.0)
- [v0.57.0](#pyrefly-v0.57.0)

---

# Pyrefly v0.28.0

**Status:** ALPHA
**Release date:** 11 August 2025

pyrefly v0.28 bundles **119 commits from 20 contributors**. This release introduces new features related to function and class metadata, improves class base type calculations, enhances import finding logic, and addresses several bug fixes related to LSP behavior, error reporting, and type inference.

## New & Improved

| Area | What's new |
| :---- | :---- |
| **Type Checker** | Export function information including names and locations, enabling Pysa to use pyrefly's function list for consistency. Add parent information to functions and classes to facilitate unique fully qualified name generation in Pysa. Introduce function flags (e.g., is_overload, is_classmethod) for richer metadata. |
| **Class Metadata & Type Inference** | Support Pydantic metadata by adding it to class metadata. |
| **Import Resolution** | Implement preference for .pyi files over .py over .pyc files in import finding, even if the import occurs later in the includes list. This is particularly useful for go-to-definition in the LSP, preferring implementation files over interfaces. |
| **Error Reporting** | Add a check for unused coroutines, similar to pyright and mypy, to flag coroutines that are not awaited, passed to a function, or assigned to a variable. |
| **LSP** | Improvements to LSP shutdown behavior, ensuring the language server exits gracefully on connection close and removing previous workarounds for exit notifications. |
| **Configuration** | Add support for ignore files (.gitignore, .ignore, .git/exclude) with a config (use-ignore-files) to enable and disable their use. |
| **Website Sandbox** | Added a dropdown to select the Python version in the sandbox, allowing users to test against different Python versions (3.8-3.12). |

## Bug Fixes

- Fix handling of args and kwargs for lambda functions, where they were incorrectly treated as position arguments.
- Address autocomplete issue where redeclarations were not appearing (reported as #818).
- Fix issues with completions (#850).
- Filter out synthetic narrow keys from named_bindings (#851, #841).
- Improve error suppression logic to handle suppressions on the line above or same line as the error.
- Fix overload selection involving type dict field errors, addressing a bug where an overload that should fail was erroneously selected.
- Fix for #811, where while True with no breaks treated as non-terminating.

## Potentially Breaking

If you are using Pyrefly in your CI processes, the following changes may cause your CI to break when upgrading to the new version of Pyrefly:

- `pyrefly autotype` renamed to `pyrefly infer` to reduce confusion with existing tools.

## Upgrade

```
pip install --upgrade pyrefly==0.28
```

## Contributors this release

Aaron Pollack, Abby Mitchell, Adist319, Carlos Fernandez, Conner Nilsen, Danny Yang, Dogac Eldenk, Fangyi Zhou, Jia Chen, Kyle Into, Maggie Moss, Maxime Arthaud, Neil Mitchell, Rebecca Chen, Rubmary Rojas, Sam Goldman, Sam Zhou, Xavi Simpson, Zeina Migeed

---


# Pyrefly v0.29.0

**Status:** ALPHA
**Release date:** 18 August 2025

Pyrefly 0.29 bundles **192** commits from **23** contributors. This release focuses on improving the type checker's behavior, particularly around handling Any and LiteralString, enhancing LSP features like "go to definition" and hover, and various internal refactors for maintainability.

## New & Improved

| Area | What's new |
| :---- | :---- |
| **Type Checker Behavior** | The type checker now treats `return` outside of a function, and `await`, `async for`, `async with` outside of async function definitions as syntax errors. Support special `LiteralString` behavior for methods like `str.join` and `str.format`. `Any` handling for `super().__init__` and `super().__new__` calls have been improved. `typing.Self` is now correctly substituted inside method bodies. Using the new PEP 728 keywords (`closed`, `extra_items`) in `TypedDict` no longer throws type errors. Full support for PEP 728 is coming soon. |
| **LSP Features** | "Go to definition" now prioritizes `.py` files over `.pyi` for a more useful real implementation view. Docstrings now appear for standard library functions. Display ranges for generator expressions have been adjusted for consistency with Python AST. |

## Bug Fixes

We closed **21** bug issues this release

- #940 - cannot use items as a key in a TypedDict
- #915 - incorrect autocomplete suggestions for read_database
- #872 - false positive inside while loops
- #853 - excessive memory usage in VScode due to multiple processes being spawned
- Too many panics, not enough discos:
  - #864 - panic for helion with pytorch installed
  - #848 - panic key lacking binding
- Other bug fixes: #931, #833, #815, #748, #725, #655, #906, #905, #901, #895, #894, #891, #889, #931, #922

## Upgrade

```
pip install --upgrade pyrefly==0.29
```

### How to safely upgrade your codebase

Upgrading the version of Pyrefly you're using or a third-party library you depend on can reveal new type errors in your code. Fixing them all at once is often unrealistic. We've written scripts to help you temporarily silence them. After upgrading, follow these steps:

1. `pyrefly check --suppress-errors`
2. run your code formatter of choice
3. `pyrefly check --remove-unused-ignores`
4. Repeat until you achieve a clean formatting run and a clean type check.

This will add `# pyrefly: ignore` comments to your code, enabling you to silence errors and return to fix them later. This can make the process of upgrading a large codebase much more manageable.

Read more about error suppressions in the [Pyrefly documentation](https://pyrefly.org/en/docs/error-suppressions/)

## Contributors this release

Aahan Aggarwal, Abby Mitchell, Conner Nilsen, Danny Yang, David Luo, Fangyi Zhou, James Gill, Kc Balusu, Kyle Into, LyricalToxic, Maggie Moss, Maxime Arthaud, Neil Mitchell, Rebecca Chen, Rubmary Rojas, Sam Goldman, Sam Zhou, Steven Troxler, Vladimir Matveev, Xavi Simpson, Zach Riser, Zeina Migeed

---

# Pyrefly v0.30.0

**Status:** ALPHA
**Release date:** 25 August 2025

pyrefly v0.30.0 bundles **195 commits** from **20 contributors**

## New & Improved

| Area | What's new |
| :---- | :---- |
| **Completions** | Enhanced completions for `super` with multiple inheritance implemented completion for unions. When the attribute base is a union, suggestions are now combined from every branch of the union. This might lead to errors, but it's more helpful to make the suggestion. |
| **Type System** | Added a configuration flag to turn off "infer from first use" behavior. Inferring types from first usage is valuable for catching bugs but it can lead to false positives so this new flag allows users to turn it off. Turning it off behaves like pyright, leaving it on behaves like mypy. Implicit attribute definition errors are now turned off by default. These errors were raised when an attribute is only declared in a non-constructor method, but it can be a source of false positives in dynamic code. When type args are not explicitly set, push the class type params into methods (matches mypy approach, differs from pyright that implicitly instantiates with `Any`/default). Ensured qualified names are printed for enums as part of unions. Enable creation of empty enums and named tuples using functional syntax. Default excludes now align with Pylance (incl. `Node_modules`, `__pycache__` and `*venv`). |

## Bug Fixes

We closed **11** bug issues this release

- #984 - empty enums incorrectly given type `Never`, rather than `Any`
- #865 - completion not working for Pytorch `torch.zeros` module
- #514 - issue returning generic `Self`
- Other bug fixes: #971, #967, #743, #567, #348, #261, #186, #103

## Upgrade

```
pip install --upgrade pyrefly==0.30
```

### How to safely upgrade your codebase

1. `pyrefly check --suppress-errors`
2. run your code formatter of choice
3. `pyrefly check --remove-unused-ignores`
4. Repeat until you achieve a clean formatting run and a clean type check.

This will add `# pyrefly: ignore` comments to your code, enabling you to silence errors and return to fix them later.

Read more about error suppressions in the [Pyrefly documentation](https://pyrefly.org/en/docs/error-suppressions/)

## Contributors this release

@javabster, @capickett, Carlos Fernandez, @connernilsen, @yangdanny97, @grievejia, @kinto0, @arthaud, @prasannavenkateshgit, @rchen152, @rchiodo, @rubmary, @samwgoldman, @SamChou19815, @sargun, @stroxler, @VladimirMakaev, @migeed-z, @LycheeBay

---

# Pyrefly v0.31.0

**Status:** ALPHA
**Release date:** 02 September 2025

Pyrefly v0.31.0 bundles **69 commits** from **19 contributors**

## New & Improved

| Area | What's new |
| :---- | :---- |
| **PEP 728 support** | `extra_items` is now fully supported, with these changes support for PEP 728 is now complete! |
| **LSP Features** | Introduced Pylance config `python.analysis.importFormat`, with possible values `"absolute"` and `"relative"`. The default `"absolute"` maintains the current behavior, while `"relative"` enables importing modules using relative paths. Server name and version string now returned during LSP initialisation, useful for LSP clients for debugging/logging. |
| **Type Narrowing** | Improved narrowing for `getattr(...)`. Narrow `hasattr` with string literals to `Any` if the attribute does not already exist. |
| **Inference & autocomplete** | Improved inference for `self` annotation: inferred to be `type[Self]` when decorated with `classproperty` or `lazy_classproperty`. Improved autocomplete for imports with addition of `import` keywords. (Previously `from x imp...` would autocomplete to `from x import imp`) |
| **Build System Integration** | Standalone pre-commit hook now available, see updated docs [here](https://pyrefly.org/en/docs/installation/#pre-commit). Refactored Buck build functionality into a new `pyrefly_build` crate. This does not involve functional changes but lays the groundwork for future work to enable Pyrefly to work more effectively with various build systems in both OSS and IDE environments. |
| **Conformance** | Pyrefly has now been added to the [conformance test suite](https://github.com/python/typing/blob/main/conformance/results/results.html), enabling users to compare behaviour of static type checkers against expectations defined in the Python typing specification. |

## Bug Fixes

We closed 12 bug issues this release

- A few issues with decorators:
  - #951 - pyrefly unable to find matching overloads for decorators with arguments
  - #809 - failure to apply self
  - #625
- #921 - incorrect type signature labelled for `type[T].__new__`
- #509 - panic on `__all__` mutation without definition
- #322 - error thrown falsely on `unittest.main()`
- #71 - issue iterating over enums
- Other bugs: #932, #926, #765, #245, #162

## Upgrade

```
pip install --upgrade pyrefly==0.31.0
```

### How to safely upgrade your codebase

1. `pyrefly check --suppress-errors`
2. run your code formatter of choice
3. `pyrefly check --remove-unused-ignores`
4. Repeat until you achieve a clean formatting run and a clean type check.

This will add `# pyrefly: ignore` comments to your code, enabling you to silence errors and return to fix them later.

Read more about error suppressions in the [Pyrefly documentation](https://pyrefly.org/en/docs/error-suppressions/)

## Contributors this release

@lolpack, @aahanaggarwal, @donsbot, @kinto0, @stroxler, @rchen152, @grievejia, @brosenfeld, @yangdanny97, @SamChou19815, @migeed-z, @connernilsen, @tianhan0, @javabster, @PIG208, @samwgoldman, @IDrokin117, @maggiemoss, @arthaud, @melvinhe, @rchiodo, @VladimirMakaev

---

# Pyrefly v0.32.0

**Status:** ALPHA
**Release date:** 08 September 2025

Pyrefly v0.32.0 bundles **70 commits** from **18 contributors**

## Major Change for IDE Extension

With this release **type errors are now disabled by default in IDE** when there is no config present (in `pyrefly.toml` or `pyproject.toml`). This change was made based on feedback that displaying type errors by default in the IDE (i.e. red squiggles) was a nuisance for new users, especially those looking only for language services. Users are still able to enable type checking without a configuration file if they wish by updating their config (either in their `pyproject.toml`, `pyrefly.toml` or VSCode settings):

```json
# to enable type errors
"python.pyrefly.displayTypeErrors": "force-on"
```

## New & Improved

| Area | What's new |
| :---- | :---- |
| **Type Checker Behavior** | Implicit return validation is now performed even when `untyped-def-behavior` is set to `check-and-infer-return-any` for functions with explicit return annotations. This fixes a conflation issue where return type inference being disabled also disabled implicit return validation. Improved handling of calls on `type[...]` including `cls` (typed as `type[Self]`). Deprecation warnings are now emitted when a deprecated function is referred to, not just called. Scoping for class fields in nested scopes is now correctly handled. |
| **IDE Integration** | Type errors now disabled by default in IDE (see details above). Users can now see in their IDE status bar whether the current file has type errors enabled/disabled. Improved hover for Union types, so hover display now merges identical elements. |
| **Build Systems & Security** | Further improvements made to emulate Buck's build system file mappings within Pyrefly. This improves Pyrefly's integration with various build systems in OSS and IDEs, though build system support is still in progress. The `tracing-subscriber` crate was updated from 0.3.19 to 0.3.20 to fix `RUSTSEC-2025-0055`, addressing a vulnerability related to ANSI escape sequence injection attacks in logs. |

## Bug Fixes

We've closed 10 bug issues since our last minor release

- #1036 - class property on enums incorrectly typed as `Literal`
- #1016 - pyrefly process not terminating and consuming significant CPU and memory
- #977 - issue handling attributes in dataclasses
- Panics resolved: #509, #962
- Other bugs fixed: #980, #647, #264

## Upgrade

```
pip install --upgrade pyrefly==0.32.0
```

### How to safely upgrade your codebase

1. `pyrefly check --suppress-errors`
2. run your code formatter of choice
3. `pyrefly check --remove-unused-ignores`
4. Repeat until you achieve a clean formatting run and a clean type check.

This will add `# pyrefly: ignore` comments to your code, enabling you to silence errors and return to fix them later.

Read more about error suppressions in the [Pyrefly documentation](https://pyrefly.org/en/docs/error-suppressions/)

## Contributors this release

@rchen152, @dtolnay, @kinto0, @samwgoldman, @SamChou19815, @grievejia, @migeed-z, @yangdanny97, @arthaud, @tianhan0, @AryanBagade, @asukaminato0721, Carlos Fernandez, @connernilsen, @fangyi-zhou, @hashiranhar, @VladimirMakaev

---

# Pyrefly v0.33.0

**Status:** ALPHA
**Release date:** 15 September 2025

Pyrefly v0.33.0 bundles **117 commits** from **21 contributors**

## New & Improved

| Area | What's new |
| :---- | :---- |
| **Type Checker Behavior** | Pyrefly now distinguishes between `Any` as a value and as a type, preventing incorrect attribute access and constructor calls on `Any` values. For example, when `Any` (the value) is used as an attribute base or call target pyrefly will now error if you try to access attributes that are not defined on `type`. Basic support added for `TypeAliasType`. Experimental support for Pydantic now available! Type error messages by default now print paths relative to working directory (as opposed to absolute path). |
| **IDE improvements** | Kwarg completion now supports literals and unions of literals ([requested feature](https://github.com/facebook/pyrefly/issues/1064)). A new "typeServer/getSupportedProtocol" request handler included to fetch TSP version information. |
| **Website and Documentation** | The sandbox feature on the website has been upgraded! It can now deal with multiple files, allowing users to try out Pyrefly with cross-file imports. Give it a go [here](https://pyrefly.org/sandbox/)! New documentation guide on experimental Pydantic support, read it [here](https://pyrefly.org/en/docs/pydantic/). New editable installs section in documentation, detailing the nuances of working with editable dependencies. Read it [here](https://pyrefly.org/en/docs/import-resolution/#editable-installs). |

## Bug Fixes

We closed 4 bug issues this release

- #1052 - regression that caused issue with return type `list[Self]`
- #1034 - Python version dropdown in website sandbox not working properly
- #817 - `bad-override` error split into 2 separate error codes for type and name differences in arguments
- #573 - when a package is pip-installed as editable from a local directory as a dependency, it was added in a way that Pyrefly didn't recognise

## Upgrade

```
pip install --upgrade pyrefly==0.33.0
```

### How to safely upgrade your codebase

1. `pyrefly check --suppress-errors`
2. run your code formatter of choice
3. `pyrefly check --remove-unused-ignores`
4. Repeat until you achieve a clean formatting run and a clean type check.

This will add `# pyrefly: ignore` comments to your code, enabling you to silence errors and return to fix them later.

Read more about error suppressions in the [Pyrefly documentation](https://pyrefly.org/en/docs/error-suppressions/)

## Contributors this release

@javabster, @stroxler, @grievejia, @ndmitchell, @fangyi-zhou, @samwgoldman, @rchen152, @kinto0, @connernilsen, @AryanBagade, Brian Rosenfeld, Carlos Fernandez, @yangdanny97, @hashiranhar, @IDrokin117, @maggiemoss, @rchiodo, @VladimirMakaev, @migeed-z

---

# Pyrefly v0.34.0

**Status:** ALPHA
**Release date:** 22 September 2025

Pyrefly v0.34.0 bundles **180 commits** from **22 contributors**.

## New & Improved

| Area | What's new |
| :---- | :---- |
| **Type Parameter Handling** | Significant refactoring of internal logic for handling uninitialised variables, resolving multiple bugs in the process. Support added for referring to imported legacy `TypeVar`s by their fully qualified name ([see example in sandbox](https://pyrefly.org/sandbox/?project=N4IgZglgNgpgziAXKOBDAdgEwEYHsAeAdAA4CeS4ATrgLYAEALqcROgOZ0Q3G6UN2UYANxiooAfSbEYAHUoz0XHnzoBXBtDhyFmGGDpgAFPkRqNUOIQAqASjoBaAHxnN1xNvR0vAmA1WVPfA9BETFJZhhDIwAGGxsQABoQdVcyCjBqeilWDiVefisIgDVUeXQrOgBeOkLpEspDGRArJviAXyTUAGMNEQAxaBgKNCw8IjS2oA&version=3.12)). Support added for implementing protocols with generic methods. |
| **IDE** | IDE logic has been refactored to limit concurrent recheck and find-references tasks, preventing excessive thread spawning under heavy workload and improving overall IDE responsiveness. IDE autocomplete now displays if a method or function is deprecated. IDE autocomplete now escapes single quotes (`''`) for autocomplete. |
| **Pyrefly Configs** | Users can now use negation when setting `replace-imports-with-any`. Eg. `replace-imports-with-any: ["!sympy.printer", "sympy.*"]` will replace any sympy imports with `Any` EXCEPT for sympy printer imports. |
| **Error Messaging** | Improved how overload signatures in error messages are displayed to avoid printing confusing upper bounds of TypeVars. |
| **Documentation** | IDE documentation updated to include Jupyter Lab information and improved inlay hint instructions. |

## Bug Fixes

We closed 21 bug issues this release

- #1058 - `abstractmethod` decorated functions issue caused literal types in signatures to be promoted incorrectly
- Addressed issue that caused false positives when using the jaxtyping library: #925
- Addressed multiple issues relating to `TypeVar`s: #1050, #912, #129
- Addressed multiple issues with IDE autocomplete: #1095, #986, #798
- #200 - pyrefly `--watch` CLI command now works again!
- Other issues: #1073, #947, #936, #829, #825, #724, #698, #680, #604, #429, #340, #114, #111

## Upgrade

```
pip install --upgrade pyrefly==0.34.0
```

### How to safely upgrade your codebase

1. `pyrefly check --suppress-errors`
2. run your code formatter of choice
3. `pyrefly check --remove-unused-ignores`
4. Repeat until you achieve a clean formatting run and a clean type check.

This will add `# pyrefly: ignore` comments to your code, enabling you to silence errors and return to fix them later.

Read more about error suppressions in the [Pyrefly documentation](https://pyrefly.org/en/docs/error-suppressions/)

## Contributors this release

@rchen152, @tianhan0, @ndmitchell, @yangdanny97, @stroxler, @arthaud, @SamChou19815, @grievejia, @javabster, @connernilsen, @cooperlees, @dluo, @dtolnay, @fangyi-zhou, @kinto0, @maggiemoss, Paul Iatchenii, @samwgoldman, @sandeshbhusal, @VladimirMakaev, @migeed-z

---

# Pyrefly v0.35.0

**Status:** ALPHA
**Release date:** 29 September 2025

Pyrefly v0.35.0 bundles **226 commits** from **18 contributors**.

## New & Improved

| Area | What's new |
|------|-----------|
| **Configuration** | Improved config finding, preferring Pyrefly configurations in any parent directory to project root marker files. `pyproject.toml` files are now only considered configurations when a `[tool.pyrefly]` section is present, otherwise, they are root marker files. |
| **Build systems** | Continued work on build system support, primarily Buck. Look out for further updates coming soon! |
| **Error messages and suppressions** | Improved `reveal_type`/`assert_type` error messages to suggest importing from `typing` when needed, rather than a generic "could not find name" error message. New error kind introduced: `inconsistent-overload`. This gives overload consistency errors a separate error kind from other invalid overload errors so that users can turn off consistency checks if they wish. `suppress-error` command now has optional `--same-line` argument for easier bulk suppression of errors. Instead of adding `pyrefly ignore` comments on the line above, this flag adds the comment on the same line as where the error is. |
| **Security** | Upgraded `mdbook` crate to 0.4.52 to resolve vulnerability with ammonia dependency. |
| **Third party package support** | Improved handling of optional Pydantic fields (if they are optional we should not require them). Foundational work to support Django enums. |

## Bug Fixes

We closed 11 bug issues this release.

- [#1167](https://github.com/facebook/pyrefly/issues/1167), [#1166](https://github.com/facebook/pyrefly/issues/1166) - inaccuracies with `type(x)` returning the wrong thing
- [#954](https://github.com/facebook/pyrefly/issues/954) - fixed an issue with Pyrefly not recognising `Session` context manager from SQL Alchemy. Fixing this issue brings us one step closer to Pyrefly being fully usable with SQL Alchemy (follow progress [here](https://github.com/facebook/pyrefly/issues/920))
- [#260](https://github.com/facebook/pyrefly/issues/260) - unions of dunder methods weren't being recognised correctly
- And more - [#1165](https://github.com/facebook/pyrefly/issues/1165), [#1148](https://github.com/facebook/pyrefly/issues/1148), [#1146](https://github.com/facebook/pyrefly/issues/1146), [#1030](https://github.com/facebook/pyrefly/issues/1030), [#614](https://github.com/facebook/pyrefly/issues/614), [#527](https://github.com/facebook/pyrefly/issues/527), [#300](https://github.com/facebook/pyrefly/issues/300)

## Upgrade

```
pip install --upgrade pyrefly==0.35.0
```

### How to safely upgrade your codebase

1. `pyrefly check --suppress-errors`
2. run your code formatter of choice
3. `pyrefly check --remove-unused-ignores`
4. Repeat until you achieve a clean formatting run and a clean type check.

This will add `# pyrefly: ignore` comments to your code, enabling you to silence errors and return to fix them later.

Read more about error suppressions in the [Pyrefly documentation](https://pyrefly.org/en/docs/error-suppressions/)

## Contributors this release

[@javabster](https://github.com/javabster), [@samwgoldman](https://github.com/samwgoldman), [@grievejia](https://github.com/grievejia), [@kinto0](https://github.com/kinto0), [@rchen152](https://github.com/rchen152), [@yangdanny97](https://github.com/yangdanny97), [@tianhan0](https://github.com/tianhan0), [@stroxler](https://github.com/stroxler), [@SamChou19815](https://github.com/SamChou19815), Carlos Fernandez, [@cjlongoria](https://github.com/cjlongoria), [@connernilsen](https://github.com/connernilsen), [@maggiemoss](https://github.com/maggiemoss), [@arthaud](https://github.com/arthaud), [@mohesham88](https://github.com/mohesham88), [@VladimirMakaev](https://github.com/VladimirMakaev), [@migeed-z](https://github.com/migeed-z)

---

# Pyrefly v0.36.0

**Status:** ALPHA
**Release date:** 06 October 2025

Pyrefly v0.36.0 bundles **150 commits** from **23 contributors**. This release brings major improvements to performance, type checking, and Pydantic/Django support, along with numerous bug fixes and a brand new baseline feature.

## New & Improved

| Area | What's new |
|------|-----------|
| **Performance** | With #360 solved, Pyrefly is _much_ faster on projects & environments with a large number of dependencies. |
| **Baseline Feature** | A new experimental baseline mechanism to store existing errors in a separate file. See the docs for more info. |
| **Control Flow & Type Narrowing** | Major improvements to how we model control flow and type narrowing, reducing false-positive errors. Builtin types can now be narrowed to literals in more situations. |
| **LiteralString** | Added support for LiteralString in format(), join(), and replace() methods. |
| **Pydantic** | Enhancements and bug fixes were made to the experimental Pydantic support released last month. Pydantic's docs now contains documentation for how to use Pyrefly with Pydantic! |
| **Django** | Progress on experimental support for Django fields and enums. |
| **IDE** | Bug fixes for auto-import quick fix, improved hover info format to improve readability, reduce clutter, and enable syntax highlighting for function types. |
| **Sandbox** | Browser sandbox now supports Python stub files (.pyi). |

## Bug Fixes

We closed 20 bug issues this release.

#264, #360, #528, #657, #683, #778, #790, #812, #969, #1009, #1022, #1042, #1088, #1162, #1198, #1210, #1218, #1227, #1228, #1235

## Upgrade

```
pip install --upgrade pyrefly==0.36.0
```

### How to safely upgrade your codebase

1. `pyrefly check --suppress-errors`
2. run your code formatter of choice
3. `pyrefly check --remove-unused-ignores`
4. Repeat until you achieve a clean formatting run and a clean type check.

This will add `# pyrefly: ignore` comments to your code, enabling you to silence errors and return to fix them later.

## Contributors this release

@stroxler, @arthaud, @samwgoldman, @connernilsen, @migeed-z, @rchen152, @grievejia, @ndmitchell, @yangdanny97, @AryanBagade, @rubmary, @cjlongoria, @simonhollis, @MarcoGorelli, @mohesham88, @Viicos, @Adist319, @airvzxf, @brianrosenfeld, @ivanloskutov, @ahornby, @kinto0

---

# Pyrefly v0.37.0

**Status:** ALPHA
**Release date:** 14 October 2025

Pyrefly 0.37.0 bundles **198 commits** from **15 contributors**. This release brings major improvements to typechecking control flow, overload resolution, Django/enum support, and more.

## New & Improved

| Area | What's new |
|------|-----------|
| **Control Flow & Type Narrowing** | Improve type narrowing across `if/else`, `match`, boolean ops, and loops. This enables more precise type inference and fixes several long-standing false positives and negatives. |
| **Overload Resolution** | Fully implemented the Python typing spec for overload call evaluation, including argument type expansion for unions, bools, enums, tuples, and `type`. Overloads are now filtered by argument count and variadic-ness, with better error messages and performance. |
| **Django & Enum Support** | Enhanced Django model field type inference, improved enum attribute handling (including Django enums), and fixed bugs with `auto()`, `Choices`, and mixed-in enum data types. |
| **LSP & Editor Features** | Added inlay hints for function argument names, docstrings for attribute and variable completions, and improved workspace folder detection for LSP clients. |

## Bug Fixes

We closed 12 bug issues this release.

## Upgrade

```
pip install --upgrade pyrefly==0.37.0
```

### How to safely upgrade your codebase

1. `pyrefly check --suppress-errors`
2. run your code formatter of choice
3. `pyrefly check --remove-unused-ignores`
4. Repeat until you achieve a clean formatting run and a clean type check.

This will add `# pyrefly: ignore` comments to your code, enabling you to silence errors and return to fix them later.

Read more about error suppressions in the [Pyrefly documentation](https://pyrefly.org/en/docs/error-suppressions/)

## Contributors this release

[@stroxler](https://github.com/stroxler), [@yangdanny97](https://github.com/yangdanny97), [@arthaud](https://github.com/arthaud), [@connernilsen](https://github.com/connernilsen), [@rchen152](https://github.com/rchen152), [@migeed-z](https://github.com/migeed-z), [@grievejia](https://github.com/grievejia), [@fangyi-zhou](https://github.com/fangyi-zhou), [@kinto0](https://github.com/kinto0), [@fatelei](https://github.com/fatelei), [@ndmitchell](https://github.com/ndmitchell), [@brosenfeld](https://github.com/brosenfeld), [@dtolnay](https://github.com/dtolnay), [@john](https://github.com/john) Van Schultz

**Full changelog:** [Comparing 0.36.0...0.37.0](https://github.com/facebook/pyrefly/compare/0.36.0...0.37.0)

---

# Pyrefly v0.40.0

**Status:** ALPHA
**Release date:** 03 November 2025

Pyrefly v0.40.0 bundles **143 commits** from **26 contributors**.

## New & Improved

### Language Server
- Go-to definition / hover to overloads on operators like `==`
- Fold on docstrings
- New LSP configuration arguments added allowing users to enable/disable specific IDE features like Go-to-definition, Autocomplete, and Hover tooltips. See docs [here](https://pyrefly.org/en/docs/IDE/#customization) for details.
- Unreachable code will now be grayed out.
- Fixed bug with symlinked paths in VSCode

### Type Resolution
Implemented a `TypeCache` to significantly optimize `is_subtype` queries by caching resolved types. This leads to faster type checks and a more responsive IDE experience.

### Error Reporting
Introduced a new `FindError` that will inform a user when there is a stub available for a package that they do not have installed. The default severity level for this will be a warning.

### Django Integration
Continued work on Django support, including improved type inference for lazy enum labels.

### Jupyter Notebook support
Basic type checking support (`pyrefly check path/to/file.ipynb`) now available for jupyter notebooks. Still very experimental, further improvements coming soon.

## Bug Fixes

We closed 15 bug issues this release.

- [#1414](https://github.com/facebook/pyrefly/issues/1414) - Ensured `typing_extensions.dataclass_transform` works correctly.
- [#1351](https://github.com/facebook/pyrefly/issues/1351) - Correctly accept `TypedDict` as `Partial` when unpacking one `TypedDict` into another and detect more open-unpacking errors.
- [#1257](https://github.com/facebook/pyrefly/issues/1257) - Addressed a crash related to `Var::ZERO` during autocomplete by using `deep_force` to replace unsolved variables with `Unknown`.
- [#1104](https://github.com/facebook/pyrefly/issues/1104) - Resolved an issue where the VSCode extension did not highlight problems in files with symlinked paths under Linux.
- [#1045](https://github.com/facebook/pyrefly/issues/1045) - Addressed a false-positive in `raise..from` statements by correctly handling `None` when it appears in a union.
- And more: [#1412](https://github.com/facebook/pyrefly/issues/1412), [#1409](https://github.com/facebook/pyrefly/issues/1409), [#1405](https://github.com/facebook/pyrefly/issues/1405), [#1396](https://github.com/facebook/pyrefly/issues/1396), [#1356](https://github.com/facebook/pyrefly/issues/1356), [#1294](https://github.com/facebook/pyrefly/issues/1294), [#1288](https://github.com/facebook/pyrefly/issues/1288), [#1234](https://github.com/facebook/pyrefly/issues/1234), [#1145](https://github.com/facebook/pyrefly/issues/1145), [#1119](https://github.com/facebook/pyrefly/issues/1119)

## Upgrade

```
pip install --upgrade pyrefly==0.40.0
```

### How to safely upgrade your codebase

1. `pyrefly check --suppress-errors`
2. run your code formatter of choice
3. `pyrefly check --remove-unused-ignores`
4. Repeat until you achieve a clean formatting run and a clean type check.

This will add `# pyrefly: ignore` comments to your code, enabling you to silence errors and return to fix them later.

Read more about error suppressions in the [Pyrefly documentation](https://pyrefly.org/en/docs/error-suppressions/)

---

# Pyrefly v0.41.0

**Status:** ALPHA
**Release date:** 10 November 2025

Pyrefly v0.41.0 bundles **166 commits** from **22 contributors**.

## New & Improved

| Area | What's new |
|------|-----------|
| **Jupyter Notebook support** | Extensive improvements for language server support in Jupyter Notebooks including diagnostics (red squiggles), inlay hints, go-to-definition, hover, semantic tokens, signature help, and completions. No extra configuration required to enable language server for notebook files (`.ipynb`), they are now included by default. |
| **Type Checking** | Pyrefly now correctly models and narrows types for variables reassigned within loops, fixing several long-standing issues (`#726`, `#747`). Improved pattern matching support for certain built-in types, better aligning with typing specification. More accurate type checking for `TypedDicts` with union keys. |
| **Language Server** | Hover cards on variables now link to built in types. Unused parameters are now flagged in IDE. Deprecated functions now suggested below non-deprecated ones in auto-completions. Inlay hints are suppressed for variables with unknown or `Any` types to reduce noise in IDE. |
| **Error messages** | Users can now use `--info=<error-code>` command line flag to set error severity to `info` for specific error codes. Users can now filter summarized errors by specific error codes using `pyrefly check --summarize-errors --only <error-code>`. |

## Bug Fixes

We closed 15 bug issues this release.

- [#1505](https://github.com/facebook/pyrefly/issues/1505) - fixed incorrect `bad-assignment` error inside loop with `dict.get`
- [#1411](https://github.com/facebook/pyrefly/issues/1411) - Fixed `bad-argument-type` false positive when a parameter type was inferred as `Never`
- [#1327](https://github.com/facebook/pyrefly/issues/1327) - Fixed issue where Hover did not show docstring for nested classes
- [#1184](https://github.com/facebook/pyrefly/issues/1184) - Fixed a bug where `NewType` was not treated as a type
- [#1301](https://github.com/facebook/pyrefly/issues/1301) - Fix for nested folder imports in the Pyrefly website sandbox
- [#1173](https://github.com/facebook/pyrefly/issues/1173) - Fixed issue where VSCode extension import autocomplete included hidden directories
- And more: [#1092](https://github.com/facebook/pyrefly/issues/1092), [#747](https://github.com/facebook/pyrefly/issues/747), [#726](https://github.com/facebook/pyrefly/issues/726), [#1407](https://github.com/facebook/pyrefly/issues/1407), [#1487](https://github.com/facebook/pyrefly/issues/1487), [#1420](https://github.com/facebook/pyrefly/issues/1420), [#1417](https://github.com/facebook/pyrefly/issues/1417), [#1536](https://github.com/facebook/pyrefly/issues/1536), [#1508](https://github.com/facebook/pyrefly/issues/1508)

## Upgrade

```
pip install --upgrade pyrefly==0.41.0
```

### How to safely upgrade your codebase

1. `pyrefly check --suppress-errors`
2. run your code formatter of choice
3. `pyrefly check --remove-unused-ignores`
4. Repeat until you achieve a clean formatting run and a clean type check.

This will add `# pyrefly: ignore` comments to your code, enabling you to silence errors and return to fix them later.

Read more about error suppressions in the [Pyrefly documentation](https://pyrefly.org/en/docs/error-suppressions/)

## Contributors this release

[@aaron-ang](https://github.com/aaron-ang), [@alokpr](https://github.com/alokpr), [@AryanBagade](https://github.com/AryanBagade), [@asukaminato0721](https://github.com/asukaminato0721), [@connernilsen](https://github.com/connernilsen), [@yangdanny97](https://github.com/yangdanny97), [@dtolnay](https://github.com/dtolnay), [@kv9898](https://github.com/kv9898), [@fangyi-zhou](https://github.com/fangyi-zhou), [@grievejia](https://github.com/grievejia), [@jvansch1](https://github.com/jvansch1), [@kshitijgetsac](https://github.com/kshitijgetsac), [@maggiemoss](https://github.com/maggiemoss), [@arthaud](https://github.com/arthaud), [@ndmitchell](https://github.com/ndmitchell), [@rchen152](https://github.com/rchen152), [@rubmary](https://github.com/rubmary), [@stroxler](https://github.com/stroxler), [@VladimirMakaev](https://github.com/VladimirMakaev), [@migeed-z](https://github.com/migeed-z)

---

# Pyrefly v0.42.0

**Status:** BETA
**Release date:** 17 November 2025

## Pyrefly is now in Beta!

We're thrilled to announce with this release that Pyrefly has transitioned from Alpha to Beta status! This marks a new chapter for the project, with increased stability, feature maturity, and readiness for broader adoption. When using a version of Pyrefly with Beta status you can feel confident that the IDE extension is ready for production use, while core type-checking features can be used, but be aware some edge cases are still being addressed as we make progress towards a later stable v1 release ([bug reports welcome](https://github.com/facebook/pyrefly/issues)!).

These release notes will use a slightly different format than usual, covering the major highlights of the last 6 months that have contributed to this milestone.

## Major Highlights from Alpha Releases

### 1. IDE & Language Server Features

- Instant IDE Startup: Major performance improvements for large environments.
- Semantic Highlighting, Inlay Hints: Enhanced code readability and added highlighting for unreachable blocks and unused function parameters. Unnecessary/obvious inlay hints are no longer displayed.
- Hover Cards & Docstrings: Richer information in tooltips, including docstrings for built-in types and links to error documentation.
- Configurable LSP Features: Greater control over enabling/disabling IDE features like autocomplete, go-to-definition, and hover.
- Extensions released for a range of popular IDEs, including [VSCode](https://marketplace.visualstudio.com/items?itemName=meta.pyrefly), [PyCharm](https://www.jetbrains.com/help/pycharm/2025.3/lsp-tools.html#pyrefly) and [more](https://pyrefly.org/en/docs/IDE/#other-editors)!
- Some relevant issues: [#853](https://github.com/facebook/pyrefly/issues/853), [#1403](https://github.com/facebook/pyrefly/issues/1403), [#1335](https://github.com/facebook/pyrefly/issues/1335), [#873](https://github.com/facebook/pyrefly/issues/873), [#802](https://github.com/facebook/pyrefly/issues/802), [#716](https://github.com/facebook/pyrefly/issues/716)

### 2. Type Checking Enhancements

- Improved overload evaluation: Fully compliant with the Python typing spec
- Smarter Type Narrowing: Improved handling of variable types in loops, control flow, and pattern matching. Added ability to understand patterns like `getattr()`, `hasattr()`, and `dict.get()`
- `TypedDict` & Protocol Improvements: More accurate type checking for structural types, including handling for union keys and improved inheritance checks.
- Additional Python typing features: `LiteralString` & `TypeAliasType` support, abstract class instantiation checks, unused awaitable checks, multiple inheritance consistency checks, and more!
- Configurable Type Inference: Advanced features like empty container inference and return type inference can be toggled via config.
- Some relevant issues: [#684](https://github.com/facebook/pyrefly/issues/684), [#872](https://github.com/facebook/pyrefly/issues/872), [#1058](https://github.com/facebook/pyrefly/issues/1058), [#940](https://github.com/facebook/pyrefly/issues/940), [#815](https://github.com/facebook/pyrefly/issues/815), [#1461](https://github.com/facebook/pyrefly/issues/1461), [#44](https://github.com/facebook/pyrefly/issues/44)

### 3. Error Reporting & Suppression

- Granular Error Filtering: Filter summarized errors by code, set error severity, and get clickable links to docs.
- Suppression Workflow: Scripts and commands to silence new errors after upgrades, making large codebase migrations manageable.
- Improved Error Messages: Clearer, more actionable diagnostics for overloads, missing imports, and more.
- Some relevant issues: [#1276](https://github.com/facebook/pyrefly/issues/1276), [#1302](https://github.com/facebook/pyrefly/issues/1302), [#890](https://github.com/facebook/pyrefly/issues/890), [#1401](https://github.com/facebook/pyrefly/issues/1401)

### 4. Framework Integrations

- Django: Partial support, with ongoing improvements for model field inference, enum support, and lazy label handling.
- Pydantic: Experimental support, with growing documentation and bug fixes.
- Some relevant issues: [#316](https://github.com/facebook/pyrefly/issues/316), [#334](https://github.com/facebook/pyrefly/issues/334), [#1276](https://github.com/facebook/pyrefly/issues/1276), [#839](https://github.com/facebook/pyrefly/issues/839)

### 5. Jupyter Notebook Support (Experimental)

- Full Language Server Integration: Diagnostics, inlay hints, go-to-definition, hover, semantic tokens, signature help, and completions now work out-of-the-box for .ipynb files.
- CLI Type Checking for Notebooks
- Jupyter Lab Integration
- Some relevant issues: [#381](https://github.com/facebook/pyrefly/issues/381), [#591](https://github.com/facebook/pyrefly/issues/591), [#1045](https://github.com/facebook/pyrefly/issues/1045), [#1529](https://github.com/facebook/pyrefly/issues/1529), [#925](https://github.com/facebook/pyrefly/issues/925)

### 6. Build Systems, Dependencies & Configuration

- Initial Build System Support: Foundational work for Buck and other build systems to support generated file materialization and source file remapping
- Editable Installs: Documentation and support for working with editable dependencies.
- Improved dependency handling: performance improvements and automatic recognition of third party libraries with typeshed stubs
- Some relevant issues: [#573](https://github.com/facebook/pyrefly/issues/573), [#1](https://github.com/facebook/pyrefly/issues/1), [#2](https://github.com/facebook/pyrefly/issues/2), [#1239](https://github.com/facebook/pyrefly/issues/1239), [#360](https://github.com/facebook/pyrefly/issues/360)

Check out the [Pyrefly documentation](https://pyrefly.org/en/docs/) for more details on how to use each of these features.

## What does Beta status mean for you?

The Beta release is your invitation to start using Pyrefly in real-world projects, provide feedback, and continue helping shape its future. You do not need to change how you install and upgrade Pyrefly now that we are in Beta status, the same semantic versioning format will remain in place.

You can continue providing feedback as usual by [opening a GitHub issue](https://github.com/facebook/pyrefly/issues) or [joining our Discord server](https://discord.gg/Cf7mFQtW7W). In this new phase of development as we push towards our next milestone (a stable v1.0 release) we are particularly interested in hearing about how you are using Pyrefly in real production environments, including any edge cases you may encounter.

## Upgrade

```
pip install --upgrade pyrefly==0.42.0
```

### How to safely upgrade your codebase

1. `pyrefly check --suppress-errors`
2. run your code formatter of choice
3. `pyrefly check --remove-unused-ignores`
4. Repeat until you achieve a clean formatting run and a clean type check.

This will add `# pyrefly: ignore` comments to your code, enabling you to silence errors and return to fix them later.

Read more about error suppressions in the [Pyrefly documentation](https://pyrefly.org/en/docs/error-suppressions/)

## What's next?

Going forward we'll be continuing development work the same as usual with weekly releases of new features and bug fixes. However we will be closing out our [Beta milestone](https://github.com/facebook/pyrefly/milestone/4?closed=1) and moving on to the next, [a stable V1 Release](https://github.com/facebook/pyrefly/milestone/2). We'll be continuing to listen to your feedback and address open issues, so if there are specific things you want to see in a stable release, or blockers for adopting Pyrefly in your project, let us know!

---

# Pyrefly v0.43.0

**Status:** BETA
**Release date:** 24 November 2025

Pyrefly v0.43.0 bundles **213 commits** from **24 contributors**.

## New & Improved

### IDE/LSP Features
- New warnings for unused imports and unused variables
- Pyrefly LSP features now apply to Python documents with the `inmemory` schema, improving support for interactive consoles and temporary code cells in IDEs like Positron
- Added a configuration option to control whether "Go to definition" links appear in hover tooltips, enabled by default

### Type Checker
Improved support for unpacking `tuple[*Ts]`, fixing cases where unpacking would previously degrade to `object`. If a `TypeVarTuple`-annotated `varargs` is unmatched, it now resolves to an empty tuple in all cases.

### Error Messages
- New `--enabled-ignores` command line flag and configuration option allow specifying exactly which tools' ignore comments (e.g., `# pyright: ignore`) pyrefly respects
- CLI flags like `--error`, `--warn`, etc., now accept comma-separated lists, making it easier to enable or disable multiple error codes at once

### Pydantic Support
Improved handling of `RootModel` types, including nested and union scenarios. Basic support for Pydantic `BaseSettings`. Fields in `BaseSettings` are now treated as optional by default.

## Bug Fixes

We closed 17 bug issues this release.

- [#1642](https://github.com/facebook/pyrefly/issues/1642) - `TypedDicts` with any required keys are now correctly treated as always truthy, allowing for proper type narrowing in boolean expressions
- [#1611](https://github.com/facebook/pyrefly/issues/1611) - Correctly detects `await` in nested generator expressions, preventing them from being incorrectly typed as `Generator` instead of `AsyncGenerator`
- [#1609](https://github.com/facebook/pyrefly/issues/1609) - Fixed a bug where positional argument inlay hints for callables with `*args` would show incorrect positions or duplicates
- [#1598](https://github.com/facebook/pyrefly/issues/1598) - Literal strings now correctly participate in protocol subset checks (e.g., against `typing.Container`), fixing an issue with type signature mismatch
- [#1462](https://github.com/facebook/pyrefly/issues/1462) - Allow referencing inherited attributes from a class body's top level
- [#1256](https://github.com/facebook/pyrefly/issues/1256) - Fixes crash when the walrus operator (`:=`) appears in annotation positions
- [#273](https://github.com/facebook/pyrefly/issues/273) - Correctly handles `tuple[()] | tuple[int, *tuple[int, ...]]` to simplify to `tuple[int, ...]`
- And more: [#1644](https://github.com/facebook/pyrefly/issues/1644), [#1635](https://github.com/facebook/pyrefly/issues/1635), [#1633](https://github.com/facebook/pyrefly/issues/1633), [#1631](https://github.com/facebook/pyrefly/issues/1631), [#1625](https://github.com/facebook/pyrefly/issues/1625), [#1604](https://github.com/facebook/pyrefly/issues/1604), [#1268](https://github.com/facebook/pyrefly/issues/1268), [#1230](https://github.com/facebook/pyrefly/issues/1230), [#1016](https://github.com/facebook/pyrefly/issues/1016), [#981](https://github.com/facebook/pyrefly/issues/981)

## Upgrade

```
pip install --upgrade pyrefly~=0.43.0
```

### How to safely upgrade your codebase

1. `pyrefly check --suppress-errors`
2. run your code formatter of choice
3. `pyrefly check --remove-unused-ignores`
4. Repeat until you achieve a clean formatting run and a clean type check.

This will add `# pyrefly: ignore` comments to your code, enabling you to silence errors and return to fix them later.

Read more about error suppressions in the [Pyrefly documentation](https://pyrefly.org/en/docs/error-suppressions/)

## Contributors this release

[@javabster](https://github.com/javabster), [@AryanBagade](https://github.com/AryanBagade), [@asukaminato0721](https://github.com/asukaminato0721), [@austin3dickey](https://github.com/austin3dickey), [@yangdanny97](https://github.com/yangdanny97), Dhruv Mongia, [@fangyi-zhou](https://github.com/fangyi-zhou), [@Imran-S-heikh](https://github.com/Imran-S-heikh), [@grievejia](https://github.com/grievejia), [@jvansch1](https://github.com/jvansch1), [@keito](https://github.com/keito), [@kshitijgetsac](https://github.com/kshitijgetsac), [@kinto0](https://github.com/kinto0), [@arthaud](https://github.com/arthaud), [@nathan-liyiming](https://github.com/nathan-liyiming), [@rchen152](https://github.com/rchen152), Ron Mordechai, [@samwgoldman](https://github.com/samwgoldman), [@cybardev](https://github.com/cybardev), [@stroxler](https://github.com/stroxler), [@tianhan0](https://github.com/tianhan0), [@vinnymeller](https://github.com/vinnymeller), [@migeed-z](https://github.com/migeed-z)

---

# Pyrefly v0.44.0

**Status:** BETA
**Release date:** 03 December 2025

Pyrefly v0.44.0 bundles 163 commits from 23 contributors.

## New & Improved

### LSP / IDE
- Inlay hints are now clickable, allowing Go-To-Definition for class and type names directly from hint overlays
- Completions disabled inside comments and string literals to reduce noise

### Type checking
- Recursive type solving logic significantly refactored for more reliable inference
- Bundled typeshed and third-party stubs updated for improved library compatibility

### Pydantic/Dataclass
- Added support for Pydantic range constraints (e.g., `PositiveInt`)
- New infrastructure for Pydantic lax mode support
- Support added for `Decimal` type

### Error Handling
- Error suppressions now stricter, applying only when on start line

## Bug Fixes

11 bug issues closed this release:
- #1692: Fixed args/kwargs of wrapper functions wrongly reported as not subscriptable
- #1665: Prevented unnecessary `Literal[...]` tooltips in docstrings
- #1559: Updated dependencies fixing ruff parser crashes in Jupyter notebooks
- #1372: Resolved inherited properties incorrectly overridden by assignment
- Additional fixes: #1371, #1035, #1675, #1701, #1688, #1710, #1719

## Upgrade

```
pip install --upgrade pyrefly==v0.44.0
```

### Safe upgrade process

1. `pyrefly check --suppress-errors`
2. Run code formatter
3. `pyrefly check --remove-unused-ignores`
4. Repeat until clean

This adds `# pyrefly: ignore` comments for temporary error suppression during upgrades.

---

# Pyrefly v0.45.0

**Status:** BETA
**Release date:** 08 December 2025

Pyrefly v0.45.0 bundles **146 commits** from **21 contributors**.

## New & Improved

### Type Checking
- Dict literals without contextual hints now infer anonymous typed dicts, reducing false positives when unpacking heterogeneous dicts as kwargs
- Enhanced suggestion hints for name and attribute errors (e.g., `my_variuble` -> `my_variable`)
- Callable types narrowed more intuitively in `isinstance` checks; Callable Enums now supported
- New comparison check (defaults to warn) for predictable or inappropriate results like `True is False`, reflecting support for Pyright's `reportUnnecessaryComparison` and Mypy's `comparison-overlap`

### Language Server
- Automatic file renaming improved to allow editable third-party packages to be renamed

### Third Party Package Support
- Pydantic lax mode extended to support container types (lists, dicts, etc.)
- All Django fields now support nullability, fixing issues with `TextField` and others
- Correct signature inference for functions decorated with `numba.jit` and `numba.njit`

## Bug Fixes

19 bug issues closed this release:

- #1736: Fixed prepareRename LSP request for renaming symbols from editable packages
- #1732: Fixed handling of await in async comprehensions at module scope
- #1686: Fixed highlighting for Chinese/multi-byte characters using UTF-16 encoding
- #1632: Fixed tracking of unresolvable modules
- #1565: Fixed false positive type errors in nested loops
- #204: Fixed missing error validation for `__all__` implementations
- Additional fixes: #1773, #1765, #1742, #1739, #1720, #1698, #1624, #1479, #1475, #1325, #1289, #974, #548

## Upgrade

```
pip install --upgrade pyrefly==0.45.0
```

### How to safely upgrade your codebase

1. `pyrefly check --suppress-errors`
2. Run code formatter
3. `pyrefly check --remove-unused-ignores`
4. Repeat until clean

This adds `pyrefly: ignore` comments, enabling phased error fixes.

## Contributors

stroxler, migeed-z, rchen152, yangdanny97, samwgoldman, connernilsen, asukaminato0721, kinto0, grievejia, arthaud, AryanBagade, lolpack, ndmitchell, gvozdvmozgu, javabster, KaranPradhan266, jvansch1, tianhan0, Owen Valentine, Vladimir Matveev

---

# Pyrefly v0.46.0

**Status:** BETA
**Release date:** 15 December 2025

Pyrefly v0.46.0 bundles 125 commits from 19 contributors.

## New & Improved

### Language Server
- Call Hierarchy support implemented for the Pyrefly LSP server
- Function documentation now available in signature help
- Signature help uses single-line labels for editor compatibility
- Variables in `ALL_CAPS` correctly highlighted as constants in VSCode and website sandbox

### Type Checking
- Implemented PEP 765 warnings for `return`, `break`, `continue` statements exiting `finally` blocks

### Third Party Library Support
- Added stubs for scipy, matplotlib, scikit-learn, scikit-image, vispy, sympy, pandas, conan, botocore, and boto3
- Support for Pydantic "lax mode" with new inferred types (LaxInt, LaxBool)

### Errors / Debugging
- Debug symbols now included in release builds for usable stack traces

## Bug Fixes

14 bug issues closed, including:
- Fixed panic from capacity overflow with large tuple expansion
- Fixed `__future__` imports incorrectly flagged as unused
- Fixed text range calculation crashes exceeding buffer limits
- Fixed function hover popups missing names from stub files
- Fixed numba jit decorator special casing path handling

## Upgrade

```
pip install --upgrade pyrefly==v0.46.0
```

### How to safely upgrade your codebase

1. `pyrefly check --suppress-errors`
2. Run code formatter
3. `pyrefly check --remove-unused-ignores`
4. Repeat until clean

## Contributors

19 contributors including javabster, AryanBagade, asukaminato0721, connernilsen, DanielNoord, yangdanny97, and others.

---

# Pyrefly v0.47.0

**Status:** BETA
**Release date:** 05 January 2026

Pyrefly v0.47.0 bundles **207 commits** from **35 contributors**.

## New & Improved

### LSP Features
- Enhanced autocompletion for keys in `.get()` operations on `TypedDict` objects
- Improved documentation display in hover tooltips for attributes such as dataclass fields
- Enhanced import handling: `import X as X` and `from Y import X as X` now recognised as explicit re-exports, no longer flagged as unused

### Type Checking
- `TypeIs` calls now correctly narrow to type intersection rather than `Never`
- `Final` attributes lacking initialization on the class are now properly enforced
- Enhanced typing specification compliance by extending reassignment checks for `Final` variables across unpacking, context managers, and loops

### Pydantic Support
- Added support for Annotated fields
- Validation of `BaseModel` fields at instantiation, catching out-of-range arguments

### Error Types / Diagnostics
- New `non-exhaustive-match` error warns about incomplete `match` statements over Enums and literal unions
- New `missing-override-decorator` error emitted when methods override parent class methods without the `override` decorator (disabled by default)
- Suggestions provided for mistyped stdlib module imports

## Bug Fixes

**34 bug issues closed** this release

- #1952: `typing_extensions` now permitted for typing module special callables
- #1906: Single-underscore function `_` correctly exported from custom builtins for i18n support
- #1905: Fixed stack overflow panic in protocol conformance checks with recursive `Forall` types
- #1784: Fixed subtype checking for `LiteralString` and `Collection`
- #1723: Improved LSP server handling of `textDocument/didClose` for notebooks
- #1597: Type narrowing now uses intersection logic, preventing overly strict errors
- #1375: Improved support for `object.__new__(C)` type inference
- #1086: Autocomplete for string `Literal` values no longer inserts redundant quotes
- Additional fixes: #1048, #880, #844, #821, #792, #504, #1274, #1271, #1213, #1209, #1091, #1393, #1659, #1657, #1603, #1783, #1758, #1903, #1880, #1856, #1791, #1948, #1910, #1962, #1961

## Upgrade

```
pip install --upgrade pyrefly==0.47.0
```

### Safe Upgrade Process

1. Run `pyrefly check --suppress-errors`
2. Execute your code formatter
3. Run `pyrefly check --remove-unused-ignores`
4. Repeat until clean

This adds `# pyrefly: ignore` comments, enabling gradual error resolution during large codebase upgrades.

Read more in the [Pyrefly documentation](https://pyrefly.org/en/docs/error-suppressions/)

## Contributors

35 contributors including: martindemello, samwgoldman, stroxler, ArchieBinnie, dtolnay, arthaud, zanieb, grievejia, rchen152, aleivag, and 25 others

---

# Pyrefly v0.49.0

**Status:** BETA
**Release date:** 20 January 2026

Pyrefly v0.49.0 includes **231 commits** from **25 contributors**.

## New & Improved

### Type Inference
- Partial type inference now possible in loops
- Variance inference for self-referential generic types (PEP 695)
- `typing.Mapping` now recognised as a type alias

### Language Server
- Improved refactoring capabilities for moving module members and lifting local functions/methods to top-level module scope, with automatic import/shim creation
- Hover support for `in` keyword when used in iteration contexts (for-loops and comprehensions)
- "Go to symbol" now correctly includes methods inside a class
- Enabled clickable inlay hints for built-in types like tuple, dict, str, and special forms

### Pydantic Support
- Added detection and support for `pydantic.dataclasses.dataclass` decorator, including strict and lax mode support

### Error Reporting
- Improved error messages for `yield`, `yield from`, augmented assignment, missing imports, and missing stubs

## Bug Fixes

**38 bug issues closed** this release, including fixes for:
- Variance inference for stdlib generic classes
- Type narrowing for sequence patterns in match/case statements
- Class naming conflicts with overloaded methods
- Overload resolution for nested tuple element literals
- Plus 33 additional issues resolved

## Upgrade

```
pip install --upgrade pyrefly==0.49.0
```

### Safe Upgrade Process

1. Run `pyrefly check --suppress-errors`
2. Run your code formatter
3. Run `pyrefly check --remove-unused-ignores`
4. Repeat until clean

This adds `# pyrefly: ignore` comments to silence errors during upgrades.

## Contributors

25 contributors including: samwgoldman, stroxler, diliop, ndmitchell, arthaud, and 20 others.

---

# Pyrefly v0.50.0

**Status:** BETA
**Release date:** 26 January 2026

Pyrefly v0.50.0 contains 183 commits from 27 contributors.

## New & Improved

### Language Server
- Constructor calls now display instance types instead of `-> None` in hover and signature help
- Structured comment headers (e.g., `# Title ----`) now create hierarchical folding regions and outline symbols
- Go-to-definition supports intermediate submodule components (e.g., `a.b.c.D`)
- Inlay hints available for tuple unpacking variables
- Type variable bounds, constraints, and defaults display in generic type formatting
- `reveal_type` now distinguishes functions with different type variable restrictions

### Type Checking
- Type narrowing works for `TypedDict` using `in`/`not in` checks, including inherited keys

### File Structure/Detection
- Improved detection and renaming of editable installed packages

### Error Reporting
- The `suppress` command accepts JSON values for flexible error suppression

## Bug Fixes

30 bug issues were resolved, including:
- Stack overflow prevention in recursive type patterns
- Starred expressions in membership checks
- Subclass function assignments to abstract method attributes
- Imports referenced only in `__all__` no longer flagged as unused

## Upgrade

```
pip install --upgrade pyrefly==0.50.0
```

Follow the documented upgrade process using `--suppress-errors` and `--remove-unused-ignores` flags for large codebases.

---

# Pyrefly v0.51.0

**Status:** BETA
**Release date:** 02 February 2026

Pyrefly v0.51.0 bundles 162 commits from 25 contributors.

## New & Improved

### Type Checking
- Fixes for incorrect type narrowing in boolean operations with generic functions
- Intersection logic fixed to identify empty intersections (`Never`) for `final` classes and populated `Enums`, as they cannot be subclassed
- Added support for `GenericAlias` type, including special attributes like `__origin__`, `__args__`, and support for the pipe operator
- Added support for checking that class-scoped type variables are not used in `self` annotations in `__init__` methods, per typing spec

### Language Server
- During module resolution, "phantom paths" (non-existent paths checked during import) are now tracked, improving watch mode and import re-resolution
- List addition (`+`) now propagates type hints to both operands, making assignments like `l2: list[Base] = [A()] + [B()]` work as expected

### Error Handling
- Added configuration to debug and/or bypass deep recursion, which is useful for diagnosing stack overflow issues in large or generated codebases
- Multiple error messages have been shortened, clarified, or made more precise for better user experience (e.g., TypedDict, protocol variables, unpacking errors, pydantic, descriptor defaults)

### Performance Improvements
- Faster suggested standard library imports (did you mean...?)
- Reduced memory usage for type aliases of unions
- Further improvements to tracking for fine-grained dependencies, improving IDE performance during incremental rechecks and accuracy of features like auto-imports

## Bug Fixes

21 bug issues closed this release:

- #2269: Attribute lookups on classes inheriting from `Any` now fall back to `Any` instead of reporting `missing-attribute`
- #2250: `Self` is now properly bound in class body expressions
- #2236: Fixed issue where `--removed-unused-ignores` incorrectly removed ignores from other type checkers
- #2196: Tuple length checks with `isinstance` now work correctly with unpacked tuples and type variables
- #2118: Legacy `TypeVars` are now correctly inferred in `Callable` annotations without assignment
- Additional fixes: #990, #842, #496, #2036, #1917, #1807, #1714, #1680, #2169, #2141, #2211, #2208, #2246, #2267, #2256, #2274

## Upgrade

```
pip install --upgrade pyrefly==0.51.0
```

### How to safely upgrade your codebase

1. `pyrefly check --suppress-errors`
2. Run your code formatter of choice
3. `pyrefly check --remove-unused-ignores`
4. Repeat until clean

This adds `# pyrefly: ignore` comments to silence errors for later resolution.

Read more about error suppressions in the [Pyrefly documentation](https://pyrefly.org/en/docs/error-suppressions/)

## Contributors

rchen152, stroxler, migeed-z, kinto0, yangdanny97, arthaud, maggiemoss, connernilsen, grievejia, samwgoldman, rubmary, fangyi-zhou, AryanBagade, bluetech, ddrcoder, javabster, ndmitchell, praskr-wisdom, shayne-fletcher, tsembp, jvansch1, tianhan0, dtolnay

---

# Pyrefly v0.52.0

**Status:** BETA
**Release date:** 09 February 2026
**Commits:** 250 from 23 contributors

## Performance Improvements

### 18x Faster Updated Diagnostics After Saving a File
Type errors and diagnostics appear in your editor after saving a file with dramatic speed improvements. Edge cases that took several seconds now complete in under 200ms through fine-grained dependency tracking and streaming diagnostics.

### 2-3x Faster Initial Indexing Time
Initial indexing (when Pyrefly scans your project and builds its internal type map) has been optimised for speed, enabling faster access to type-aware features in large repositories.

### 40-60% Less Memory Usage
The language server now uses 40-60% less RAM during both initial indexing and incremental type checking, allowing Pyrefly to scale to larger projects.

### Edge case exponential memory blow up contained
A critical bug causing exponential memory usage with unions of dictionary types has been resolved.

## New & Improved

| Area | What's new |
|------|-----------|
| **Type System** | Added support for template strings (PEP 750); Type narrowing enhancements: `dict.get` narrowing and safer negative narrowing |
| **Language Server** | Automatic transformation of `from ... import *` statements into explicit imports; Improved auto-complete for match Literal patterns; Further support for type hierarchy; Go-to definition on `__all__` entries |

## Bug Fixes

13 bug issues were closed:

- [#713](https://github.com/facebook/pyrefly/issues/713) - Fixed unsound narrowing for `isinstance` and `type[T]`
- [#289](https://github.com/facebook/pyrefly/issues/289) - Expanded key-existence narrowing to dict-like types so `mapping.get("key")` returns the value type (not `None`)
- [#2248](https://github.com/facebook/pyrefly/issues/2248) - Fixed augmented assignment (`+=`) when `__iadd__` is missing to correctly fall back to `__add__`
- [And more!](https://github.com/facebook/pyrefly/issues?q=is%3Aissue%20state%3Aclosed%20closed%3A2026-02-02..2026-02-09%20type%3ABug)

## Upgrade

```
pip install --upgrade pyrefly==0.52.0
```

### How to safely upgrade your codebase

1. `pyrefly check --suppress-errors`
2. Run your code formatter of choice
3. `pyrefly check --remove-unused-ignores`
4. Repeat until clean

This adds `# pyrefly: ignore` comments, enabling you to silence errors and return to fix them later.

## Contributors this release

@ndmitchell, @rchen152, @stroxler, @kinto0, @asukaminato0721, @yangdanny97, @grievejia, @migeed-z, @jvansch1, @samwgoldman, @avikchaudhuri, @maggiemoss, @arthaud, @AryanBagade, @connernilsen, @diliop, @fangyi-zhou, @javabster, @robertoaloi, @tianhan0, @dtolnay, Carlos Fernandez, Miles Conn

---

# Pyrefly v0.53.0

**Status:** BETA
**Release date:** 17 February 2026
**Commits:** 245 from 27 contributors

## New & Improved

### Type Checking
- Recursive type aliases now supported with proper resolution and type-checking
- Error raised when `Self` is used in invalid locations (outside classes, static methods, metaclasses)
- Support for the idiom `class Foo(namedtuple("Bar", ...))`
- Warnings for protocol type variables with mismatched variance usage

### Language Server
- Completion suggestions ranked by most recently used (MRU) items
- Auto-import completions and unknown-name quick fixes honour common aliases (e.g., `numpy as np`)
- Improved error messages for signature mismatches with ASCII-style diffs

### Config
- JSON schemas added for `pyrefly.toml` and `pyproject.toml` enabling editor auto-completion and validation

### Performance
- Approximately 26% less CPU usage when type-checking PyTorch codebase on M1 Pro MacBook with 10 cores

## Bug Fixes

17 bug issues resolved, including:
- Lambdas with `yield`/`yield from` correctly inferred as generator-returning callables
- `Literal` imported via `try/except` recognised as special form
- Methods overriding base class methods no longer trigger false positive `@override` errors
- First parameter of class methods correctly handled regardless of naming

## Upgrade

```
pip install --upgrade pyrefly==0.53.0
```

---

# Pyrefly v0.54.0

**Status:** BETA
**Release date:** 23 February 2026
**Commits:** 93 from 18 contributors

## New & Improved

### Type Checker
- Support for `type(X)` in base class lists for metaclass expression compatibility
- Enhanced static evaluation for `sys.platform` and `sys.version_info` checks

### Language Server
- Autoimport and quickfix now surface explicit re-export paths
- Diagnostics can be controlled independently per workspace folder in multi-root setups
- "Generate code" quick fix actions now infer parameter types from call-site arguments
- Code completion and signature help show keyword/literal completions from all compatible overloads

## Bug Fixes

Seven bug issues closed this release, including:
- Adjusted overload-to-Callable subtyping for single-parameter overloads
- Improved TypeVar matching when union contains both bare and wrapped TypeVars
- Additional fixes: issues #2398, #949, #2421, #2457, #1122, #2434, #787

## Upgrade

```
pip install --upgrade pyrefly==0.54.0
```

### Safe Upgrade Steps

1. Run `pyrefly check --suppress-errors`
2. Format code
3. Run `pyrefly check --remove-unused-ignores`
4. Repeat until clean

See [Pyrefly documentation](https://pyrefly.org/en/docs/error-suppressions/) for details.

## Contributors

15 contributors listed, including stroxler, jvansch1, rchen152, and others.

---

# Pyrefly v0.55.0

**Status:** Released
**Release date:** 03 March 2026
**Commits:** 166 from 27 contributors

## New & Improved

### Type Checking
- Support for `Annotated` types, treating them as base types for typing specification conformance
- Inference of `Self` when constructing classes using `cls()`
- Evaluation of `os.name` similar to existing `sys.platform` support
- Correct treatment of `(*args: Any, **kwargs: Any)` as equivalent to gradual signature `...`
- Improved handling of star-unpacking arguments

### Language Server
- Hover results for generic overloads now include type parameters
- New "Type source" section reporting narrowing and first-use inference origins

### Documentation / Tooling
- `pyrefly report` now measures type completeness and annotation completeness for functions and parameters

### Performance Improvements
- Improved concurrent recheck performance, reducing LSP request latency

## Bug Fixes

35 bug issues closed this release, including:
- Fixed `unbound-name` errors with `NoReturn`
- Subclasses of `Any` no longer incorrectly flagged as non-callable
- Unpacked tuples as type arguments now work correctly
- Dynamically computed `__all__` handling improved
- False positives with imported functions fixed
- Override checking for empty tuples made more permissive
- Implicit type aliases with `|` operator handled correctly

## Upgrade

```
pip install --upgrade pyrefly==0.55.0
```

### Safe Upgrade Steps

1. Run `pyrefly check --suppress-errors`
2. Run code formatter
3. Run `pyrefly check --remove-unused-ignores`
4. Repeat until clean

This adds `# pyrefly: ignore` comments for gradual error fixing.

---

# Pyrefly v0.56.0

**Status:** Beta
**Release date:** 09 March 2026
**Commits:** 248 from 22 contributors

## New & Improved

### Type Checking
- Inferred return type width limited to prevent large unions
- Comparison checks with `Any` now yield `Any` instead of `bool`, aligning with gradual typing principles

### Language Server
- Experimental workspace mode added to `diagnosticMode` for publishing diagnostics across projects
- TSP `typeServer`/`getPythonSearchPaths` message support added
- Relative imports now functional for navigation and completions

### Type Errors
- Default severities tuned: `implicit-import` downgraded to warning; `unreachable` and `redundant-condition` default to warnings
- New error code: `non-convergent-recursion`

### Performance Improvements
- Dedicated thread pool added for LSP operations to prevent main thread blocking

## Bug Fixes

33 bug issues closed, including fixes for:
- Lambda expressions with default parameters in missing-argument checks
- False positives with untyped classmethods
- Builtins wildcard import shadowing
- `StrEnum` classmethod handling
- Generic function/callable subscripting
- Enum method callability checks
- Dict.setdefault behaviour on unpinned dicts
- Type alias-related overload resolution

## Upgrade

```
pip install --upgrade pyrefly==0.56.0
```

**Safe upgrade process:** Use `--suppress-errors` flag, format code, then `--remove-unused-ignores` to manage new type errors incrementally.

---

# Pyrefly v0.57.0

**Status:** Beta
**Release date:** 16 March 2026
**Commits:** 116 from 17 contributors

## New & Improved

### Type Checking
- Improved type narrowing for `hasattr` inside loops
- `pyrefly suppress` no longer corrupts multiline f-strings/t-strings by inserting suppression comments inside the string; it now places comments above the string and also matches suppressions correctly for errors inside multiline f/t-strings
- Improved `namedtuple` support with `*` field unpacking
- Fewer false-positive "variable is not initialised" errors

### Language Server
- If a nested pyproject.toml contains `[tool.ruff]` / `[tool.mypy]` / `[tool.pyright]`, it's treated as a strong "this is a Python project root" marker, preventing parent pyrefly.toml from incorrectly shadowing it (notably improving go-to-def accuracy on some repos)

### Performance
- Typechecking speed has improved, making it now ~20% faster to type check Pytorch on recent benchmarks

## Bug Fixes

We closed 24 bug issues this release:

- [#2696](https://github.com/facebook/pyrefly/issues/2696): Fixed an issue where Pyrefly's LSP incorrectly flagged `from typing import NewType` as unused, even when `NewType(...)` was referenced.
- [#2743](https://github.com/facebook/pyrefly/issues/2743): Fixed an issue where `TypedDict` fields named items/values prevented access to the corresponding `dict.items()` / `dict.values()` methods via attribute lookup.
- [#2745](https://github.com/facebook/pyrefly/issues/2745): Fixed an issue where chained/nested narrowing expressions (e.g. multi-clause and conditions) failed to narrow correctly when using negative subscript indices.
- [#2737](https://github.com/facebook/pyrefly/issues/2737): Fixed an issue where `functools.partial(...)` results couldn't be assigned back to a Callable typed with a `ParamSpec`, causing a false-positive type error.
- [#2650](https://github.com/facebook/pyrefly/issues/2650): Fixed an issue where a `Protocol` parameterised by `ParamSpec[...]` wasn't considered compatible with an equivalent "gradual" protocol using `*args: Any, **kwargs: Any`.
- [#2334](https://github.com/facebook/pyrefly/issues/2334): Fixed an issue where calling `__init__` on parametrised bound methods could trigger a false-positive type error due to incorrect attribute lookup behaviour.
- [#2731](https://github.com/facebook/pyrefly/issues/2731): Fixed an issue where `super()` calls to abstract methods that do have a concrete runtime body were incorrectly reported as missing-attribute / abstract-call errors.
- [#828](https://github.com/facebook/pyrefly/issues/828): Fixed an issue where reading a conditionally-initialised variable didn't "commit" the initialisation, leading to redundant follow-on "may be uninitialised" errors.
- [#835](https://github.com/facebook/pyrefly/issues/835): Fixed an issue where type information for subclasses wasn't handled correctly, leading to failures when type-checking subclass relationships.
- And more! [#2522](https://github.com/facebook/pyrefly/issues/2522), [#1800](https://github.com/facebook/pyrefly/issues/1800), [#2736](https://github.com/facebook/pyrefly/issues/2736), [#2382](https://github.com/facebook/pyrefly/issues/2382), [#913](https://github.com/facebook/pyrefly/issues/913), [#1397](https://github.com/facebook/pyrefly/issues/1397), [#2261](https://github.com/facebook/pyrefly/issues/2261), [#2669](https://github.com/facebook/pyrefly/issues/2669), [#2744](https://github.com/facebook/pyrefly/issues/2744), [#2739](https://github.com/facebook/pyrefly/issues/2739), [#1575](https://github.com/facebook/pyrefly/issues/1575), [#903](https://github.com/facebook/pyrefly/issues/903), [#1043](https://github.com/facebook/pyrefly/issues/1043), [#1429](https://github.com/facebook/pyrefly/issues/1429), [#2607](https://github.com/facebook/pyrefly/issues/2607)

## Upgrade

```
pip install --upgrade pyrefly==0.57.0
```

### How to safely upgrade your codebase

1. `pyrefly check --suppress-errors`
2. run your code formatter of choice
3. `pyrefly check --remove-unused-ignores`
4. Repeat until you achieve a clean formatting run and a clean type check.

This will add `# pyrefly: ignore` comments to your code, enabling you to silence errors and return to fix them later.

Read more about error suppressions in the [Pyrefly documentation](https://pyrefly.org/en/docs/error-suppressions/)

## Contributors this release

@stroxler, @grievejia, @yangdanny97, @migeed-z, @jvansch1, @rchen152, @asukaminato0721, @maggiemoss, @arthaud, @lolpack, @samwgoldman, @Adist319, David Tolnay, @avikchaudhuri, @rubmary, @javabster

---
