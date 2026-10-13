*Release date: August 27, 2026*

> **About dev releases**
> Dev releases (versions like `X.Y.Z-dev.N`) are non-stable snapshots cut periodically from trunk. They give early adopters a chance to try in-progress features and surface issues before the next stable release, but they don't carry the same stability or compatibility guarantees as a stable release — don't pin production projects to a dev version.

Pyrefly v1.3.0-dev.3 bundles **285 commits** from **33 contributors**.

---

## ✨ New & Improved

### Type Checking

- Bound methods now display without their receiver parameter, so a diagnostic no longer reads as `(self: A) -> None` is not assignable to `(self: A) -> None` when the two sides genuinely differ.
- Bound methods are parenthesized when displayed inside a union, so `(int) -> str | None` is no longer ambiguous.
- Assigning a value of type `Any` to a variable with a declared type no longer erases the annotation, so later checks on that variable keep working.
- Wide `Literal` type hints are no longer discarded when solving generic calls, fixing false errors on `Literal` unions with more than four members.
- An attribute confirmed present by `hasattr` can now be deleted with `del`, and the narrowing is cleared afterwards so later reads correctly error again.
- Illegal shadowing of a legacy `TypeVar` by a same-named type parameter in an enclosing scope is now detected and reported.
- `match` statements on tuple subjects now narrow to `Never` for exhaustiveness, so `assert_never` works on multi-subject matches.
- Exhaustiveness checking for all `match` statements is now available as an opt-in.
- Descriptor `__get__` is now applied when the attribute type is a union, rather than only for a single class.
- `assert callable(x)` now narrows to a callable returning `Any` instead of `object`, matching what the assertion is usually meant to express.
- Lambda parameter types are now inferred from the iterable passed to `map()`, so `map(lambda foo: ..., items)` is type checked.
- A value annotated as `TypeForm` can no longer be used as a type annotation, matching mypy, pyright, and ty.
- Unpacking `**kwargs` in a class header is now supported, with the unpacked fields validated against the class keywords.
- Calls that forward `*args` and `**kwargs` alongside known positional and keyword parameters are now checked against the keys the mapping actually provides.
- An unpacked `TypeVarTuple` in a callable parameter now accounts for optional arguments, so passing only the required ones checks.
- `enum.Flag` unions now match `typing.Self` in methods and classmethods, so combining members with `|` and returning `Self` checks.
- A forward reference to a nested class now resolves from annotations earlier in the enclosing class body.
- Reverse slices (`[::-1]`) of a fixed-length tuple now preserve the arity and the positional element types.
- Importing a re-exported `@deprecated` class or function now produces a deprecation warning at the import site, not just at the original definition.
- `__all__.remove(...)` is now taken into account when determining explicit exports, so removed names are no longer importable without an `implicit-reexport` error.
- Inferred return types that are callables are no longer truncated, preserving complex signatures.
- `.so` and `.dll` modules are now findable and treated as `Any` rather than failing to resolve.
- Equality narrowing on positive branches is now equality-aware, improving the types inferred after an `==` comparison.
- Literal regular expressions are now checked for unbalanced patterns and capturing groups, under a new `regex` error kind.
- Several stack overflows on recursive definitions are fixed: a recursive `Self` `__call__`, a recursive `TypedDict`, a recursive type alias used in a subscript, and protocol member lookup reached through `__getattr__`.

### Language Server

- A new `change_signature` refactoring lets you edit a function's parameters and have call sites updated for you.
- A new "remove unused import" quick fix cleans up unused imports, including inside nested blocks, and leaves a `pass` behind where an import cannot simply be deleted.
- A new quick fix inserts `assert x is not None` for an optional value, so you can narrow it without writing the assertion yourself.
- Renaming a legacy `TypeVar` or `ParamSpec` now also updates the string name argument in its constructor call.
- Hover no longer expands type aliases, so you see the alias you wrote rather than its full expansion.
- Hover now shows the package a symbol comes from, and aligns the type and default value at the `:` or `=`.
- Hovering over `not` now highlights the full unary expression instead of just the operator.
- Go-to-definition and hover on an operator no longer include the left-hand side of the expression.
- Autocomplete now works for discriminated unions, including dict literals in call arguments, returns, and nested TypedDict fields.
- Go-to-type-definition on a function call now targets the function's own definition rather than the definitions of each parameter and return type.
- Go-to-definition on a named argument in a dataclass or Pydantic `__init__` call now jumps to the specific field.
- Go-to-definition now resolves module paths written inside strings, such as `"accounts.urls"`.
- Call hierarchy now finds cross-file callers without requiring the calling file to have been opened first.
- Type hierarchy now reports subtypes declared in other modules, not just those in the file being queried.
- Auto-import completions now work on the first request after a reload, instead of only appearing once an import has been typed.
- Notebooks that are not `.ipynb` files on disk now get language server features too.
- A relative binary path in the `pyrefly.lspPath` VS Code setting is now resolved against the workspace root rather than the extension's working directory.

### Framework & Library Support

- Django REST Framework serializers no longer produce false `bad-override` errors for the inner `Meta` class pattern or for declared fields named like `Field` attributes.
- SQLAlchemy `update()` values are now checked against the mapped fields of the model, and SQL expressions are accepted in a `values()` call.
- SQLModel constructors now offer better hover text and completions.
- Polars support gained recursive and nested dtypes, plus owned dtype handling.
- Pydantic `Field()` defaults now infer correctly against wide `Literal` annotations.
- Attributes registered with `self.register_buffer(...)` and `self.register_parameter(...)` in a `torch.nn.Module` subclass are now recognized instead of reported as missing.

### CLI & Configuration

- A new `python-interpreter-find-command` setting lets you supply your own command for discovering the Python interpreter.
- `pyrefly.toml` can now enforce a required Pyrefly version at runtime, using PEP 440 constraints.
- Snippets can now be read from stdin.
- Virtual environment discovery no longer treats a project directory containing a `pyvenv.cfg` as the virtual environment itself; project discovery looks for conventional `.venv`, `venv`, and `env` children instead.
- A package installed in `site-packages` no longer shadows Pyrefly's bundled stubs, which previously produced conflicting definitions of the same type.
- `pyrefly init` no longer emits redundant `ignore-missing-imports` entries alongside a global `*` wildcard when migrating a mypy configuration.
- Incomplete `from` import statements are now ignored rather than panicking.

### Tensor Shapes

- We are building out a new DSL for describing shape transformations as type-level functions, and migrating the existing shape rules onto it. This work is ongoing.
- A new `pyrefly-numpy-stubs` package provides shape-aware NumPy stubs, alongside the existing `pyrefly-torch-stubs`.

---

## 🐛 Bug fixes

We closed **28** bug issues this release 👏

- **#4585:** Fixed a regression introduced in 1.3.0.dev1 where a module-qualified legacy `TypeVar` used as a type argument in a class base list produced a false `invalid-type-var` error saying the type variable was not in scope.
- **#4678:** Fixed pathological check times — over a thousand seconds on a small script — caused by repeatedly rescanning very long non-ASCII lines while collecting string ranges.
- **#4569:** Fixed an issue where `Literal` types with more than four members lost type inference, producing false errors on Pydantic `Field()` defaults and on any generic call given a wide `Literal` hint.
- **#4187:** Fixed an issue where narrowing of `TypeVar`-typed keyword arguments depended on the order the arguments were written in, so the same call could check or fail depending on argument order.
- **#4557:** Fixed a false positive `bad-override` when a subclass assigned a matching function to a callable `ClassVar` inherited from its parent.
- **#4657:** Fixed `enum.Flag` methods that combine members with `|` and return `typing.Self` being reported as returning the wrong type.
- **#4595:** Fixed false `bad-override` errors on Django REST Framework's `ModelSerializer.Meta` pattern, which is a configuration class rather than a true inheritance override.
- **#911:** Fixed `assert callable(x)` narrowing too aggressively: calling the narrowed value now yields `Any` rather than `object`.
- **#3636:** Fixed call hierarchy returning no incoming calls for cross-file call sites unless the calling file had already been opened in the editor.
- **#4410:** Fixed a forward reference to a nested class failing to resolve when used by an earlier annotation in the same enclosing class.
- And more! #4246, #4066, #4507, #4519, #4473, #3810, #438, #4576, #756, #4560, #4108, #3933, #4517, #4564, #3958, #3375, #1269, #1517

Thank-you to all our contributors who found these bugs and reported them! Did you know this is one of the most helpful contributions you can make to an open-source project? If you find any bugs in Pyrefly we want to know about them! Please open a bug report issue [here](https://github.com/facebook/pyrefly/issues).

---

## 📦 Upgrade

```bash
pip install --upgrade pyrefly==1.3.0-dev.3
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

@stroxler, @samwgoldman, @rchen152, @asukaminato0721, @grievejia, David Tolnay, @shobhitmehro, @connernilsen, generatedunixname1699489071355949, @ting-hong-shieh, @kavix, @WilliamK112, @lyydsheep, @Vishwaspatel2401, @ndmitchell, @markselby9, @danielgaskins, @kinto0, @NathanTempest, generatedunixname89002005307016, @fangyi-zhou, @javabster, @MarcoGorelli, @thomaspolasek, @tague, @paranoa233, @d34db3ff, @AMR5210, @DarkNightForge, @ternaus, @austin3dickey, @cakeni, @lolpack

---

*Please note: These release notes summarize major updates and features. For brevity, not all individual commits are listed.*
