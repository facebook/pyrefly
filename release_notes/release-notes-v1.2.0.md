*Release date: July 31, 2026*

Pyrefly v1.2.0 bundles **901 commits** from **59 contributors**.

---

## ✨ New & Improved

### Type Checking

- Attrs classes are now fully supported with comprehensive field synthesis, validation, and special-method generation. Pyrefly recognizes `@attr.s`, `@define`, `@frozen`, and their variants, handling field specifiers (`attr.ib()`, `field()`), converters, validators, defaults, and private-field aliasing. Converters are typed from their real input type rather than `Any`, including the `@<field>.converter` decorator, `attrs.converters.pipe`, `attrs.converters.default_if_none`, and generic converters like `copy.deepcopy`. See the new [attrs documentation](https://pyrefly.org/en/docs/attrs/) for details.
- `functools.partial` is now type-checked instead of treated as opaque. Bound arguments are validated at construction and Pyrefly synthesizes a precise residual signature for the remaining parameters, so errors surface at both the `partial(...)` call and the later invocation. Generic functions, overloads, constructors, bound methods, and `**Unpack[TypedDict]` parameters are supported, and results expose `.func`, `.args`, and `.keywords`. Precise residual callable assignment is enabled by the `strict` and `all` presets or explicitly with `strict-partial-subtyping`.
- Functions decorated with `functools.singledispatch` now type-check calls using the signature of the function you decorated, and registered implementations are checked against that function's first parameter. Generic `singledispatch` functions infer from call arguments instead of reporting `Unknown`.
- Pyrefly can now warn when a function declared to return a concrete type returns `Any`, with separate diagnostics for explicit and inferred cases. Off by default, and migrated automatically from mypy's `warn_return_any`.
- Pattern matching is substantially more precise: positional class patterns resolve attribute names from `__match_args__` at solve time, sequence element captures read from the narrowed subject so sibling constraints refine them, and `isinstance` on a facet filters the parent union to matching members. Fully covered class and sequence patterns now subtract their union member from later cases.
- Unpacking a variadic tuple with a fixed prefix and suffix no longer smears those elements into the starred capture, so `a, *rest, b = t` where `t: tuple[int, *tuple[bool, ...], str]` types `rest` as `list[bool]`.
- TypedDict classes now synthesize `__required_keys__` and `__optional_keys__`, so protocols requiring them (like those in langgraph) satisfy structural checks. `.get()` and `.pop()` with literal defaults preserve the field type.
- Overload handling improved throughout: constrained TypeVar arguments expand during resolution, rejected overloads no longer leak argument errors into diagnostics, and exact callback-forwarding signatures resolve overloaded callbacks against forwarded arguments for helpers like `asyncio.to_thread`.
- Enum `.value` on an enum type now infers as the union of member literal values instead of widening to the mixed-in data type.
- Lambda parameters are contextually typed more thoroughly. Types are stored directly rather than resolved through inference variables, and `*args`/`**kwargs` resolve from callable hints to `tuple[...]` and `dict[str, ...]` inside the body while preserving element types in the signature.
- All properties are now treated as data descriptors, and reflected binary-operation dunders are tried first for proper subclasses, fixing a class of incorrect attribute and operator results.
- Cyclic type aliases reachable from an annotation are now detected instead of hanging, and class finality is taken into account when deciding whether a condition is redundant.
- Continued basedpyright parity work: class instance truthiness is recognized, code under `if TYPE_CHECKING` is handled consistently, `typing_extensions.sentinel` is supported with relaxed naming restrictions, and `super(cls, cls)` is accepted.
- `copy.replace` is now type-checked like dataclass replacement, and `isinstance(x, type)` preserves type arguments when narrowing unions.
- Narrowing a receiver typed as `Self` to a subclass now preserves `Self`, eliminating false return errors in methods that return the narrowed receiver.

### Library Support

- Pydantic constructor synthesis now honors `populate_by_name` and built-in `alias_generator` functions, ignores `Field(init=False)` to match runtime, and treats `frozen` as a readonly field reason.
- Django `ForeignKey` targets now resolve attribute expressions and string references such as `"app_label.Model"`, preserving the generated relation and `<field>_id` types.
- Bare factory-boy factory calls now return the model type via `FactoryMetaClass`.
- PEP 561 partial stub packages are now supported in imports. Pyrefly reads the `partial` marker from `py.typed` and merges the stubs with the runtime package, deferring to the runtime package for omitted modules while preserving `.pyi`-before-`.py` precedence for provided ones.
- A configured `typeshed-path` now supplies stdlib stubs as well as third-party stubs, enabling complete custom-typeshed testing and overrides.

### Language Server

- Hover now resolves keyword arguments, renders callable protocols as their `__call__` signature, preserves overload docstrings at call sites, shows enum fields, wraps nested callable and `Concatenate[...]` signatures for readability, and covers `and`/`or` operators, augmented assignments, and union methods.
- Auto-import completions respect `python.analysis.autoImportCompletions`, rank deprecated stdlib typing aliases below their modern equivalents, avoid duplicates for already-imported modules, and preserve import aliases correctly.
- Rename now works on aliased imports and across files for keyword arguments, and renaming a Protocol class targets the class rather than `__init__`.
- Go-to-definition now navigates directly to symbols in non-Python files such as `.thrift`, including nested attribute and enum access and intermediate components of multi-dotted imports.
- Inlay hints debounce server-side (default 150ms), preventing width jitter while typing. `NewType` values now use their callable constructor signature instead of producing an invalid `type[N]` annotation.
- Notebook support improved: hover and type lookups work past the first cell, and inlay hints, document symbols, references, and diagnostic grouping work in cells following markdown cells.
- Document symbols fall back to flat `SymbolInformation` for clients such as Helix, semantic tokens cover `with ... as` and `except ... as` bindings, and cross-file diagnostics refresh on save in strict-spec clients such as Zed.
- `#region`/`#endregion` markers create folding regions, and selection ranges now follow AST nesting from expression to statement to scope to document.
- Baselined errors now appear as hints instead of errors, making it easier to distinguish new issues from known technical debt. Baselining also now applies correctly to unused-ignore diagnostics.
- Match captures receive consistent semantic highlighting and preserve their declaration identity for go-to-definition.
- `lspArguments` defaults to `["lsp"]` when empty, preventing startup failures in dev containers and remote environments.
- The VSCode extension adds an "Infer Types for Current File" command to the command palette.

### Error Reporting

- New opt-in diagnostics include `implicit-bool`, `invalid-cast`, `invalid-abstract-method`, `unused-call-result`, `empty-body`, `missing-super-call`, `unsupported-dynamic-base`, `implicit-reexport`, `unknown-argument-type`, `unknown-attribute-type`, `unknown-variable-type`, `implicit-any-lambda`, `untyped-function-decorator`, `untyped-class-decorator`, `unused-type-ignore`, and `no-any-return`.
- Dict literal element type errors now report on the exact line of the offending key or value, and missing-attribute errors gain a single-line header with the full type when more than one class lacks the attribute.
- Error summary output is sorted by count then name for stable ordering, and `--output-format=min-text` flattens multi-line headers so each error stays on one line.
- `--output-format=code-climate` produces GitLab-compatible Code Quality reports, while `full-text-with-github` combines readable diagnostics with GitHub workflow commands and is now used by the Pyrefly action.

### Coverage

- `pyrefly coverage` is no longer marked experimental.
- A `[coverage]` config table with `includes` and `excludes` can target a file set different from the project globs, with `--project-excludes` and positional `FILES...` taking precedence over it.
- `--public-only` keeps public re-exports from excluded files, so a symbol is only excluded when all of its public re-export locations are. Privately named classes and their members are excluded unless exported via `__all__`.
- `--fail-under` always prints findings whether or not the threshold is met.
- Coverage runs stream results as modules are solved, substantially reducing peak memory and improving runtime on large projects.

### Configuration & CLI

- `pyrefly infer --dry-run` previews inferred annotations without modifying files and exits unsuccessfully when changes would be made, making it suitable for CI.
- `python-platform` accepts one platform, multiple platforms, or all platforms. Platform guards are folded only when every configured platform agrees.
- File-level `# pyrefly: ignore-errors[code]` directives selectively suppress named error codes using the same parent-kind matching as line-level ignores.
- The `suppress` command's `--remove-unused=type` mode also removes unused `# type: ignore` comments.
- `--update-baseline` validates the baseline after config resolution, so a baseline configured in `pyproject.toml` works without repeating `--baseline`.
- Config migration surfaces a malformed `[tool.mypy]` section instead of silently falling back to pyright, and TOML parse errors point at the offending value rather than the start of the file.

### Stubgen

- Explicit `TypeAlias` right-hand sides and static `__all__` literals are now preserved, so re-exports survive in the stub.
- Implicit class variables are wrapped in `ClassVar[...]`, and multi-line parenthesized expressions keep their parentheses, preventing `IndentationError` in generated stubs.
- Async generators are emitted as plain `def` with `AsyncGenerator` return annotations, following the typeshed convention.
- Callable-typed values with differing return types across overloads render as `Callable[..., Incomplete]` instead of bare `Incomplete`, and statements guarded by conditions other than `TYPE_CHECKING` are no longer dropped.

### Bazel Integration

- A new, experimental `bazel-check` command adds Bazel-focused type-checking for projects that provide source information through JSON manifests. It is intended to collaborate with an in-progress `rules_pyrefly` Bazel rule implementation.

### Tensor Shape Types

- Many improvements. Still early alpha state.

### Performance

- Caching working-directory resolution eliminates repeated process-global syscalls, with reported multithreaded speedups of up to roughly 35% across the benchmark corpus. Directory-entry caching is now enabled unconditionally, and repeated `pkgutil` namespace probes are cached during import resolution.
- Bundled typeshed archives are now split into independently versioned stdlib and third-party archives, reducing optimized stdlib construction from 20.45ms to 4.97ms and deferring third-party lookup until site-package resolution.
- Bundled stub paths are stored once using immutable load-map indices, reducing retained heap and improving optimized construction by about 4% for stdlib and 6% for third-party bundles.
- Interpreter discovery now validates candidates with the environment query directly and reuses its cached result, reducing empty auto-configured checks from 49.8ms to 35.3ms.
- Smaller calculation cells reduce memory by roughly 2-4% on tested repositories, while lock-free reads of computed answers improve wall and CPU time by roughly 2%.
- Diagnostics are more stable across repeated runs, fixing flaky error-count changes observed in projects such as SymPy and apprise.

---

## ⚠️ Behavior Changes

These may surface new errors or warnings in code that checked cleanly under v1.1.0.

- `isinstance` narrowing now consumes dynamic uncertainty before intersecting with the target type, so `x: Any` narrowed by `isinstance(x, C)` refines to `C` instead of preserving the `Any`. This closes a type-safety hole while sacrificing the gradual guarantee in exchange for more precise runtime-evidence-based narrowing.
- Unannotated class attributes are now inferred by unioning the types of all assignments in the constructor, instead of taking the type of the first assignment.
- A file-level `# pyrefly: ignore-errors` directive placed after code now emits a `misplaced-ignore` warning, directing users to move it into the file preamble or use a line-level suppression.
- `direct-abstract-base-instantiation` is now separated from `bad-instantiation` and is a warning by default.
- `invalid-type-checking-constant` is a new default error for user-defined `TYPE_CHECKING` constants that are not typed as `bool`.
- `useless-overload-body` is a new default warning for executable logic in `@overload` declarations, whose bodies are never executed at runtime.
- `implicit-any-attribute` now focuses on `Any` synthesized at an attribute assignment; propagated `Any` is reported separately as the off-by-default `unknown-attribute-type`, so per-code configurations may need updating.
- More runtime-invalid constructs are now rejected, including incorrect string-literal unpacking and builtin slices, invalid dataclass and NamedTuple forms, assignments to bound-method attributes, duplicate `KW_ONLY` markers, and invalid `__init_subclass__` calls.
- Additional typing-spec checks now diagnose legacy `TypeVar`s not assigned to their declared name, raw-string quoted types, defaults on `self` or `cls`, and `super()` in directly defined NamedTuple methods.

---

## 🐛 Bug fixes

We closed **137** bug issues this release 👏

- **#3867:** Fixed a 1.1.0 regression where `isinstance()` narrowing broke after an unrelated `isinstance()` check in a returning sibling branch, restoring correct narrowing behavior.
- **#3893:** Fixed a stack overflow when checking self-referential protocols with overloaded methods.
- **#3841:** Unpacking a `TypeVar` bounded by a heterogeneous tuple now preserves each positional element type instead of widening every element to their union.
- **#3881:** Fixed inherited `kw_only=True` handling for SQLAlchemy-style declarative bases, preventing false "field without default may not follow field with default" errors in subclasses.
- **#1323:** `enumerate()` over tuple literals now preserves element types, so `for i, x in enumerate(("a", "b"))` types `x` as `Literal["a"] | Literal["b"]`.
- **#3437:** Fixed a stack overflow when checking recursive enum classes by removing the recursive Flag subtype check.
- **#3354:** Fixed a panic with fuzzed NamedTuple code by adding a distinct binding key for invalid assignment targets.
- **#4071:** Fixed an incorrect `bad-override` error when a child class generic over a `TypeVarTuple` overrides a parent method with the same `*args: *Ts` signature.
- **#3561:** Fixed a false positive `unannotated-return` for `@no_type_check` functions.
- **#3900:** Dataclass and Pydantic `BaseModel` fields named `"self"` are no longer incorrectly flagged on construction.
- And more! #3445, #3596, #3552, #2945, #3879, #3887, #3891, #2858, #3787, #3293, #3928, #3688, #3890, #3912, #3945, #3926, #3924, #3429, #3873, #3976, #3882, #897, #3585, #3584, #2689, #2346, #4024, #4025, #4019, #4020, #3345, #3321, #3989, #3998, #4021, #3730, #3732, #3997, #4062, #4083, #4018, #4023, #3949, #4022, #4093, #3974, #4073, #4094, #4035, #4109, #3619, #3240, #3305, #3638, #3546, #4107, #3329, #2495, #4034, #3435, #3701, #3154, #3085, #2817, #1510, #964, #1866, #3821, #3669, #3368, #1663, #907, #2257, #3595, #4155, #3742, #3888, #2132, #4170, #3731, #1621, #2851, #4247, #4228, #4163, #3365, #4255, #3238, #2942, #1583, #4060, #3805, #3883, #4177, #4193, #1819, #3422, #4165, #2474, #2932, #3745, #4072, #3668, #4305, #2517, #4192, #3658, #2574, #3911, #3897, #4319, #2287, #2965, #2503, #4307, #4325, #632, #3903, #4334, #4225, #3876, #4168, #4262, #3065, #2064, #4324, #4070

Thank-you to all our contributors who found these bugs and reported them! Did you know this is one of the most helpful contributions you can make to an open-source project? If you find any bugs in Pyrefly we want to know about them! Please open a bug report issue [here](https://github.com/facebook/pyrefly/issues).

---

## 📦 Upgrade

```bash
pip install --upgrade pyrefly==1.2.0
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

@stroxler, @shobhitmehro, @yangdanny97, @asukaminato0721, @rchen152, @kinto0, @samwgoldman, @grievejia, @jorenham, @NathanTempest, @anushamukka-dev, @ericzhonghou, @connernilsen, @tobyh-canva, @aodihis, @nitishagar, @kz357, @WilliamK112, @mikeleppane, @markselby9, @goutamadwant, @brosenfeld, @alexander-beedie, @ak4-sh, @xaskii, @qnox, @QEDady, @oprypin, @MannXo, @javabster, @fangyi-zhou, @durvesh1992, @arthaud, @vivekjm, @VasuAgrawal, @teerthsharma, @SarahMKosowsky, @roian6, @Punisheroot, @paranoa233, @olekuhlmann, @ndmitchell, @magic-akari, @maggiemoss, @lesbass, @lennardwalter, @knQzx, @Kartikey077, @JakobDegen, @ibraheemshaikh5, @hrolfurgylfa, @fhoehle, @dillydill123, @DibbayajyotiRoy, @coclique, @arogozhnikov, David Tolnay, Myles Matz, pratved64

---

*Please note: These release notes summarize major updates and features. For brevity, not all individual commits are listed.*
