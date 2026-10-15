*Release date: October 14, 2026*

Pyrefly v1.4.0 bundles **1191 commits** from **66 contributors**.

---

## Release Highlights

### Language Server

- **The VS Code extension now runs the Pyrefly from your environment.** By default, the extension launches the Pyrefly installed in your active Python environment (for example, your project's `.venv`), so your editor and your command line use the same version. If the environment has no Pyrefly, the extension falls back to its bundled binary. Set `pyrefly.pyreflyExecutable` to `bundled` to keep the old behavior. See [Pyrefly binary selection](https://pyrefly.org/en/docs/IDE/#pyrefly-binary-selection).

### Configuration & CLI

- **uv workspace members are now type checked.** In a uv workspace, the interpreter puts each member's source directory on `sys.path` as an editable install. Pyrefly treated those directories as installed packages and skipped them, so errors in workspace members went unreported. Pyrefly now checks editable sources inside your project, while regular `site-packages` stay excluded.

### Type Checking

- **Easier migration from BasedPyright.** Pyrefly reads `basedpyright` sections in `pyproject.toml` and applies the matching type checking mode preset.
- **New opt-in checks for untyped code.** New error kinds such as `unknown-member-type`, `unknown-return-type`, `untyped-function-decorator`, and `uninitialized-instance-variable` help you find where `Any` enters your code.

### Tensor Shapes

- **JAX shape stubs on PyPI.** This is the first stable release that ships JAX shape stubs as the `jax-pyrefly-stubs` package.

---

## ⚠️ Behavior Changes

These may change your editor setup or surface new errors in code that checked cleanly under v1.3.0.

- The VS Code extension now uses the Pyrefly in your active environment by default instead of its bundled binary. If your environment pins an older Pyrefly, the editor now runs that version. The **Pyrefly language server** output channel shows which binary was chosen and why.
- Pyrefly now checks the source of uv workspace members (editable installs inside your project). Errors in those packages that were hidden before can now appear.
- `--typeshed-path` now replaces the bundled typeshed entirely instead of overlaying it.
- The `incompatible-overload-residual` error kind is renamed to `incompatible-overload-argument`. The old name still works as a deprecated alias.
- `site-package-path` is renamed to `site-packages-path`. The old name still works.

---

## ✨ New & Improved

### Type Checking

- Pyrefly now recognizes `basedpyright` configuration sections in `pyproject.toml` and applies appropriate type checking mode presets, making it easier to migrate from BasedPyright.
- TypedDict handling has been significantly improved: generic `extra_items` are now instantiated correctly, extra items can supply missing arguments, and `definitely_seen` calculation in callable checking has been fixed.
- TypeVar restriction checking is now enforced even when the expected type is gradual, preventing silent violations of TypeVar bounds and constraints when `Any` appears in the expected type.
- Improved handling of overloaded function arguments in generic function calls through a new overload table mechanism, replacing the previous residual-based approach. The `incompatible-overload-residual` error kind has been renamed to `incompatible-overload-argument` (the old name remains as a deprecated alias).
- Variance inference has been improved with fixes for nested variance reliability bugs and explicit stored variance evidence.
- Property decorator validation now reports invalid signatures with extra required parameters.
- Added opt-in `uninitialized-instance-variable` check for annotated instance attributes never assigned in `__init__` or class body, including `@dataclass(init=False)` fields.
- Improved higher-order function call handling with quantified finalization and better overload branch capture.
- `functools.partial` now uses standard quantified finalization and handles generic overload targets.
- Improved ParamSpec variance in generic comparisons and handling of generic callable residuals.
- Added support for propagating divergence through compound statements to detect unreachable code after `if`/`else` blocks where all paths diverge.
- Added support for evaluating `sys.version_info` comparisons against tuples with release levels like `(3, 14, 0, "beta")`.

### Type Checking Fixes

- Overload evaluation now forces per-overload argument types before rolling back variables, so variables are no longer pinned by accident.
- Overload evaluation step 5 (argument materialization) now runs for implicit descriptor `__get__` calls.
- Invalid `type(...)` calls in annotations now suggest valid type forms like `types.NotImplementedType`, `types.EllipsisType`, `None`, `types.ModuleType`, or `type[MyClass]`.
- Invalid function calls in return annotations now produce clearer error messages.
- Unknown unions in class bases now accept valid known classes while recording dynamic bases.
- Conditionally imported base classes now keep the known class while recording a dynamic base.
- Tuple constructor type variables no longer leak when the constructor is passed as a higher-order factory callable.
- Overloaded constructors no longer leak type parameters through gradual arguments.
- Version-gated returns now preserve configured reachability.
- Environment-dependent branch tests (like `TYPE_CHECKING = False`) no longer report their suites as unreachable.
- Descriptor setter override checking now checks every overload.
- Module resolution now honors first-match-wins between single-file modules and packages in later roots.
- `Never` containers (`list[Never]`, `dict[Never, Never]`, etc.) are now recognized as falsy.
- Unpacking an overloaded tuple now combines branches position by position.
- Assigning to an overloaded type now succeeds when the value fits any branch.
- Facet narrowing now narrows overloaded values by ruling out branches.
- Attribute access and subset checks on overloaded types are more accurate.
- Overload selection now ignores type variables that appear only in the return type.
- Overloads that violate TypeVar restrictions under a gradual expected type are now rejected.
- Return type variables of overloaded protocol methods in invariant classes are now handled correctly.
- Union type arguments in indirect diamond inheritance are now compared without regard to member order.
- `max`/`min` calls with `key=` and `default=` no longer report argument errors from rejected union hint members.
- Attribute facets are now rebased correctly when the base type is replaced.
- `Annotated` metadata is no longer type checked, which removes spurious errors.
- Methods typed as a union with `Any`, or whose decorator returns `T | Any`, now bind correctly.
- A custom `__new__` inherited from an intermediate NamedTuple subclass is now handled correctly.
- Identity and membership tests with an `Any` operand now infer `bool`.
- Soft errors no longer drive constructor, default, and decorator checks.
- A `python-version` below 3.5 is now rejected with an error instead of crashing.

### Language Server

- The VS Code extension now launches the Pyrefly installed in your active Python environment by default, and falls back to the bundled binary. A new `pyrefly.pyreflyExecutable` setting (`from-environment` or `bundled`) controls this. Selecting a different interpreter restarts the server with the new binary when it changes.
- Pyrefly now watches explicit `configPath` settings and reloads configuration when those files change.
- Hover now prefers interface (`.pyi`) docstrings over implementation (`.py`) docstrings when both exist.
- Go-to-definition for standard library symbols now searches the interpreter's stdlib for `.py` files.
- Go-to-definition now prefers stub definitions over unresolved `__all__` entries and handles re-export cycles.
- Go-to-definition now works for TypedDict subscript keys.
- LSP rename now includes attribute assignments and deletions across files.
- Added LSP quick-fix code action for removing unused ignores.
- Added LSP registration for `workspace/didChangeConfiguration` on `python` section.
- Fixed language server to answer startup requests with workspace settings instead of defaults.
- Fixed language server deadlock when configuration response triggers file watcher registration.
- Added TSP `pyrefly/typeFacts` request for batch type queries.
- Fixed TSP `--config` to pin config for every file.
- Added code action videos to IDE features documentation.
- Fixed attribute definition index recording for assignment and deletion targets in rename.
- Added object attributes to LSP attribute completion, ranked after class attributes.
- `Callable` class attributes now bind `self` when first parameter is the containing class.

### Polars Support

- `select` and `with_columns` now handle both positional and keyword arguments correctly, preserving schema information.
- Added validation for Polars conditional predicates, duplicate `with_columns` outputs, and sole-argument list expansion rules.
- Improved Polars column-width tracking with `OutputCount` enum to distinguish provable width from unknown width.
- Added support for Polars `pl.col.` attribute completion, suggesting columns from the enclosing DataFrame operation.
- Added validation for column names in Polars `filter`, `sort`, `unique`, and `drop_nulls` methods.

### Django Support

- `QuerySet.annotate()` now preserves keyword annotation attributes on result rows for attribute lookup and LSP completion.
- Fixed Django relation `to=` keyword argument resolution.
- Fixed Django `ForeignKey` keyword-form `to=` resolution.

### Configuration & CLI

- uv workspace members (editable installs inside the project) are now checked instead of being excluded as site packages ([#2667](https://github.com/facebook/pyrefly/issues/2667)).
- Added `pyrefly completion` command to generate shell completion scripts for fish, bash, and zsh.
- Added `baseline-auto-update` project setting that automatically prunes baseline files when saving files in the IDE.
- Added configuration `extends` support for layered configurations.
- Renamed `site-package-path` to `site-packages-path` with backward compatibility.
- Added support for both adding and removing suppressions in one pass programmatically.
- Changed `--typeshed-path` to act as a full typeshed replacement instead of an overlay over the bundled typeshed.
- Added `embed::Checker::try_new_with_typeshed` to allow embedders to replace the bundled stdlib typeshed with in-memory `.pyi` stubs, useful for sandboxed interpreters like Monty.
- Added `--report-glean-ownership` flag to emit Glean ownership units for ACL-protected paths.
- Fixed `--report-glean` to write compact JSON instead of pretty-printed.

### Error Reporting

- Added `unknown-member-type` error kind for accessing attributes whose type is unknown.
- Fixed `unknown-variable-type` to report loop, unpacking, and `with` targets in addition to plain assignments.
- Added `untyped-function-decorator` check for decorators that return unknown types.
- Added `untyped-class-decorator` check for class decorators that return unknown types.
- Fixed `default-severity` for `potential-bad-keyword-argument` to be `ignore` as documented.
- Fixed dict display splat duplicate keywords to report `bad-keyword-argument` instead of `potential-bad-keyword-argument`.
- Added `unknown-return-type` for returning unknown values from unannotated functions and lambdas.
- Added `untyped-namedtuple` for `collections.namedtuple` without field types.
- Added `assert-type-unspellable-subtype` for unspellable types that are subtypes of expected type.
- Fixed `unannotated-return` false positive on `__init__` methods.
- Added content-level syntax highlighting for string escapes, format specifiers, and f-string expressions.
- Added hover info for language keywords with Python documentation links.
- Added Pylint compatibility documentation page.

### Performance

- Use `parking_lot` for per-module computing locks, reducing memory footprint in large projects.
- Store bindings in `Answers` step directly instead of nested shared ownership.
- Simplified AST step plumbing to avoid special-casing.
- Improved stubgen: reduced memory usage by using `Require::Exports` for implicit dependencies instead of `Require::Everything`.

### Tensor Shapes

- Updated NumPy, Torch, and JAX stubs to use direct shape splats in annotations instead of `Elements` wrappers where postponed annotations allow it.
- Added new "compatibility mode" syntax for Pyrefly shape types using `shape_extensions.Shaped` and `shape_extensions.shape_vars`, allowing libraries to use shape types without requiring downstream users to use Pyrefly.
- Fixed `no-any-return` warnings to avoid false positives when the return type union already contains `Any` or `Unknown`.
- Improved shape tracking for various Torch and NumPy operations including FFT frequency factories, distribution constraints, attention modules, weight initialization, and many others.
- Fixed shadowed Torch builtins in shape stubs by qualifying `int` and `bool` annotations.
- Added missing Torch module type symbols and top-level `torch.unique` stub.
- Fixed Torch module parameter iterators to return `Parameter` type.
- Added scalar `AdaptiveAvgPool2d` size support and one-bound `torch.randint` overload.
- Added `torch.is_tensor` narrowing support.
- Improved JAX nn coverage.
- Fixed NumPy array-like indices to accept pandas `Series` masks.
- Made `jax.typing.ArrayLike` shape-aware.
- Fixed `np.interp` type by adding missing `__float__`.
- Updated tensor shape documentation with current APIs and examples.
- Published JAX shape stubs to PyPI as `jax-pyrefly-stubs`. This is the first stable release that includes them.
- Added symbolic `abs` support in shape arithmetic.
- Fixed `ndarray` to be assignable to `SupportsAbs` by adding missing `__abs__`.
- Added integer literal arithmetic constant folding for `+`, `-`, `*`, `//`.
- Added opt-in `jaxtyping` configuration option to enable experimental jaxtyping shape checking support.
- Improved `jax.lax.scan` shape inference for tuples of arrays.
- `Tensor.T` shape tracking now reverses axes.
- Bare shaped array annotations now accept symbolic shapes.
- Gradual shape arguments are now preserved over defaults in generic inference.
- Rejected `nn.Module` registration types now fall back to the declared parameter type.

---

## 🐛 Closed Bug Reports

Many of the changes above came from user reports: we closed **87** GitHub bug issues this release 👏 Here are some highlights:

- **#4609:** Fixed an issue where concrete types with `TypeVarTuple` were not assignable to gradual types. Pyrefly was incorrectly wrapping the tuple in an extra layer, causing `tuple[int, *Ts]` to be rejected when assigned to `tuple[Any, *Ts]`.
- **#4565:** Fixed Polars DataFrame schema tracking for `select` and `with_columns`. Previously, `select` lost schema information for keyword expressions and `with_columns` lost it for positional expressions. Both now preserve schema correctly.
- **#4918:** Fixed an issue where `@property` decorators on methods with extra required parameters were not flagged as errors. Pyrefly now validates that property getters have the correct signature.
- **#4826:** Fixed overload step 5 (argument materialization) being skipped for implicit descriptor `__get__` calls. When a descriptor's argument type contained `Any`, the first overload would incorrectly win instead of properly evaluating all branches.
- **#4650:** Fixed excessive memory usage in `stubgen`. Generating a `.pyi` for a single file with a large transitive import closure previously used ~14 GB; it now uses comparable memory to `pyrefly check` by using `Require::Exports` instead of `Require::Everything`.
- **#4939:** Fixed LSP resolving third-party types against the wrong project's environment. In multi-root workspaces, a project's types were incorrectly resolved against a different project's venv, causing false errors.
- **#4825:** Fixed false `bad-return` error for generator functions with `Iterator[X] | T` return annotations. Pyrefly now correctly accepts `return` statements with non-None values in such generators.
- **#4849:** Fixed `TypeIs[T]` guards with free TypeVars producing no narrowing. When a `TypeIs` guard used a free TypeVar as its type argument, Pyrefly now correctly narrows to the TypeVar's bound.
- **#4932:** Fixed false errors in `max`/`min` key lambdas when `default=` is passed. Pyrefly was incorrectly typing the `key` callable's parameter from the expected type of the whole call rather than from the iterable's element type.
- **#5072:** Fixed errors when using conditionally imported Pydantic metaclasses. When a Pydantic metaclass was imported through a compatibility `try`/`except`, inference would degrade and report spurious errors.
- And more! #3525, #4901, #4525, #2667, #4813, #4989, #4995, #4991, #4844, #4592, #4694, #4986, #4846, #4858, #3828, #3970, #3722, #4606, #4725, #2260, #3704, #4848, #4879, #4885, #4640, #4919, #4741, #4368, #4129, #4605, #4688, #4730, #5021, #5037, #5080, #3407, #4494, #5058, #4618, #4400, #4888, #4478, #5073, #4935, #5070, #4881, #5071, #3380, #4264, #2911, #3323, #4876, #4703, #3977, #4977, #5204, #5169, #3689, #4987, #3361, #1211, #3466, #4792, #3822, #5188, #5210, #5016, #3465, #4118, #5082, #5084, #4332, #5233, #5236, #5213

Thank-you to all our contributors who found these bugs and reported them! Did you know this is one of the most helpful contributions you can make to an open-source project? If you find any bugs in Pyrefly we want to know about them! Please open a bug report issue [here](https://github.com/facebook/pyrefly/issues).

---

## 📦 Upgrade

```bash
pip install --upgrade pyrefly==1.4.0
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

@stroxler, @rchen152, @kinto0, @connernilsen, generatedunixname2066905484085733, @asukaminato0721, @jakevdp, @samwgoldman, @grievejia, @ndmitchell, David Tolnay, generatedunixname1699489071355949, @yeetypete, @QEDady, @javabster, @shahhamdihassan, @manmartgarc, @DetachHead, @yangdanny97, @alexander-beedie, generatedunixname89002005307016, @gaborv, @MarcoGorelli, @tonyyuyiding, @nitishagar, @vgvr0, @tobyh-canva, Catherine Gasnier, @stepanhruda, @pavelzw, @hrolfurgylfa, @NathanTempest, @oprypin, @arthaud, @akhil9tiet, @Aayushyaash, @savagecx, @n-gao, @dedsec-terminal, @Vishwaspatel2401, Willem Kokke, @Aniketsy, @Pager-dot, Alan Kurusingal, @digvijaysai29, @igorsugak, @shobhitmehro, @Trighap52, @maggiemoss, @dantrapp, @fineguy, @itamaro, @danielpassy, @aodihis, @kain88-de, Boyan Atanasov, @Parzivalart3mis, @cweider, @eriksimonson, @Divyansh151005, @iamhaseebn, @nikhil-CLTI, @WilliamK112, @nikmos3, @Wilfred, @kevin9327

---

*Please note: These release notes summarize major updates and features. For brevity, not all individual commits are listed.*
