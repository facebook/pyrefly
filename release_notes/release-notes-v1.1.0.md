*Release date: June 17, 2026*

Pyrefly v1.1.0 bundles **250 commits** from **25 contributors**.

---

## ✨ New & Improved

### Type Checking

- Pyrefly now correctly narrows `TypedDict` types after `isinstance(x, dict)` checks, treating them as runtime `dict` instances while preserving field information in the positive branch.
- Type narrowing for bounded and constrained `TypeVar`s has been significantly improved. `isinstance` checks now correctly narrow `Self` and bounded type variables via their disjoint-base representatives, and negative narrowing on bounded TypeVars no longer produces false positives.
- Constrained `TypeVar`s are now preserved through method calls and binary operations, so `T & int` where `T: (int, str)` correctly returns `T & int` instead of losing the TypeVar.
- Classes decorated with `@dataclass(slots=True)` are now recognized as PEP 800 disjoint bases when they synthesize non-empty `__slots__`, enabling proper multiple-inheritance conflict detection and type narrowing.
- Pyrefly now detects when methods override parent class methods without using the `@override` decorator, and a new quick fix can automatically add the decorator for you.
- Variance inference now correctly skips receiver parameters (like `self` and `cls`) when determining class variance, fixing false invariance reports for container-like classes.

### Performance

- File discovery now uses the `ignore` crate with parallel traversal, making initial indexing 2-3x faster and dramatically reducing wall-clock time for large projects.
- Completed calculation results are now stored outside the mutex, reducing lock contention and improving performance on large codebases like `homeassistant-core` and `transformers`.
- Implicit builtins are now lazily materialized on first use instead of eagerly adding hundreds of import bindings to every module, reducing memory usage by 5-13% and CPU time by 10-28%.
- Literal unions are now capped at 256 non-enum literals or 4096 enum members per kind to prevent exponential type growth, and overall union size is capped at 4096 members.
- Bytes literals larger than 1024 bytes are now widened to `bytes` to keep type representations bounded.
- Wide tuple unions (more than 8 variants) are now collapsed into unbounded tuple types to prevent performance hangs and out-of-memory errors during type checking.

### Language Server

- Hover tooltips now work correctly on subscript expressions without spaces (e.g., `c[0]`), showing the `__getitem__` signature instead of garbled output.
- Literal completions are now offered in expected-type contexts like annotated assignments, return statements, and attribute targets, not just call arguments.
- Dictionary key completions now work in empty subscripts (`d[`) and half-typed subscripts (`d["`), suggesting known keys from `TypedDict`s and narrowed facets.
- Go-to-definition on `from X import Y` statements now falls back to the module file when the imported name can't be resolved (e.g., for non-Python modules like `.thrift` files).
- Quick fixes now use the file's actual line ending (CRLF or LF) instead of hardcoded `\n`, preventing mixed line endings.

### Type Server Protocol (TSP)

- The TSP backend now emits resolvable `typing` classes and `TypeVar`s instead of opaque builtins, so hovering over special forms like `Literal`, `Final`, and `ParamSpec` now shows the correct name instead of `Unknown`.
- Function types in TSP responses now carry specialized parameter and return types, enabling Pylance to display real types instead of `Unknown`.
- Pyrefly now emits the protocol-conformant lowercase `unknown` sentinel in TSP responses instead of uppercase `Unknown`.
- Expected-type queries now return the contextually expected type (e.g., the parameter type for a call argument) instead of the computed type, making the `typeServer/getExpectedType` request actually useful.

### Error Reporting

- No-matching-overload errors now show only the parameters relevant to the arguments you passed, hiding irrelevant optional parameters to make signatures easier to read.
- Arity mismatch details and call errors are now included in no-matching-overload error messages, making it clearer why an overload didn't match.
- The `None` hint in error messages now only suggests narrowing when it makes sense (e.g., in conditionals) and only suggests adding `| None` when you control the type.
- Variance errors on overloaded methods now point to the offending overload instead of always pointing to the first one.

### Configuration

- The `tensor-shapes` config option has been removed. Tensor shape support is now automatically derived from whether `shape_extensions` is resolvable for the module being checked.
- Pyrefly now understands `$MYPY_CONFIG_FILE_DIR` when migrating mypy configs, stripping it to produce portable relative paths.
- Glob patterns in config files now require `/` separators (not OS-specific separators) for cross-platform consistency.

---

## 🐛 Bug fixes

We closed **89** bug issues this release 👏

- **#3671:** Fixed a stack overflow crash when a descriptor's `__get__` method is itself a descriptor. Pyrefly now detects this self-referential case and terminates instead of recursing indefinitely.
- **#3712:** Fixed a crash when instantiating a nested functional `NamedTuple` in constructor defaults. Pyrefly now calls `ensure_expr` on the default value to avoid panicking during binding.
- **#3789:** Fixed a panic ("a variable has leaked from one module to another") when opening `sympy/core/expr.py` in the LSP. The lambda parameter cache is now keyed by `(ModuleName, ModulePath, LambdaParamId)` to distinguish in-memory and on-disk module versions.
- **#3749:** Fixed a false positive `incompatible-overload-residual` error on `functools.reduce(operator.*, ...)`. The overload resolves correctly; the spurious diagnostic has been removed.
- **#3819:** Fixed `unsupported-operation` errors when matching on overloads. Protocol conformance now filters overloads by `self:` type, so a `timedelta` receiver correctly selects the matching overload instead of erroneously picking the first one.
- **#3751:** Pyrefly now rejects `@dataclass` decorators on `NamedTuple` classes, matching CPython's runtime behavior which raises a `TypeError`.
- **#3773:** Optional imports like `try: import a except ImportError: a = None` are no longer reported as untyped by `pyrefly coverage`.
- **#357:** Runtime-evaluated annotations now report uninitialized forward names for Python < 3.14 unless `from __future__ import annotations` is active, matching CPython's behavior.
- **#3624:** `pyrefly coverage report` no longer includes definitions from `if __name__ == "__main__"` guards, as those aren't importable at runtime.
- **#3387:** `Foo[Bar]` under `from __future__ import annotations` now binds `Bar` as static type information when `Foo` is a known class, avoiding false "uninitialized" diagnostics.
- And more! #2798, #3188, #3385, #3208, #3237, #3378, #3413, #3455, #3392, #3299, #3446, #3448, #3396, #1492, #2451, #3431, #3210, #3381, #2547, #3147, #3524, #3506, #3519, #3343, #3505, #3369, #3344, #3400, #3394, #3221, #3514, #3520, #3544, #3458, #3593, #3548, #3576, #3292, #3526, #2869, #3228, #3578, #3568, #3410, #3602, #3300, #3356, #3607, #3612, #1496, #3623, #3059, #3598, #3536, #3066, #3418, #3235, #3673, #3541, #3169, #3075, #1838, #3550, #955, #2711, #498, #3113, #3197, #3215, #1592, #2002, #2082, #2199, #2553, #2693, #3220, #2751, #3743, #3790, #3708, #3703

Thank-you to all our contributors who found these bugs and reported them! Did you know this is one of the most helpful contributions you can make to an open-source project? If you find any bugs in Pyrefly we want to know about them! Please open a bug report issue [here](https://github.com/facebook/pyrefly/issues).

---

## 📦 Upgrade

```bash
pip install --upgrade pyrefly==1.1.0
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

@rchen152, @grievejia, @kinto0, @yangdanny97, @jorenham, @samwgoldman, @connernilsen, @stroxler, @shobhitmehro, @asukaminato0721, @javabster, @NathanTempest, David Tolnay, @mikeleppane, @maggiemoss, @arthaud, Eric Hou, Gregory Carlin, @thatfunkymunki, @hrolfurgylfa, @nitishagar, @tobyh-canva, @vivekjm, @QEDady, @rchiodo

---

*Please note: These release notes summarize major updates and features. For brevity, not all individual commits are listed.*
