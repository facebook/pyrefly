# Stubgen Architecture

Pyrefly's `stubgen` command generates PEP 484 `.pyi` stub files from Python
source. It combines two strategies: **AST-based structure extraction** (fast,
always runs) and **solver-powered type inference** (uses Pyrefly's full
type-checker to fill in types for unannotated items).

## Pipeline

```
                        ┌─────────────────────────────────────────────┐
                        │           commands/stubgen.rs               │
                        │                                             │
  pyrefly stubgen ──────▶  1. Resolve input files (FilesArgs)        │
                        │  2. Set up State / Transaction / Handles    │
                        │  3. For each .py file:                      │
                        │     a. Run the solver                       │
                        │     b. Collect inferred types               │
                        │     c. Apply annotations in-memory          │
                        │     d. Call generate_stub() on              │
                        │        the annotated source                 │
                        │     e. Write .pyi to output dir             │
                        └──────────────┬──────────────────────────────┘
                                       │
                          generate_stub() entry point
                                       │
                        ┌──────────────▼──────────────────────────────┐
                        │            stubgen/mod.rs                   │
                        │                                             │
                        │  1. Parse source with ruff (Ast::parse)     │
                        │  2. Build Definitions for __all__ filtering │
                        │  3. Build VisibilityFilter                  │
                        │  4. Collect overloaded function names       │
                        │  5. Walk AST, calling emit_stmt() for each  │
                        │     top-level statement                     │
                        │  6. Prepend `from typing import Any` if     │
                        │     any unannotated assignments remain      │
                        └──────────────┬──────────────────────────────┘
                                       │
                          emit_stmt() dispatches by node type
                                       │
              ┌────────────────────────▼────────────────────────────┐
              │                 stubgen/emit.rs                      │
              │                                                      │
              │  emit_stmt()         -- top-level dispatch           │
              │  ├─ Import/ImportFrom  → verbatim from source        │
              │  ├─ FunctionDef        → emit_function()             │
              │  ├─ ClassDef           → emit_class()                │
              │  ├─ AnnAssign          → name: annotation [= ...]    │
              │  ├─ Assign                                           │
              │  │  ├─ __all__         → verbatim                    │
              │  │  ├─ TypeVar(...)    → verbatim                    │
              │  │  ├─ List[int] etc.  → verbatim (type alias)       │
              │  │  └─ else            → name: Any = ...             │
              │  ├─ TypeAlias          → verbatim (PEP 695)          │
              │  └─ anything else      → skipped                     │
              │                                                      │
              │  emit_function()     -- def name(params) -> Ret: ... │
              │  emit_parameters()   -- handles /, *, *args, **kw    │
              │  emit_param()        -- name[: ann][ = ...]          │
              │  emit_class()        -- class Name(bases): body      │
              │                                                      │
              │  Helpers:                                             │
              │  source_at()              -- slice source by range    │
              │  is_type_var_call()       -- detect TypeVar(...)      │
              │  is_type_alias_value()    -- detect Subscript or |    │
              │  collect_overloaded_names() -- find @overload funcs   │
              │  is_overload_impl()       -- detect non-@overload    │
              │  stmt_uses_any()          -- track `Any` import need  │
              └──────────────────────────────────────────────────────┘

              ┌──────────────────────────────────────────────────────┐
              │             stubgen/visibility.rs                     │
              │                                                      │
              │  VisibilityFilter::Explicit(names)                   │
              │    └─ Only names in __all__ are included              │
              │  VisibilityFilter::Inferred                          │
              │    └─ Exclude _private names (unless --include-       │
              │       private), always include __dunder__ names       │
              └──────────────────────────────────────────────────────┘
```

## File Layout

```
pyrefly/lib/
├── commands/
│   └── stubgen.rs       CLI command: solver setup, file I/O, path mapping
├── stubgen/
│   ├── mod.rs           Public API: generate_stub(), StubgenOptions, tests
│   ├── emit.rs          AST-to-.pyi emission logic
│   └── visibility.rs    __all__ and public/private filtering
```

## Key Design Decisions

### Source-slicing for annotations

Rather than reconstructing type annotations from the AST, we slice the original
source text at each node's byte range (`source_at()`). This means every
annotation style the user writes -- `Union[int, str]`, `int | str`,
`'ForwardRef'`, `Callable[[int], bool]` -- is preserved verbatim in the stub.

### Solver-first approach

The CLI command (`commands/stubgen.rs`) always runs Pyrefly's full type-checker
before generating stubs:

1. **State / Transaction** -- The same machinery used by `pyrefly check` and
   `pyrefly infer`. Handles module resolution, dependency loading, and solving.
2. **Inferred types** -- `transaction.inferred_types()` returns return-type and
   container-variable annotations. `transaction.infer_parameter_annotations()`
   returns parameter types inferred from call sites.
3. **In-memory annotation** -- The inferred types are formatted to strings,
   sorted by reverse position, and spliced into the source text. The stub
   generator then sees a fully annotated module and emits high-quality stubs.

This makes our stubs strictly better than what AST-only tools produce: functions
like `def add(x, y)` that are called as `add(1, 2)` get `x: int, y: int`
parameter annotations in the stub.

### Overload handling

When a function has `@overload` decorators, we drop the non-overloaded
implementation (the one without `@overload`). This matches PEP 484 stub
conventions and mypy's behaviour. The detection uses
`collect_overloaded_names()` + `is_overload_impl()` and applies both at module
level and inside class bodies.

### TypeVar and type-alias bypass

Assignments like `T = TypeVar('T')` and `Coords = Tuple[float, float]` are
always emitted verbatim, regardless of `__all__` or private-name filtering.
These are type-level declarations required for the stub to be valid, not
runtime values a consumer would import.

Detection is by AST heuristics:
- `is_type_var_call()` -- RHS is a call to `TypeVar`, `ParamSpec`,
  `TypeVarTuple`, `NewType`, `NamedTuple`, or `TypedDict`.
- `is_type_alias_value()` -- RHS is a subscript (`List[int]`) or bitwise-or
  (`int | str`).

### Default-value handling

Stub files replace default values with `...` (the ellipsis literal).
`def foo(x: int = 42)` becomes `def foo(x: int = ...): ...`.
This is the PEP 484 convention: default values are implementation details that
stubs should not expose.

### `from typing import Any`

The `Any` import is only prepended when the string `Any` actually appears in
the output. This happens only for unannotated `Assign` statements that fall
through to the `name: Any = ...` path. Functions with missing annotations are
emitted bare (no `: Any`, no `-> Any`) since the solver usually fills them in.

## Data Flow (per file)

```
┌────────────────┐
│  Python source │  (original .py on disk)
└───────┬────────┘
        │
        ▼
┌────────────────┐
│  Solver (run)  │  transaction.run() → type-checks the module
└───────┬────────┘
        │
        ▼
┌────────────────────────────────────────────────────────────┐
│  Collect inferred types                                    │
│  • transaction.inferred_types()       → return types,      │
│                                         container vars     │
│  • transaction.infer_parameter_annotations() → param types │
└───────┬────────────────────────────────────────────────────┘
        │
        ▼
┌────────────────────────────────────────────────────────────┐
│  format_type_hints()                                       │
│  • Type → String (simplify literals, clean vars)           │
│  • Filter out Any, @-types, Unknown, Never                 │
│  • Produce (TextSize, ": int") / (TextSize, " -> str")     │
└───────┬────────────────────────────────────────────────────┘
        │
        ▼
┌────────────────────────────────────────────────────────────┐
│  apply_annotations()                                       │
│  • Sort by reverse position                                │
│  • Insert annotation strings into source (back-to-front)   │
│  Result: annotated Python source (in-memory only)          │
└───────┬────────────────────────────────────────────────────┘
        │
        ▼
┌────────────────────────────────────────────────────────────┐
│  generate_stub()                                           │
│  • Parse annotated source with ruff                        │
│  • Build Definitions → VisibilityFilter                    │
│  • Walk AST, emit stub syntax via emit_stmt()              │
│  Result: .pyi content string                               │
└───────┬────────────────────────────────────────────────────┘
        │
        ▼
┌────────────────┐
│  Write .pyi    │  to output directory
└────────────────┘
```

## Statement Handling Reference

| Python statement | Stub output | Notes |
|---|---|---|
| `import os` | `import os` | Verbatim |
| `from typing import List` | `from typing import List` | Verbatim |
| `def foo(x: int) -> str:` | `def foo(x: int) -> str: ...` | Body replaced with `...` |
| `async def fetch(url: str):` | `async def fetch(url: str): ...` | Async preserved |
| `@overload` + impl | Only `@overload` signatures | Impl dropped |
| `class Foo(Bar):` | `class Foo(Bar): ...` | Recursive emit for body |
| `x: int = 42` | `x: int = ...` | Value replaced with `...` |
| `x: int` | `x: int` | Declaration only |
| `x = 42` | `x: Any = ...` | Unannotated fallback |
| `T = TypeVar('T')` | `T = TypeVar('T')` | Verbatim, always included |
| `Alias = List[int]` | `Alias = List[int]` | Verbatim, always included |
| `type Vector = list[float]` | `type Vector = list[float]` | PEP 695, verbatim |
| `__all__ = ['foo']` | `__all__ = ['foo']` | Verbatim, always included |
| Anything else | *(skipped)* | `if`, `for`, `try`, etc. |
