# Pyrefly: A fast type checker and language server for Python with powerful IDE features

[![pyrefly](https://img.shields.io/endpoint?url=https://pyrefly.org/badge.json)](https://github.com/facebook/pyrefly)
[![PyPI](https://img.shields.io/pypi/dm/pyrefly?color=blue&label=pypi)](https://pypi.python.org/pypi/pyrefly)
[![Visual Studio Marketplace](https://img.shields.io/visual-studio-marketplace/d/meta.pyrefly?color=blue&label=VS%20Code)](https://marketplace.visualstudio.com/items?itemName=meta.pyrefly)
[![Open VSX](https://img.shields.io/open-vsx/dt/meta/pyrefly?color=blue&label=Open%20VSX)](https://open-vsx.org/extension/meta/pyrefly)
[![Discord](https://img.shields.io/badge/Discord-%235865F2.svg?logo=discord&logoColor=white)](https://discord.gg/Cf7mFQtW7W)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

Pyrefly is a type checker and language server for Python, which provides
lightning-fast type checking along with IDE features such as code navigation,
semantic highlighting, and code completion. It is available as a
[command-line tool](https://pyrefly.org/en/docs/installation/) and an extension
for popular IDEs and editors such as
[VSCode](https://marketplace.visualstudio.com/items?itemName=meta.pyrefly),
[Neovim](https://pyrefly.org/en/docs/IDE/#neovim),
[Zed](https://zed.dev/extensions/pyrefly), and
[more](https://pyrefly.org/en/docs/IDE/).

See the [Pyrefly website](https://pyrefly.org) for full documentation and how to
add Pyrefly to your editor of choice.

Currently under active development with known issues. Please open an issue if
you find bugs.

### Getting Started

- Try out pyrefly in your browser: [Sandbox](https://pyrefly.org/sandbox/)
- Get the command-line tool: `pip install pyrefly`
- Get the IDE Extension: [IDE installation page](https://pyrefly.org/en/docs/IDE/)

### Key Features:

- Type Inference: Pyrefly infers types in most locations, apart from function
  parameters. It can infer types of variables and return types.
- Flow Types: Pyrefly can understand your program's control flow to refine
  static types.
- Incrementality: Pyrefly aims for large-scale incrementality at the module
  level, with optimized checking and parallelism.

## Getting Involved

If you have questions or would like to report a bug, please
[create an issue](https://github.com/facebook/pyrefly/issues).

See our
[contributing guide](https://github.com/facebook/pyrefly/blob/main/CONTRIBUTING.md)
for information on how to contribute to Pyrefly.

Join our [Discord](https://discord.com/invite/Cf7mFQtW7W) to chat about Pyrefly
and types. This is also where we hold biweekly office hours.

## Choices

There are a number of choices when writing a Python type checker. We are taking
inspiration from [Pyre1](https://pyre-check.org/),
[Pyright](https://github.com/microsoft/pyright) and
[MyPy](https://mypy.readthedocs.io/en/stable/). Some notable choices:

- We infer types in most locations, apart from parameters to functions. We do
  infer types of variables and return types. As an example,
  `def foo(x): return True` would result in something equivalent to had you
  written `def foo(x: Any) -> bool: ...`.
- We attempt to infer the type of `[]` to however it is used first, then fix it
  after. For example `xs = []; xs.append(1); xs.append("")` will infer that
  `xs: List[int]` and then error on the final statement.
- We use flow types which refine static types, e.g. `x: int = 4` will both know
  that `x` has type `int`, but also that the immediately next usage of `x` will
  be aware the type is `Literal[4]`.
- We aim for large-scale incrementality (at the module level) and optimized
  checking with parallelism, aiming to use the advantages of Rust to keep the
  code a bit simpler.
- We expect large strongly connected components of modules, and do not attempt
  to take advantage of a DAG-shape in the source code.

## Code layout

Pyrefly is split into a number of crates (mostly under `crates/`):

- `pyrefly_util` are general purpose utilities, which have nothing to do with
  Python or type checking. Examples include IO wrappers, locking, command line
  helpers etc.
- `pyrefly_derive` are proc-macros for deriving traits such as `TypeEq` and
  `Visit`.
- `pyrefly_python` are Python utilities with no type-checking aspects, such as
  modelling modules or `sys.info`.
- `pyrefly_graph` provides utilities for indexing values and caching
  computations that may depend on each other.
- `pyrefly_bundled` are the third-party
  [typeshed stubs](https://github.com/python/typeshed).
- `pyrefly_config` defines the Pyrefly configuration, along with support for
  reading Mypy/Pyright configuration.
- `pyrefly_types` defines the Pyrefly type along with operations on it.
- `pyrefly_wasm` defines the sandbox code that compiles to WASM.
- `pyrefly` itself is the type checker and everything else.

## Design

There are many nuances of design that change on a regular basis. But the basic
substrate on which the checker is built involves three steps:

1. Figure out what each module exports. That requires solving all `import *`
   statements transitively.
2. For each module in isolation, convert it to bindings, dealing with all
   statements and scope information (both static and flow).
3. Solve those bindings, which may require the solutions of bindings in other
   modules.

If we encounter unknowable information (e.g. recursion) we use `Type::Var` to
insert placeholders which are filled in later.

For each module, we solve the steps sequentially and completely. In particular,
we do not try and solve a specific identifier first (like
[Roslyn](https://github.com/dotnet/roslyn) or
[TypeScript](https://www.typescriptlang.org/)), and do not use fine-grained
incrementality (like [Rust Analyzer](https://github.com/rust-lang/rust-analyzer)
using [Salsa](https://github.com/salsa-rs/salsa)). Instead, we aim for raw
performance and a simpler module-centric design - there's no need to solve a
single binding in isolation if solving all bindings in a module is fast enough.

### Example of bindings

Given the program:

```python
1: x: int = 4
2: print(x)
```

We might produce the bindings:

- `define int@0` = `from builtins import int`
- `define x@1` = `4: int@0`
- `use x@2` = `x@1`
- `anon @2` = `print(x@2)`
- `export x` = `x@2`

Of note:

- The keys are things like `define` (the definition of something), `use` (a
  usage of a thing) and `anon` (a statement we need to type check, but don't
  care about the result of).
- In many cases the value of a key refers to other keys.
- Some keys are imported from other modules, via `export` keys and `import`
  values.
- In order to disambiguate identifiers we use the textual position at which they
  occur (in the example we've used `@line`, but in reality it's the byte offset
  in the file).

### Example of `Var`

Given the program:

```python
1: x = 1
2: while test():
3:     x = x
4: print(x)
```

We end up with the bindings:

- `x@1` = `1`
- `x@3` = `phi(x@1, x@3)`
- `x@4` = `phi(x@1, x@3)`

The expression `phi` is the join point of the two values, e.g. `phi(int, str)`
would be `int | str`. We skip the distinction between `define` and `use`, since
it is not necessary for this example.

When solving `x@3` we encounter recursion. Operationally:

- We start solving `x@3`.
- That requires us to solve `x@1`.
- We solve `x@1` to be `Literal[1]`
- We start solving `x@3`. But we are currently solving `x@3`, so we invent a
  fresh `Var` (let's call it `?1`) and return that.
- We conclude that `x@3` must be `Literal[1] | ?1`.
- Since `?1` was introduced by `x@3` we record that `?1 = Literal[1] | ?1`. We
  can take the upper reachable bound of that and conclude that
  `?1 = Literal[1]`.
- We simplify `x@3` to just `Literal[1]`.
