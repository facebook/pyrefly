# Contributing to Pyrefly

The [rust toolchain](https://www.rust-lang.org/tools/install) is required for
development. You can use the normal `cargo` commands (e.g. `cargo build`,
`cargo test`).

Refer to other sections of the document for
[coding conventions](#coding-conventions) and where to add
[new tests](#testing).

## Packaging

We use [maturin](https://github.com/PyO3/maturin) to build wheels and source
distributions. This also means that you can pip install `maturin` and, from the
inner `pyrefly` directory, use `maturin build` and `maturin develop` for local
development. `pip install .` in the inner `pyrefly` directory works as well. You
can also run `maturin` from the repo root by adding `-m pyrefly/Cargo.toml` to
the command line.

## Coding conventions

We follow the
[Buck2 coding conventions](https://github.com/facebook/buck2/blob/main/HACKING.md#coding-conventions),
with the caveat that we use our internal error framework for errors reported by
the type checker.

## Testing

You can use `cargo test` to run the tests, or `python3 test.py` from this
directory to use our all-in-one test script that auto-formats your code, runs
the tests, and updates the conformance test results. It requires Python 3.9+.

Here's where you can add new integration tests, based on the type of issue
you're working on:

- configurations: `test/`
- type checking: `pyrefly/lib/test/`
- language server: `pyrefly/lib/test/lsp/`

Take a look at the existing tests for examples of how to write tests. We use a
custom `testcase!` macro that is useful for testing type checker behaviour.

Please do not add tests in `conformance/third_party`. Those test cases are a
copy of the official Python typing conformance tests, and any changes you make
there will be overwritten the next time we pull in the latest version of the
tests.

Running `./test.py` will re-generate Pyrefly's conformance test outputs. Those
changes should be committed.

## Debugging tips

Below you’ll find a few practical suggestions to help you get started with
troubleshooting issues in the project. These are not exhaustive or mandatory
steps—feel free to experiment with other debugging methods, tools, or workflows
as needed!

### Make a Minimal Test Case

When you encounter a bug or unexpected behavior, start by isolating the issue
with a minimal, reproducible test case. Stripping away unrelated code helps
clarify the problem and speeds up debugging. You can use the
[Pyrefly sandbox](https://pyrefly.org/sandbox/) to quickly create a minimal
reproduction.

### Create a failing test

Once you have a minimal reproducible example of the bug, create a failing test
for it, so you can easily run it and verify that the bug still exists while you
work on tracking down the root cause and fix the issue. See the section above on
testing and place your reproducible examples in the appropriate test file or
create a new one.

### Print debugging

Printing intermediate values is a quick way to understand what’s going on in
your code. You can use the
[Rust-provided dbg! macro](https://doc.rust-lang.org/std/macro.dbg.html) for
quick value inspection: `dbg!(&my_object);`

Or insert conditionals to focus your debug output, e.g., print only when a name
matches:
`rust     if my_object.name == "target_case" {         dbg!(my_object);     }     `

When running your test you will need to use the `--nocapture` flag to ensure the
debug print statements show up in your console. To run your single test file in
debug mode run `cargo test my_test_name -- --nocapture`.

**Note: Remember to remove debug prints before submitting your pull request.**

### Use a Rust debugger

For tricky bugs sometimes it helps to use a debugger to step through the code
and inspect variables. Many code editors,
[such as VSCode, include graphical debuggers](https://www.youtube.com/watch?v=TlfGs7ExC0A)
for breakpoints and variable watch. You can also use the command line debuggers
like [lldb](https://docs.rs/lldb/latest/lldb/#installation):

## Contributor License Agreement ("CLA")

In order to accept your pull request, we need you to submit a CLA. You only need
to do this once to work on any of Facebook's open source projects.

Complete your CLA here: <https://code.facebook.com/cla>. If you have any
questions, please drop us a line at cla@fb.com.

You are also expected to follow the [Code of Conduct](CODE_OF_CONDUCT.md), so
please read that if you are a new contributor.

## Developing Pyrefly

Development docs are WIP. Please reach out if you are working on an issue and
have questions or want a code pointer.

As described in the README, our architecture follows 3 phases:

1. figuring out exports
2. making bindings
3. solving the bindings

Here's an overview of some important directories:

- `pyrefly/lib/alt` - Solving step
- `pyrefly/lib/binding` - Binding step
- `pyrefly/lib/commands` - CLI
- `pyrefly/lib/config` - Config file format & config options
- `pyrefly/lib/error` - How we collect and emit errors
- `pyrefly/lib/export` - Exports step
- `pyrefly/lib/module` - Import resolution/module finding logic
- `pyrefly/lib/solver` - Solving type variables and checking if a type is
  assignable to another type
- `pyrefly/lib/state` - Internal state for the language server
- `pyrefly/lib/test` - Integration tests for the typechecker
- `pyrefly/lib/test/lsp` - Integration tests for the language server
- `pyrefly/lib/types` - Our internal representation for Python types
- `conformance` - Typing conformance tests pulled from
  [python/typing](https://github.com/python/typing/tree/main/conformance). Don't
  edit these manually. Instead, run `test.py` and include any generated changes
  with your PR.
- `test` - Markdown end-to-end tests for our IDE features
- `website` - Source code for [pyrefly.org](https://pyrefly.org)

## License

By contributing to Pyrefly, you agree that your contributions will be licensed
under the LICENSE file in the root directory of this source tree.
