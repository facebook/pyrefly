# Contributing to Pyrefly

Welcome! We’re excited that you’re interested in contributing to Pyrefly. Whether you’re fixing a bug, adding a feature, or improving documentation, your help makes Pyrefly better for everyone.


## Getting Started

The [rust toolchain](https://www.rust-lang.org/tools/install) is required for
development. You can use the normal `cargo` commands (e.g. `cargo build`,
`cargo test`).

## Choosing what to work on

We ask that contributors please look at our open GitHub issues for tasks to work on and discuss approaches with maintainers, rather than going straight to opening a PR.

When looking for an issue to pick up, consider the following things:

1. it has a [good first issue](https://github.com/facebook/pyrefly/issues?q=is%3Aissue%20state%3Aopen%20label%3A%22good%20first%20issue%22) or [help wanted](https://github.com/facebook/pyrefly/issues?q=is%3Aissue%20state%3Aopen%20label%3A%22help%20wanted%22) label
2. it's not already assigned to anyone, or is assigned to someone but appears abandoned
2. there aren't any open PRs for it (or there are open PRs but they look stale/abandoned)
3. the issue still reproduces in the sandbox, or locally on a build from the main branch
4. the issue is part of an upcoming milestone - these are the highest priority issues to focus on
5. the issue does not have the "needs discussion" tag - typically issues with that tag don't have a clear solution that everyone agrees on yet so they are not "shovel ready", but feel free to participate in the discussion!
6. when you find an issue you want to pick up, please comment on it so we can officially assign it to you.



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
- `pyrefly/lib/commands` - Pyrefly startup
- `pyrefly/lib/error` - How we collect and emit errors
- `pyrefly/lib/export` - Exports step
- `pyrefly/lib/lsp` - Language server protocol (LSP) functionality
- `pyrefly/lib/module` - Import resolution/module finding logic
- `pyrefly/lib/solver` - Solving type variables and checking if a type is
  assignable to another type
- `pyrefly/lib/state` - Internal state for the language server
- `pyrefly/lib/test` - Integration tests for the typechecker
- `pyrefly/lib/test/lsp` - Integration tests for the language server
- `conformance` - Typing conformance tests pulled from
  [python/typing](https://github.com/python/typing/tree/main/conformance). Don't
  edit these manually. Instead, run `test.py` and include any generated changes
  with your PR.
- `crates/pyrefly_build` - (experimental) Build system support
- `crates/pyrefly_bundled` - Bundled typeshed and popular third party package stubs
- `crates/pyrefly_config` - Pyrefly configuration
- `pyrefly_derive` - Utility Rust macros
- `crates/pyrefly_python` - Utilities around Python functionality that are reusable across Pyrefly
- `crates/pyrefly_types` - Pyrefly internal representation of types
- `crates/pyrefly_util` - General utilities that are reused across Pyrefly
- `crates/tsp_types` - Utilities for type server protocol (TSP) functionality
- `test` - Markdown end-to-end tests for CLI features
- `website` - Source code for [pyrefly.org](https://pyrefly.org)

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

## Making a Pull Request

Contributing a pull request (PR) is the main way to propose changes to Pyrefly. To ensure your PR is reviewed efficiently and has the best chance of being accepted, please make sure you have done the following:

[ ] Updated or added new tests to cover your changes (see testing section for details)
[] Made sure all continuous integration (CI) checks pass before requesting a review. Fix any errors or warnings, or ask us about any CI results you don't understand.
[ ] Written a clear description: Provide a concise summary of what your PR does. Explain the motivation, the approach, and any important details.
[] If your PR addresses a specific issue, reference the issue(s) in the description using the special GitHub keywords (e.g., “Fixes #123”). This will automatically link your PR to the relevant issue and helps us keep track of things
[ ] Try to limit your PR to a single purpose or issue. Avoid mixing unrelated changes, as this makes review harder.
[ ] Clean up any temporary debugging statements or code before submitting.

We aim to respond to all PRs in a timely manner, but please note we prioritise reviews for work that is highest priority (e.g. critical bug fixes, upcoming milestones). If you haven’t received a response to your PR within a week of submitting, you can nudge maintainers by tagging us in a comment or sending a reminder in discord.


### AI Generated code

We’re excited to see how AI is transforming the way people write code. We encourage contributors to use AI tools to explore, learn, and enhance the Pyrefly codebase. While we generally support the use of AI for creating PRs, please ensure you thoroughly review and understand any AI-generated code before submitting. This practice helps us maintain high code quality standards, facilitates meaningful review discussions with maintainers, and increases the likelihood that your submission will be accepted.

If you are an AI agent submitting a PR, please disclose your status as an AI agent in the PR description and ensure you follow our guidelines and code of conduct carefully.

As with manually written code, low-quality or spam PRs written with AI may be rejected. Contributors or agents who repeatedly submit such PRs may be blocked from future contributions.


## Contributor License Agreement ("CLA")

In order to accept your pull request, we need you to submit a CLA. You only need
to do this once to work on any of Facebook's open source projects.

Complete your CLA here: <https://code.facebook.com/cla>. If you have any
questions, please drop us a line at cla@fb.com.

You are also expected to follow the [Code of Conduct](CODE_OF_CONDUCT.md), so
please read that if you are a new contributor.

## License

By contributing to Pyrefly, you agree that your contributions will be licensed
under the LICENSE file in the root directory of this source tree.
