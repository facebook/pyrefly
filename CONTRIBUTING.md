# Contributing to Pyrefly

## Developer cheat sheet

### GitHub developers

The [rust toolchain](https://www.rust-lang.org/tools/install) is required for
development. You can use the normal `cargo` commands (e.g. `cargo build`,
`cargo test`).

You can also run `python3 test.py` from this directory to use our all-in-one
test script that auto-formats your code, runs the tests, and updates the
conformance test results. It requires Python 3.9+.

Refer to other sections of the document for
[coding conventions](#coding-conventions) and where to add
[new tests](#testing).

### Meta internal developers

From this directory, you can run:

- Check things are plausible: `./test.py` (runs the basic tests and linter)
- Run a command: `buck2 run pyrefly -- COMMAND_LINE_ARGUMENTS`
  - For example, run on a single file: `buck2 run pyrefly -- check test.py`
- Run a single test: `buck2 test pyrefly -- NAME_OF_THE_TEST`
- Run the end-to-end tests: `buck2 test test:`
- Run `arc pyre` (a.k.a. per-target type checking) with Pyrefly:
  `arc pyre check <targets_to_check> -c python.type_checker=fbcode//pyrefly:pyrefly_for_buck`
- Debug a file: `buck2 run pyrefly -- check <filename> --debug-info=debug.js`,
  then open `debug.html` in your browser
- Fetch Typeshed from upstream
  `HTTPS_PROXY=http://fwdproxy:8080 ./crates/pyrefly_bundled/update.py`

## Packaging

We use [maturin](https://github.com/PyO3/maturin) to build wheels and source
distributions. This also means that you can pip install `maturin` and, from the
inner `pyrefly` directory, use `maturin build` and `maturin develop` for local
development. `pip install .` in the inner `pyrefly` directory works as well. You
can also run `maturin` from the repo root by adding `-m pyrefly/Cargo.toml` to
the command line.

### Deploying to PyPI (Meta internal)

Once a week, a
[CodemodService job](https://www.internalfb.com/code/fbsource/xplat/scripts/codemod_service/configs/fbcode_pyrefly_version_upgrade.toml)
generates a diff to update the version number. Accept this diff to upload a new
version to PyPI.

If you'd like to do a manual release between the weekly automated releases,
follow the instructions in
[version.bzl](https://www.internalfb.com/code/fbsource/fbcode/pyrefly/version.bzl)
to update the version number.

Behind the scenes, what's happening is:

- The
  [publish_to_pypi workflow](https://github.com/facebook/pyrefly/blob/main/.github/workflows/publish_to_pypi.yml)
  triggers on any change to version.bzl.
- This workflow calls the
  [build_binaries workflow](https://github.com/facebook/pyrefly/blob/main/.github/workflows/build_binaries.yml)
  to build release artifacts, uploads them, and tags the corresponding commit
  with the version number.

## Coding conventions

We follow the
[Buck2 coding conventions](https://github.com/facebook/buck2/blob/main/HACKING.md#coding-conventions),
with the caveat that we use our internal error framework for errors reported by
the type checker.

## Testing

Here's where you can add new integration tests, based on the type of issue
you're working on:

- configurations: `test/`
- type checking: `pyrefly/lib/test/`
- language server: `pyrefly/lib/test/lsp/`

Please do not add tests in `conformance/third_party`. Those test cases are a
copy of the official Python typing conformance tests, and any changes you make
there will be overwritten the next time we pull in the latest version of the
tests.

Running `./test.py` will re-generate Pyrefly's conformance test outputs. Those
changes should be committed.

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
- `website` - Source code for [pyrefly.org](pyrefly.org)

## License

By contributing to Pyrefly, you agree that your contributions will be licensed
under the LICENSE file in the root directory of this source tree.
