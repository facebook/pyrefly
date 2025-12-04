# Guidance for Project Agents

## Project Overview

Pyrefly is a fast language server and type checker for Python.

Architecture:

- Written in Rust using Buck (mostly for meta developers) and cargo (mostly for
  open-source developers)
- Minimal dependencies, framework-free

As described in the README, our architecture follows 3 phases:

- figuring out exports
- making bindings
- solving the bindings

Here's an overview of some important directories:

- pyrefly/lib/alt - Solving step
- pyrefly/lib/binding - Binding step
- pyrefly/lib/commands - CLI
- pyrefly/lib/config - Config file format & config options
- pyrefly/lib/error - How we collect and emit errors
- pyrefly/lib/export - Exports step
- pyrefly/lib/module - Import resolution/module finding logic
- pyrefly/lib/solver - Solving type variables and checking if a type is
  assignable to another type
- pyrefly/lib/state - Internal state for the language server
- pyrefly/lib/test - Integration tests for the typechecker
- pyrefly/lib/test/lsp - Integration tests for the language server
- pyrefly/lib/test/lsp/lsp_interaction - Heavyweight integration tests for the
  language server (only add tests here if it's impossible to add them in the
  lightweight tests)
- crates/pyrefly_types/src - Our internal representation for Python types
- conformance - Typing conformance tests pulled from python/typing. Don't edit
  these manually. Instead, run test.py and include any generated changes with
  your PR.
- test - Markdown end-to-end tests for our IDE features
- website - Source code for pyrefly.org
- lsp - vscode extension written in typescript

## Codebase style and guidelines

Coding style: All code must be clean, documented and minimal. That means:

- Keep It Simple Stupid (KISS) by reducing the "Concept Count". That means,
  strive for fewer functions or methods, fewer helpers. If a helper is only
  called by a single callsite, then prefer to inline it into the caller.
- At the same time, Don't Repeat Yourself (DRY)
- There is a tension between KISS and DRY. If you find yourself in a situation
  where you're forced to make a helper method just to avoid repeating yourself,
  the best solution is to look for a way to avoid even having to do the
  complicated work at all.
- If some code looks heavyweight, perhaps with lots of conditionals, then think
  harder for a more elegant way of achieving it.
- Code should have comments, and functions should have docstrings. The best
  comments are ones that introduce invariants, or prove that invariants are
  being upheld, or indicate which invariants the code relies upon.
- Check for existing helpers in the `pyrefly_types` crate before manually
  creating or destructuring a `Type`.
- Minimize the number of places `Expr` nodes are passed around and the number of
  times they are parsed. Generally, this means extracting semantic information
  as early as possible.

## Feature guidelines

- When working on a feature, the first commit should be a failing test if
  possible
- Tests should be run with buck when developing internally
  (`buck test pyrefly:pyrefly_library -- <name of test>` from within the project
  folder)
- If you make a change to a buck file, run `arc autocargo` to validate the
  changes
- `./test.py` will run the linters and tests, but it is very heavyweight so only
  run it when you are confident the feature is complete
