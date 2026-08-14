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
- **Avoid unreachable state.** It is a code smell for a state that ought to be
  impossible due to surrounding invariants to look reachable.
  - Prefer to either encode the invariants in the Rust types so that the
    unreachable state is inexpressible, or refactor so that the code does not
    depend on implicit assumptions.
  - As a last resort, use `unreachable!("explanation")` or
    `.expect("explanation")` to make assumptions explicit.
  - Never hide the unreachable state through a silent fallback like
    `_ => default` or `.unwrap_or_default()`.
- Check for existing helpers in the `pyrefly_types` crate before manually
  creating or destructuring a `Type`.
- Minimize the number of places `Expr` nodes are passed around and the number of
  times they are parsed. Generally, this means extracting semantic information
  as early as possible.
- **Imports:** Always add `use` imports at the top of the file rather than using
  inline qualified paths (e.g., write `use crate::foo::Bar;` and then `Bar`,
  not `crate::foo::Bar` inline). The only exception is when there is a name
  collision between two imports, which is rare.
- **Line-level code quality matters:** Sloppy code introduces unnecessary reviewer
  overhead. Even if a piece of code is logically correct, it is not ready for
  review until it is also clean, elegant, and maintainable.

## Comments and Documentation

- Code should have comments and functions should have docstrings, but both should be
  concise. The best comments are ones that introduce invariants, or prove that invariants are being upheld, or indicate which invariants the code relies upon. Don't write duplicate comments, overly long comments, or comments for things that are obvious from
  reading the code.
- Prioritize readability over brevity. Reduce comments by omitting irrelevant
  information, not by compressing necessary information into fewer words. Use
  complete sentences, and do not drop words or use sentence fragments to save
  space or tokens.
- Use established, standard terminology. Do not coin new terms or shorthand for
  concepts, because doing so reduces comprehensibility.
- Write comments and documentation as statements of current truth. Never narrate
  corrections, prior framings, or what changed.
- When adding or modifying configuration options or command line flags, the corresponding
  docs should be updated.

## Commit Messages

The purpose of a commit message is to convey a commit's intent and rationale to the reader.
Use simple, plain language; keep it concise; and avoid jargon.

Do not write a laundry list of implementation changes. Focus on:

- **Why**: what problem or design gap motivated the change
- **What** (high level): the approach or solution, not individual file edits
- **Why it works**: how the code changes realize the solution

## Development environments

Pyrefly is developed both on GitHub and inside Meta's monorepo, and the
available tooling differs. **How to detect which one you are in:** check for a
`BUCK` file in the project root — BUCK files are not exported to GitHub.

- No `BUCK` → GitHub checkout. Only `cargo` is available, `buck` and `arc` do
  not exist, and source control is git. The rest of this file assumes this case.
- `BUCK` present → Meta-internal checkout. Read `facebook/AGENTS.md`, which
  covers the internal tooling and conventions (buck, arc, Sapling, Phabricator
  diffs) and overrides this file where they conflict.

## Feature guidelines

- When working on a feature, the first commit should be a failing test if
  possible

### Running tests

- `cargo test <name of test>`

### Running the full test suite

- `./test.py` runs linters and tests. It is heavyweight, so only run it when
  you are confident the feature is complete.
- For external builds, always use `python3 test.py` instead of `./test.py`.
- To run just formatting and linting (much faster than running tests):
  `./test.py --no-test --no-tensor-shapes --no-conformance --no-jsonschema`

### Before committing

**Always run formatting and linting before committing, updating a commit, or
handing code off to a human for review:**
`./test.py --no-test --no-tensor-shapes --no-conformance --no-jsonschema`

This applies whether you are committing autonomously or preparing code for a
human to commit. Do not skip this step during human-in-the-loop iteration.

- Running full tests before committing is ideal but optional since CI will run
  them. However, you must never skip formatting and linting.
- Lints may not always be fully clean due to pre-existing issues. The key
  requirement is: do not introduce *new* lint errors. If linting fails, check
  whether the errors are in code you modified. If so, fix them before
  committing.

## Writing tests

### The `bug` marker in tests

The `testcase!` macro supports a `bug = "<description>"` marker to indicate that
a test captures undesirable behavior. Important points:

- **Tests with `bug` must pass.** The marker documents that the *behavior* is
  wrong, not that the test itself should fail. Do not expect a `bug`-marked test
  to be a failing test.
- **Workflow for documenting known issues:** Add a passing test that shows the
  undesired behavior, using `bug = "..."` to explain what's wrong. This can be
  done to track issues or as part of a stack where a later diff fixes the bug.
- **Workflow for fixing bugs:** When the bug is fixed, remove the `bug` marker
  and update the test expectations to reflect the correct behavior.
- **Partial fixes:** If a test shows multiple undesired behaviors and a diff
  fixes only some of them, keep the `bug` marker but update the message if it
  has become stale.
- **Message length:** Keep the `bug` message concise. For complicated bugs, add
  detailed explanations as comments inside the test body rather than making the
  marker message very long. If there is an associated Github issue, linking to it
  in a comment is often sufficient without paraphrasing the issue in the test.

### `testcase!` header hygiene

The macro uses `line!()` to map errors in the embedded source back to the test file,
assuming a fixed layout. Extra lines in the header shift every reported line number.

- Put comments above `testcase!(`, never between it and the `r#"..."#` content.
- Keep `bug = "..."` on one line, with no blank lines in the header.
- `rustfmt` re-splits a `bug = ` line past 100 cols, so keep the message short
  enough to fit; put longer detail in a comment above the macro.

### Prefer `assert_type` over `reveal_type`

`assert_type` checks for type equivalence, whereas `reveal_type` expectations
do a more fragile text-based match. Prefer to use `assert_type` when possible.
It's acceptable to use `reveal_type` in cases in which the expected type cannot
be expressed in a type annotation - for example, a complex function signature.
