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
- **Unreachable states must panic, not silently degrade.** Do not use defensive
  programming to handle states that should be impossible. If a match arm, Option,
  or Result should never occur given the surrounding invariants, use
  `unreachable!("explanation")` or `.expect("explanation")` — never
  `_ => default`, `.unwrap_or_default()`, or silent fallbacks. A type checker
  that silently produces wrong results is far worse than one that crashes with a
  clear message. Silent fallbacks hide bugs and confuse maintainers by making
  unreachable states look reachable.
- Check for existing helpers in the `pyrefly_types` crate before manually
  creating or destructuring a `Type`.
- Minimize the number of places `Expr` nodes are passed around and the number of
  times they are parsed. Generally, this means extracting semantic information
  as early as possible.
- **Imports:** Always add `use` imports at the top of the file rather than using
  inline qualified paths (e.g., write `use crate::foo::Bar;` and then `Bar`,
  not `crate::foo::Bar` inline). The only exception is when there is a name
  collision between two imports, which is rare.

## Development environments

There are three possible development environments:

1. **External/GitHub checkout**: Only `cargo` is available. The `buck` and `arc`
   commands do not exist.
2. **Internal on-demand**: Only `buck` is available. The `cargo` command may not
   be configured.
3. **Internal devserver with cargo**: Both `buck` and `cargo` are available.

**How to detect the environment:** Check for the presence of a `BUCK` file in
the project root. BUCK files are not exported to GitHub, so:
- If `BUCK` exists → internal checkout, `buck` and `arc` are available
- If `BUCK` does not exist → GitHub checkout, only `cargo` works

### Source control

**Do not assume git.** This repo may be either a Git checkout or a Sapling
(Mercurial-based) checkout. Before running any source-control commands, detect
which VCS is in use:

- If `.git` exists at the repo root → Git. Use `git` commands.
- If `.sl` exists at the repo root → Sapling. Use `sl` commands (`sl status`,
  `sl diff`, `sl commit`, `sl amend`, etc.). **Do not use `git`.**

You can check with: `test -d "$(sl root 2>/dev/null)/.sl" && echo sapling || echo git`

The internal (Meta) checkout always uses Sapling. The GitHub checkout uses Git.

## Feature guidelines

- When working on a feature, the first commit should be a failing test if
  possible

### Running tests

- **With buck (internal):** `buck test pyrefly:pyrefly_library -- <name of test>`
  (from within the project folder)
- **With cargo (external):** `cargo test <name of test>`

### Running the full test suite

- `./test.py` runs linters and tests. It is heavyweight, so only run it when
  you are confident the feature is complete.
- By default, `test.py` auto-detects the build tool based on BUCK file presence.
  You can override this with `--mode buck` or `--mode cargo`.
- For external builds, always use `python3 test.py` instead of `./test.py`.
- To run just formatting and linting (much faster than running tests):
  `./test.py --no-test --no-conformance`

### After modifying BUCK files (internal only)

- Run `arc autocargo` to regenerate Cargo.toml files and validate changes

### Before committing

**Always run formatting and linting before committing, updating a commit, or
handing code off to a human for review:**
`./test.py --no-test --no-conformance`

This applies whether you are committing autonomously or preparing code for a
human to commit. Do not skip this step during human-in-the-loop iteration.

- Running full tests before committing is ideal but optional since CI will run
  them. However, you must never skip formatting and linting.
- Lints may not always be fully clean due to pre-existing issues. The key
  requirement is: do not introduce *new* lint errors. If linting fails, check
  whether the errors are in code you modified. If so, fix them before
  committing.

## The `bug` marker in tests

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
  marker message very long.
