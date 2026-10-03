# Contributing to Pyrefly

Welcome! We’re excited that you’re interested in contributing to Pyrefly. Whether you’re reporting an issue, fixing a bug, adding a feature, or improving documentation, your help makes Pyrefly better for everyone.

## Contributor Etiquette

* Please write concise and focused issues, pull request descriptions, and discussion comments.
* When submitting a pull request, follow [Making a Pull Request](#making-a-pull-request) and [Responding to Review Feedback](#responding-to-review-feedback).
* If you use AI assistance, review our [AI policy](AI_POLICY.md).

## Getting Started

To get started with contributing to Pyrefly:

1. [Find](#choosing-what-to-work-on) and [claim](#repository-automation) an issue that you would like to fix. If you have encountered a problem with Pyrefly, it's perfectly acceptable to open and then claim your own issue! We recommend choosing a bug fix rather than a feature request and not claiming more than one issue to start.
1. As you work on the issue, feel free to hop over to the `#dev` channel in our [Discord](https://discord.com/invite/Cf7mFQtW7W) if you have any questions.
1. If you find yourself needing to make multiple fixes or improvements, we strongly recommend [splitting your work](#splitting-a-pull-request) for faster reviews.
1. When you're ready, follow our [pull request checklist](#making-a-pull-request) to submit your code for review.
1. If a reviewer requests changes, [address their feedback and re-request review](#responding-to-review-feedback).

## Setting up your dev environment

The [rust toolchain](https://www.rust-lang.org/tools/install) is required for development. You can use the normal `cargo` commands (e.g. `cargo build`, `cargo test`).

## Choosing what to work on

We ask that contributors please look at our open GitHub issues for tasks to work on and discuss approaches with maintainers, rather than going straight to opening a PR.

When looking for an issue to pick up, consider the following things:

1. it has a [good first issue](https://github.com/facebook/pyrefly/issues?q=is%3Aissue%20state%3Aopen%20label%3A%22good%20first%20issue%22) or [help wanted](https://github.com/facebook/pyrefly/issues?q=is%3Aissue%20state%3Aopen%20label%3A%22help%20wanted%22) label
2. it's not already assigned to anyone, or is assigned to someone but appears abandoned
3. there aren't any open PRs for it (or there are open PRs but they look stale/abandoned)
4. the issue still reproduces in the sandbox, or locally on a build from the main branch
5. the issue is part of an upcoming milestone - these are the highest priority issues to focus on
6. the issue does not have the "needs discussion" tag - typically issues with that tag don't have a clear solution that everyone agrees on yet so they are not "shovel ready", but feel free to participate in the discussion!
7. when you find an issue you want to pick up, comment `#claim` on it to self-assign (see [Repository automation](#repository-automation) below).

## Repository automation

GitHub bots help manage issues and pull requests.

### Claiming issues: `#claim` / `#unclaim`

To pick up an issue, comment `#claim` on it and the bot will assign it to you. When you're done — or if you decide not to work on it after all — comment `#unclaim` to release it so someone else can take over.

How it works:
- `#claim` only works on **unassigned** issues. If the issue is already claimed by someone else, the bot leaves the existing assignee in place and tells you to coordinate with them — it won't reassign the issue to you. If it's already assigned to you, it just confirms that.
- `#unclaim` only removes *your own* assignment, and only if you're currently assigned.
- Both commands are case-insensitive and can appear anywhere in a comment (e.g. "I'd like to work on this, #claim").
- If the bot can't assign you automatically (GitHub only allows assigning users with repository access), it leaves a comment so a maintainer can assign you manually.
- If you'd like to work on an already-`#claim`ed issue, please post a comment on the issue mentioning a maintainer. You may message us in the `#dev` channel of our Discord server if we don't respond to your issue comment after a few days. We generally ask that you wait until two weeks after the issue is claimed by the current contributor, and that there's little activity indicating progress on the issue before request reassignment.

**Please note:** Claiming issues helps other contributors and maintainers see what is being worked on. If you do not claim an issue you're working on, multiple people may work on the same issue at the same time. This can lead to multiple PRs for the same task and increased review burden on maintainers. In cases where multiple PRs are opened for the same task, maintainers will prioritise reviewing the PR from the author who #claim-ed the issue. For issues marked with the `good-first-issue` tag, please only claim and work on one issue at a time to allow other newcomers to also work on issues.

## Developing Pyrefly

Development docs are WIP. Please reach out if you are working on an issue and
have questions or want a code pointer.

As described in the
[architecture overview](https://github.com/facebook/pyrefly/blob/main/ARCHITECTURE.md),
our architecture follows 3 phases:

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
[Buck2 coding convention](https://github.com/facebook/buck2/blob/main/docs/developers/basics.md),
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
`rust     if my_object.name == "target_case" {         dbg!(my_object);     }`

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

`dev` builds carry only line tables to keep `target/` small. When you need to step
through code in a debugger, build with the `dbg` profile (`cargo build --profile dbg`)
for full debug info.

## Making a Pull Request

Contributing a pull request (PR) is the main way to propose changes to Pyrefly. To ensure your PR is reviewed efficiently and has the best chance of being accepted, please make sure you have done the following:

- [ ] **IMPORTANT** [Claim the issue](#repository-automation) before starting work on it.
- [ ] Update or add new tests to cover your changes (see testing section for details).
- [ ] Limit your PR to a single purpose or issue. Avoid mixing unrelated changes, as this makes review harder. If the PR is large, consider [splitting it](#splitting-a-pull-request) for faster reviews.
- [ ] Write a clear PR description following the [template](.github/pull_request_template.md).
- [ ] Make sure all continuous integration (CI) checks pass. Fix any errors or warnings, or ask us about any CI results you don't understand. If the contributor license agreement (CLA) check fails, look for a comment from the meta-cla bot with instructions on how to sign the CLA.

We aim to respond to all PRs in a timely manner, but please note we prioritise reviews for work that is highest priority (e.g. critical bug fixes, upcoming milestones). If you're waiting on a review for more than a week, feel free to `@` one of the [maintainers](https://github.com/facebook/pyrefly/blob/main/.github/owners.json) in a comment on your PR or ask for a review in the `#dev` channel of our Discord server.

## Responding to Review Feedback

After you submit a pull request, it will be assigned to a maintainer for review. They will either accept and merge the PR, or leave review comments requesting changes. When you have made the requested changes, please do the following to request another review:
1. Acknowledge every review comment. This can be as simple as leaving a thumbs up or clicking "resolve conversation" for comments that you have resolved. Please reply to any comments that you have not fully resolved. **Do not leave any comment unacknowledged.**
1. Request another review by clicking the "re-request review" icon in the reviewers box in the top-right corner of the conversation tab. If this isn't available, you can tag the reviewer in a comment instead. (Example: "@rchen152 Ready for another review!")

## Splitting a Pull Request

Large code changes are more difficult to review, leading to longer review turnaround times and more back-and-forth. Extremely large PRs opened without prior discussion with a maintainer are unlikely to be reviewed at all.

If your change is over approximately 150 lines of non-test code, we strongly encourage doing one of the following for faster reviews:
- (Preferred when possible) Split your change into multiple, independently mergeable PRs. For example, for a bug with multiple root causes, submit one PR per root cause fix, or for a complex feature, submit a PR with a minimal core feature set and expand the feature in follow-up PRs once the core PR has been merged.
- Structure the PR as a series of small commits. For example, if you are adding a new configuration flag, you might add the flag boilerplate in one commit and the functionality in a second commit.

150 LOC is a rule of thumb, not a hard requirement; there is no need to split changes artificially just to stay under this number.

## Contributor License Agreement ("CLA")

In order to accept your pull request, we need you to submit a CLA. You only need
to do this once to work on any of Facebook's open source projects.

Complete your CLA here: <https://code.facebook.com/cla>. If you have any
questions, please drop us a line at <cla@fb.com>.

You are also expected to follow the [Code of Conduct](CODE_OF_CONDUCT.md), so
please read that if you are a new contributor.

## License

By contributing to Pyrefly, you agree that your contributions will be licensed
under the LICENSE file in the root directory of this source tree.
