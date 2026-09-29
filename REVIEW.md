# Code Review Instructions

Use this file only to guide code review. General project context lives in `AGENTS.md`; do not repeat it back in review comments unless it directly explains a finding.

## Review Scope

Focus on bugs in the changed code: incorrect logic, behavior regressions, missing edge cases, security/privacy risks, crashes, and changes that make Pyrefly produce unsound or misleading type-checker results. It is correct to return no findings when the diff is clean.

Do not report issues that are only formatting, naming preference, import ordering, syntax, unused code, type errors that Rust or TypeScript will catch, or lint findings that CI should enforce. Do not ask for tests just because a change lacks tests; only flag missing coverage when the absence creates a concrete risk that a real behavior regression will be missed.

## Severity

Use Important only for issues that are likely to cause wrong type-checker or language-server behavior, panics in reachable user flows, data loss, security/privacy exposure, or a production-quality regression that should be fixed before merging.

Use Nit for minor but actionable issues: low-risk maintainability problems, confusing control flow, missed simplifications that reduce real review burden, or project-convention violations that do not obviously break behavior.

Use Pre-existing only for bugs visible while reviewing the PR but not introduced by the PR. Do not post pre-existing findings unless they are severe enough that the author should know before building on the affected code.

## Pyrefly-Specific Checks

Always check whether a type checking change makes sense in Pyrefly's three-phase architecture: exports, bindings, then solving. One thing to look for in particular is whether added logic in the solving step actually depends on the types, or if it can be moved into the binding step.

For type-system changes, look for unsoundness and silent degradation. Prefer findings about concrete incorrect behavior: accepting code that should error, rejecting valid Python, losing type information, mishandling `Any` or unknowns, breaking the narrowing algebra, or producing diagnostics in the wrong location.

Unreachable states in Pyrefly should fail loudly. Flag silent fallbacks such as default match arms, `unwrap_or_default`, broad recovery paths, or ignored impossible cases when the surrounding invariants imply the state should be unreachable.

Some checks are optional or best-effort, in which case a broad fallback can be permissible.

Check for existing helpers in `pyrefly_types` before approving manual construction or destructuring of complex `Type` values. Flag duplicated type manipulation when it is likely to diverge from existing semantics.

For tests using the `testcase!` macro, remember that `bug = "..."` marks a passing test that documents undesirable behavior; it does not mean the test should fail. When a PR fixes the behavior, the review should expect the `bug` marker and expectations to be updated together.

Do not suggest editing generated conformance files manually. If conformance results need updating, the review may say they should be regenerated through the existing test workflow.

## Rust Review Bar

Prefer simple, direct Rust. Flag unnecessary abstractions, helpers with only one meaningful callsite, avoidable mutation-and-push loops, avoidable intermediate `collect()` calls, and defensive branches that obscure the main invariant.

Flag excessively long comments or duplicated comments that can go out of sync with the code easily.

Do not report Rust style that rustfmt or clippy will handle. Do report logic errors involving ownership, lifetimes, async blocking, error propagation, or panic behavior when they would affect runtime behavior or maintainability in a way CI will not reliably catch.

Non-test code should not use `unwrap()` in reachable paths. Either structure the code to avoid it, or use a panic that has a message explaining the invariant.

## Evidence And Confidence

Before posting a finding, verify it against the actual changed code and nearby context. Cite the root-cause line, not a downstream symptom. If the issue spans multiple lines, attach the comment to the smallest range that includes the incorrect statement or branch.

Do not claim certainty about runtime behavior, performance impact, or author intent unless the code demonstrates it. If a concern is plausible but not proven, either lower it to Nit or omit it.

Trust explicit comments and intentional-looking choices unless they create an Important issue. Do not fight the author's design preference unless it causes a concrete bug or meaningful maintainability risk.

Deduplicate findings. If the same bug appears in several locations, post one representative finding and mention the repeated pattern in that comment.

## Nit Volume

Keep Nit comments sparse. Prefer at most five Nits per review, and only include Nits that are specific, actionable, and worth interrupting the author for. If there are no Important findings, lead the summary with "No blocking issues."
