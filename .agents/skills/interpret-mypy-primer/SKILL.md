---
name: interpret-mypy-primer
description: Interpret, analyze, and summarize a pyrefly mypy_primer error differential — the type errors a change ADDS or FIXES across the open-source corpus — classifying the changes as true positives (real bugs caught/fixed) vs false positives (regressions), and recording the result in the change's summary. Use when a pyrefly change alters which errors are reported (a new or retyped diagnostic, a new error kind, or an inference change).
---

# Interpreting a pyrefly mypy_primer differential

[mypy_primer](https://github.com/hauntsaninja/mypy_primer) runs a type checker over a
large corpus of real open-source projects and reports a **differential** between two
revisions: the errors a change newly **adds** (`+`) and the errors it **fixes** (`-`),
per project. Pyrefly's CI runs this on pull requests and posts the differential as a
comment.

The counts alone do not tell you whether a change found real bugs or broke valid code.
This skill turns a raw differential into a verdict and a short write-up you put in the
change's summary. Use it whenever a change alters which errors pyrefly reports; a pure
refactor with no diagnostic change does not need it.

Core principle: **do not assert, verify.** Sample the added/removed errors, reproduce
them minimally, and let reference checkers and the Python runtime decide true-positive
vs false-positive — do not eyeball the message and guess.

## 1. Get the differential

- **From CI:** read the mypy_primer comment on the pull request.
- **Locally:** run `mypy_primer` comparing your branch against its base, or build pyrefly
  at both revisions with `cargo build --release` and run each on the **same fixed
  checkout** of a few corpus projects (`pyrefly check --preset <p> <project>`), then diff
  the error lists. Measuring on fixed code is essential — see noise, below.

## 2. Interpret by preset

Report the differential as `+added / -removed`, broken down by the preset(s) you ran.

- The **shipping presets** (`default`, `strict`) matter most: a change should ideally be
  `+0/-0` there, or add/remove only true positives. New errors in `default` are seen by
  every user, so they carry the highest bar.
- A large volume under the `all` preset is expected and acceptable for a new **opt-in**
  error kind (one that defaults to `ignore` and only reports under `all`). Say so rather
  than treating the number as alarming.
- Confirm the additions are the **kind you intended**. If a change meant to add
  `foo-error` also moves counts for unrelated kinds, investigate.

## 3. Discount noise

- A `+N / -N` pair that is **byte-identical at the same location** is churn from the
  corpus project's own HEAD moving between runs, not your change. Discount it and say so.
- **Totals are not comparable across days:** each run uses every project's current HEAD,
  so upstream churn moves counts by hundreds. Only a before/after measurement on the
  *same* code is attributable to your change.

## 4. Verify correctness (true positive vs false positive)

Sample the differential — at least the largest clusters, and a spread of kinds. For each
sampled entry:

1. Build a **minimal reproducer** that triggers the same error, without copying the
   change's intended behavior into it.
2. Run it through **pyrefly at your change**, a **reference checker** (mypy and/or
   pyright — install via `uv run --with mypy python -m mypy <file>` /
   `uv run --with pyright pyright <file>`), and, where the behavior is observable, the
   **CPython runtime**.

Classify:

- **Added error → true positive** if a reference checker reports the same error at the
  same place, or CPython fails at that exact operation for the same reason. The change
  correctly caught something; leave it.
- **Added error, reference checkers clean, runtime fine → `unclear`, not automatically a
  false positive.** mypy and pyright are not the specification, and pyrefly sometimes
  intentionally differs; a program that merely runs is not proof it is type-correct. Flag
  it for maintainer/spec review.
- **Attribute a confirmed regression by before/after build**, not by editing the program:
  run the same reproducer through pyrefly built at the base revision and at your change.
  An error only on your build is caused by the change (a false positive to fix); an error
  on both pre-exists.

Patterns worth knowing: a single root cause often explains a whole cluster of per-project
errors — fix the cause, not each symptom, and re-run. Framework classes
(`dataclass_transform`, metaclasses, `Annotated`) are a common false-positive source and
need the real framework in the reproducer, not a plain stand-in.

## 5. Record the result in the change's summary

Put a short paragraph in the **pull request description** (not buried in a comment),
stating:

- the per-preset deltas (`+X / -Y`), and that the additions are `<kind>`;
- the shipping-preset (`default`/`strict`) impact specifically;
- any noise you discounted and why;
- a one-line correctness verdict from your sampling — e.g. "sampled the 40 additions,
  all true positives (mypy agrees)", or "~95% concordant with pyright; the rest are
  known precision gaps".

The goal is that a reviewer can see the real-world impact of the change without re-running
anything.

## Inside Meta (internal checkout)

The interpretation, noise handling, and verification above are identical; only the
plumbing differs.

- **Get the differential** from CI with the `ci-signals` skill: the `pyrefly-mypy-primer`
  signal (plus the `instagram-server` and `pytorch` signals) carries it. Build before/after
  with `buck2 build @fbcode//mode/opt fbcode//pyrefly:pyrefly` (copy the binary out
  immediately — the output path is reused across builds) rather than cargo.
- **For a large or land-blocking differential, use the `triage-pyrefly-ci` skill**, which
  fans out one agent per entry to produce per-error verdicts with evidence.
- **Record the paragraph in the Phabricator diff Summary** (`meta phabricator.diff update
  --number=D<n> --summary=...` then `jf sync -r D<n>`), not just the Test Plan — the
  Summary is where reviewers and future readers look for real-world impact.
