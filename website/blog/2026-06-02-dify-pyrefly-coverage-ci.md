---
title: Making Type Coverage Visible in Dify's CI
description: A case study in adding Pyrefly reporting, diagnostic diffs, and backend coverage signals to a large Python codebase.
slug: dify-pyrefly-coverage-ci
authors: [asukaminato]
tags: [typechecking, coverage, ci, python]
hide_table_of_contents: false
---

Dify is a large open-source platform for building LLM applications. Its backend is a Python Flask application with workflows, RAG pipelines, model providers, agents, Celery tasks, database migrations, and a large test suite. That makes it a useful case study for Pyrefly adoption: a real codebase where static analysis needs to fit into existing CI without blocking daily work.

The goal was not to make every Pyrefly diagnostic fail CI on day one. That would have been noisy and counterproductive. The better approach was to split the rollout into two CI surfaces: a blocking check for the files we were ready to enforce, and full-project reporting that stayed non-blocking and showed up in PR comments.

<!-- truncate -->

## Start with Measurement

For Dify, Pyrefly was added as a backend development dependency and configured in `api/pyproject.toml`:

```toml
[tool.pyrefly]
project-includes = ["."]
project-excludes = [".venv", "migrations/"]
python-platform = "linux"
python-version = "3.12.0"
infer-with-first-use = true
min-severity = "warn"
```

That project configuration gives Pyrefly enough information to analyze the backend in the same Python version and platform CI uses. But Dify also needed a second layer for adoption: a local exclude list at `api/pyrefly-local-excludes.txt`.

The blocking path uses that file to exclude known-problem areas while still failing on Pyrefly errors in the enforceable subset. The script builds a stricter command line, disables ignore-file heuristics, excludes broad legacy areas such as migrations and tests, then adds each path from `pyrefly-local-excludes.txt` as a `--project-excludes` entry.

That gives CI a practical gate: Pyrefly can block regressions where the project is ready, without pretending the whole repository is already clean.

## Keep a Full Non-Blocking Signal

The second path runs over the full backend and reports what changed.

```bash
uv run --directory api --dev pyrefly coverage report
```

The report produces structured JSON. Dify then renders the summary into a compact Markdown table for CI. A local run currently reports:

| Metric | Value |
| --- | ---: |
| Modules | 2790 |
| Typable symbols | 54,901 |
| Typed symbols | 24,989 |
| Untyped symbols | 29,635 |
| Any symbols | 277 |
| Type coverage | 46.02% |
| Strict coverage | 45.52% |

That number is a full-project baseline.

## Compare PRs

The non-blocking coverage workflow runs `pyrefly report` twice: once on the pull request branch and once on the base branch. A helper script extracts the summary and renders a comparison table with deltas for type coverage, strict coverage, typed symbols, untyped symbols, and module count.

This matters because large projects rarely improve through one giant typing push. Instead, they improve when contributors can see that a PR added 200 typed symbols, removed 20 untyped symbols, or accidentally moved coverage in the wrong direction.

The PR comment becomes a lightweight code review signal:

```text
### Pyrefly Type Coverage

| Metric | Base | PR | Delta |
| --- | ---: | ---: | ---: |
| Type coverage | 45.90% | 46.02% | +0.12% |
| Strict coverage | 45.40% | 45.52% | +0.12% |
| Typed symbols | 24,700 | 24,989 | +289 |
| Untyped symbols | 29,800 | 29,635 | -165 |
```

This encourages better typing without turning the first rollout into a migration project.

The workflow also avoids comment spam. Each comment starts with a stable marker like `### Pyrefly Type Coverage`; the GitHub Action looks for an existing comment with that marker and updates it in place. If no matching comment exists, it creates one. That keeps repeated pushes from filling the PR timeline with near-identical coverage tables.

## Keep Diagnostics Reviewable

Dify also has a non-blocking Pyrefly diff workflow. It runs `pyrefly check` on the PR branch and the base branch, normalizes the output, and posts a diff only when the diagnostic line count changes.

The normalization step is small but important. Full checker output includes source excerpts and caret lines, which create noisy diffs. Dify's helper keeps the diagnostic headline and location line:

```python
_DIAGNOSTIC_PREFIXES = ("ERROR ", "WARNING ")
_LOCATION_PREFIX = "-->"
```

That gives reviewers the part they need: what changed, and where. It avoids burying the signal inside pages of repeated context.

The diff comment uses the same pattern: one stable `### Pyrefly Diff` comment that gets updated as the PR changes.

## Handle Fork PRs Safely

One subtle CI detail is forked pull requests. Same-repository PRs can post comments directly from the pull request workflow. Forked PRs need a safer path.

Dify solves this by uploading structured artifacts from the untrusted workflow, then using a separate `workflow_run` job on trusted default-branch code to download the artifacts and render the comment. The trusted workflow posts the final PR comment.

That separation keeps the useful contributor experience while avoiding the common mistake of giving untrusted PR code write permissions.

## Fit Type Coverage Alongside Test Coverage

Pyrefly coverage is not a replacement for runtime coverage. Dify's backend test workflow still runs unit tests and integration tests separately, uploads coverage artifacts, combines them, prints a coverage summary, and uploads XML coverage to Codecov.

The two signals answer different questions:

- Runtime coverage asks: "Did tests execute this code?"
- Type coverage asks: "Can static analysis understand this interface?"

For a codebase like Dify, both matter. Runtime coverage catches behavior regressions. Type coverage catches API drift, missing annotations, accidental `Any`, and unclear boundaries before they become harder to reason about.

## What Worked

The most effective choices were simple:

- Keep two Pyrefly configurations: one blocking path with local excludes, and one full-project path that reports in comments.
- Compare PRs against base instead of enforcing a global threshold immediately.
- Render comments in Markdown so reviewers do not need to open CI logs.
- Update one stable PR comment per signal instead of creating a new comment on every push.
- Normalize diagnostics before diffing them.
- Keep fork-PR commenting on trusted code.
- Use the same dependency manager and working directory shape locally and in CI.

This setup makes typing visible, reviewable, and incremental. That is the part that matters most at the beginning.

## Looking Ahead

Once a project has stable reporting, it can decide where to tighten. Some teams may add thresholds. Others may focus on high-value modules first: request builders, service boundaries, decorators, provider contracts, or database-facing code.

For Dify, Pyrefly's value is already visible before strict enforcement. It gives contributors a concrete way to see whether a change made the Python backend easier or harder to understand. In a fast-moving LLM application platform, that feedback loop is worth a lot.
