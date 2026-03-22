# Issue Ranker Pipeline

An automated pipeline that fetches GitHub issues, enriches them with type
checker data and primer results, then ranks them by priority using a multi-pass
LLM pipeline.

## Architecture

The pipeline has two main phases: **collect** and **rank**.

**Collect phase** (`--mode collect`): Fetches and enriches GitHub issues.
1. Fetch issues from GitHub API (`github_issues.py`)
2. Categorize as typechecking vs non-typechecking (`__main__.py:_is_typechecking_issue`)
3. Extract code blocks from issue bodies — regex + LLM fallback (`code_extractor.py`)
4. Repair broken snippets with LLM (`code_extractor.py:repair_snippet`)
5. Run pyrefly/pyright/mypy on snippets (`issue_checker.py`, `dep_resolver.py`)
6. Classify issue status from checker results (`status_classifier.py`)
7. Resolve relationships — duplicates, blockers, parents (`relationship_resolver.py`)

**Rank phase** (`--mode rank`): 5-pass LLM pipeline (`pipeline.py`).
1. **Categorize** (`passes/categorize.py`) — classify issue type/area
2. **Primer impact** (`passes/primer_impact.py`) — match issues to primer errors
   (mostly deterministic string matching with LLM fuzzy fallback)
3. **Dependencies** (`passes/dependencies.py`) — identify blocking relationships
4. **Score** (`passes/score.py`) — weighted signals including false positives,
   performance, strategic adoption, team priority, IDE usability, primer
   breadth, and more (see `_SYSTEM_PROMPT` in `score.py` for current weights)
5. **Rank** (`passes/rank.py`) — final ordering with tie-breaking

## Key files

- `__main__.py` — CLI entry point, collect/rank/full modes
- `pipeline.py` — orchestrates the 5-pass LLM ranking
- `issue_checker.py` — runs pyrefly/pyright/mypy on code snippets
- `dep_resolver.py` — scans code snippets for imports, resolves pip package
  names, and pre-installs dependencies before running checkers
- `llm_transport.py` — LLM API calls (Anthropic + Llama) with retry logic
- `report_formatter.py` — generates markdown and JSON output
- `spec_fetcher.py` — fetches typing spec sections for grounding

## Primer pipeline (`scripts/compare_typecheckers.py`)

Clones ~137 open-source Python projects and runs pyrefly/pyright/mypy on each.
Produces `primer_errors.json` consumed by the ranker's primer_impact pass.

- Supports sharding via `--shard-index` / `--num-shards` for parallel CI runs
- `scripts/merge_primer_shards.py` merges shard outputs back together
- Timeout marker: `"CHECKER_TIMEOUT"` (not plain `"timeout"`)

## GitHub Actions workflow (`.github/workflows/issue_ranking.yml`)

- **primer** job: 4-shard matrix running `compare_typecheckers.py`
- **primer-merge** job: merges shard artifacts with `merge_primer_shards.py`
- **rank** job: runs the full collect + rank pipeline
- See the workflow file for timeouts, secrets, and token requirements

## Running locally

```bash
# Primer (needs pyrefly binary — use --pyrefly if no cargo)
python3 scripts/compare_typecheckers.py --output /tmp/primer.json \
  --pyrefly /path/to/pyrefly

# Collect issues (needs GITHUB_TOKEN)
python3 -m scripts.issue_ranker --mode collect \
  --pyrefly /path/to/pyrefly --output /tmp/issues.json

# Rank (needs ANTHROPIC_API_KEY)
python3 -m scripts.issue_ranker --mode rank \
  --primer-data /tmp/primer.json --issue-data /tmp/issues.json \
  --output /tmp/ranking.md --output-json /tmp/ranking.json
```

## Tests

- Unit tests: `scripts/issue_ranker/tests/`
- LLM integration tests: `scripts/issue_ranker/llm_tests/`
- Run with buck: `buck test pyrefly:issue_ranker_tests`

## Key findings (March 2026)

- **Primer data matters**: Running type checkers on 137 real-world projects
  grounds rankings in actual impact. Without primer data, issue scores can
  swing 30-44 points and tier distributions shift dramatically.
- **Labels matter**: Including GitHub project priorities (P0/P1/P2) boosts
  V1 milestone overlap by ~16 percentage points.
- Both signals combined produce the best ranking quality.
