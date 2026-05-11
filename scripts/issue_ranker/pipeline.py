#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""5-pass LLM pipeline orchestrator.

Runs all 5 passes in sequence, managing context between passes.
Logs each pass with timing and cost estimates.

Supports resuming from cached pass results to iterate on later passes
without re-running expensive earlier passes.
"""

from __future__ import annotations

import json
import logging
import time

from .passes.categorize import categorize_all
from .passes.dependencies import build_dependencies
from .passes.primer_impact import compute_primer_impact
from .passes.rank import rank_issues
from .passes.score import score_all


# Cost estimates per call (input + output tokens, approximate)
_COST_ESTIMATES = {
    "haiku_call": 0.002,  # ~2K tokens in, ~200 out
    "sonnet_call": 0.03,  # ~5K tokens in, ~500 out
    "sonnet_large": 0.10,  # ~20K tokens in, ~2K out
    "opus_large": 0.75,  # ~20K tokens in, ~8K out
}


def run_pipeline(
    issue_data: dict,
    primer_data: dict | None = None,
    pass_results_cache: str | None = None,
) -> dict:
    """Run the full 5-pass ranking pipeline.

    Args:
        issue_data: Output from --mode collect (issues + relationships)
        primer_data: Output from compare_typecheckers.py --output-json
        pass_results_cache: Path to save/load intermediate pass results.
            If the file exists and contains valid results, passes 1-4
            are skipped and only Pass 5 is re-run.

    Returns:
        Full pipeline results including final ranking, V1 gap analysis,
        and all intermediate pass results.
    """
    issues = issue_data.get("issues", [])
    relationships = issue_data.get("relationships", {})

    if not issues:
        logging.warning("No issues to rank")
        return {"ranking": {}, "pass_results": {}}

    total_start = time.time()
    estimated_cost = 0.0

    # Try to load cached pass results (skip passes 1-4)
    cached = None
    if pass_results_cache:
        try:
            with open(pass_results_cache) as f:
                cached = json.load(f)
            if all(
                k in cached
                for k in ("categorizations", "primer_impacts", "dep_graph", "scores")
            ):
                logging.info(
                    f"=== Loaded cached pass results from {pass_results_cache} ==="
                )
                logging.info("  Skipping passes 1-4, re-running Pass 5 only.")
            else:
                logging.info(
                    "  Cache file exists but incomplete, running full pipeline"
                )
                cached = None
        except (FileNotFoundError, json.JSONDecodeError):
            cached = None

    if cached:
        # Convert string keys back to int keys
        categorizations = {int(k): v for k, v in cached["categorizations"].items()}
        primer_impacts = {int(k): v for k, v in cached["primer_impacts"].items()}
        dep_graph = cached["dep_graph"]
        scores = {int(k): v for k, v in cached["scores"].items()}
    else:
        # === Pass 1: Categorize (Haiku, 1 call per issue) ===
        logging.info(f"=== Pass 1: Categorize ({len(issues)} issues, Haiku) ===")
        start = time.time()
        categorizations = categorize_all(issues)
        pass1_time = time.time() - start
        pass1_cost = len(issues) * _COST_ESTIMATES["haiku_call"]
        estimated_cost += pass1_cost
        logging.info(f"  Pass 1 complete: {pass1_time:.1f}s, ~${pass1_cost:.2f}")

        # === Pass 2: Primer Impact (mostly deterministic + ~20 Haiku calls) ===
        logging.info(f"=== Pass 2: Primer Impact ({len(issues)} issues) ===")
        start = time.time()
        primer_impacts = compute_primer_impact(issues, primer_data, categorizations)
        pass2_time = time.time() - start
        pass2_cost = 20 * _COST_ESTIMATES["haiku_call"]  # worst case
        estimated_cost += pass2_cost
        logging.info(f"  Pass 2 complete: {pass2_time:.1f}s, ~${pass2_cost:.2f}")

        # === Pass 3: Dependencies (Opus, 1 call) ===
        logging.info("=== Pass 3: Dependency Graph (1 Opus call) ===")
        start = time.time()
        try:
            dep_graph = build_dependencies(issues, categorizations, relationships)
        except Exception as e:
            logging.warning(f"  Pass 3 failed: {e}, using empty dependency graph")
            dep_graph = {
                "dependency_groups": [],
                "blocking_chains": [],
                "duplicate_clusters": [],
            }
        pass3_time = time.time() - start
        pass3_cost = _COST_ESTIMATES["opus_large"]
        estimated_cost += pass3_cost
        logging.info(f"  Pass 3 complete: {pass3_time:.1f}s, ~${pass3_cost:.2f}")

        # === Pass 4: Priority Score (Sonnet, 1 call per issue) ===
        logging.info(f"=== Pass 4: Priority Score ({len(issues)} issues, Sonnet) ===")
        start = time.time()
        scores = score_all(issues, categorizations, primer_impacts, dep_graph)
        pass4_time = time.time() - start
        pass4_cost = len(issues) * _COST_ESTIMATES["sonnet_call"]
        estimated_cost += pass4_cost
        logging.info(f"  Pass 4 complete: {pass4_time:.1f}s, ~${pass4_cost:.2f}")

        # Save intermediate results for future re-ranking
        if pass_results_cache:
            cache_data = {
                "categorizations": {str(k): v for k, v in categorizations.items()},
                "primer_impacts": {str(k): v for k, v in primer_impacts.items()},
                "dep_graph": dep_graph,
                "scores": {str(k): v for k, v in scores.items()},
            }
            with open(pass_results_cache, "w") as f:
                json.dump(cache_data, f, indent=2)
            logging.info(f"  Pass results cached to {pass_results_cache}")

    # === Pass 5: Final Ranking (Opus, batched) ===
    n_batches = (len(issues) + 49) // 50  # BATCH_SIZE = 50 in rank.py
    logging.info(f"=== Pass 5: Final Ranking ({n_batches} Opus calls, batched) ===")
    start = time.time()
    try:
        ranking = rank_issues(
            issues, scores, categorizations, primer_impacts, dep_graph
        )
    except Exception as e:
        logging.error(f"  Pass 5 failed completely: {e}, using score-based ordering")
        # Fallback: sort by score with proper tiering
        sorted_issues = sorted(
            issues,
            key=lambda i: scores.get(i.get("number", 0), {}).get("priority_score", 0),
            reverse=True,
        )
        from passes.rank import _mechanical_tier

        fallback = _mechanical_tier(sorted_issues, scores)
        ranking = {
            **fallback,
            "summary": f"Fallback ranking by score (Pass 5 failed: {e})",
        }
    pass5_time = time.time() - start
    pass5_cost = n_batches * _COST_ESTIMATES["opus_large"]
    estimated_cost += pass5_cost
    logging.info(f"  Pass 5 complete: {pass5_time:.1f}s, ~${pass5_cost:.2f}")

    total_time = time.time() - total_start
    logging.info(
        f"\n=== Pipeline complete: {total_time:.1f}s total, "
        f"~${estimated_cost:.2f} estimated cost ==="
    )

    return {
        "ranking": ranking,
        "pass_results": {
            "categorizations": categorizations,
            "primer_impacts": primer_impacts,
            "dep_graph": dep_graph,
            "scores": scores,
        },
        "timing": {
            "pass5_ranking": round(pass5_time, 1),
            "total": round(total_time, 1),
        },
        "cost_estimate": round(estimated_cost, 2),
    }
