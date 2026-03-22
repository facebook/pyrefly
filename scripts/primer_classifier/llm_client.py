# @nolint
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""LLM client for classifying primer diff entries.

Classification-specific logic: system prompts, response parsing with
strict key validation, and the classify / verdict / suggestion passes.

Low-level API transport (backend detection, HTTP calls, retries) lives in
``llm_transport`` so that both primer_classifier and issue_ranker can
share it without depending on each other's internals.
"""

from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Optional

# Import shared LLM transport layer.  We keep the old private names so
# that primer_classifier/test_classifier.py (which patches and imports
# them by name) continues to work without changes.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from llm_transport import (
    LLMError,
    get_backend as _get_backend,
    call_llama_api as _call_llama_api,
    call_anthropic_api as _call_anthropic_api,
    extract_text as _extract_text_from_response,
)


@dataclass
class CategoryVerdict:
    """Verdict for a single error category within a project."""

    category: str
    verdict: str
    reason: str


@dataclass
class LLMResponse:
    """Response from the LLM classification call."""

    verdict: str  # "regression", "improvement", "neutral"
    reason: str  # human-readable explanation
    categories: list[CategoryVerdict] = field(default_factory=list)
    needs_files: list[str] = field(
        default_factory=list
    )  # file paths the LLM wants to see
    pr_attribution: str = ""  # which part of the PR diff caused the change
    raw_response: Optional[dict] = None


def _build_system_prompt() -> str:
    return """You are classifying changes in pyrefly's type checking output. Pyrefly is a Python type checker. You are evaluating whether pyrefly got BETTER or WORSE, not whether the user's code is good or bad.

'+' lines are NEW errors that pyrefly now reports (didn't before).
'-' lines are errors that pyrefly no longer reports (used to report).

For large projects, errors may be grouped into CATEGORIES instead of listed individually. Each category shows the error kind, affected class, count, example message, affected attributes, and files. Use this aggregate view to assess the overall pattern.

Classify as one of:
- "improvement": Pyrefly got better. This means:
  - New errors ('+') that correctly catch REAL bugs in the code (true positives — pyrefly is now smarter)
  - Removed errors ('-') that were wrong (false positives removed — pyrefly is now less noisy)
- "regression": Pyrefly got worse. This means:
  - New errors ('+') that are WRONG — the code is actually correct and pyrefly is flagging it incorrectly (false positives introduced)
  - Removed errors ('-') where the code actually had a bug that pyrefly used to catch but no longer does
- "neutral": Message wording changes with no behavioral impact

NEW ERRORS fall into exactly one of three categories — you must determine which:
1. **Spec says so, code is wrong** → IMPROVEMENT. The typing spec requires this check, pyrefly applies it correctly, and the code genuinely violates the rule. Pyrefly is right to flag it.
2. **Pyrefly is just wrong** → REGRESSION. The error is incorrect — pyrefly misunderstands the code (e.g., inference failure producing `Never`, missing attribute that actually exists via inheritance). The spec doesn't support this error.
3. **Too strict** → REGRESSION. The typing spec defines a rule, but pyrefly applies it more broadly than the spec requires. The rule exists but doesn't apply in this context. For example, a rule that the spec limits to protocols being applied to regular classes, or a rule that the spec limits to generic types being applied to non-generic types. Even though there may be a theoretical type issue, mypy/pyright would not flag it because the spec doesn't require it here.

REMOVED ERRORS ('-') require the OPPOSITE reasoning — you must determine which:
1. **The error was a false positive** → IMPROVEMENT. Pyrefly was wrong to flag it before (e.g., flagging a valid callable as not-callable, claiming an attribute is missing when it exists via inheritance, reporting an incorrect type mismatch). Removing a false positive means pyrefly got smarter.
2. **The error was catching a real bug** → REGRESSION. The code genuinely has a type error that pyrefly used to catch but no longer does (e.g., removing detection of actual argument count mismatches, actual missing attributes, actual type incompatibilities). Pyrefly lost capability.

When ALL changes are removals ('-'), the most common explanation is that pyrefly fixed false positives (improvement). Ask yourself: "Were these errors correct?" If the errors claimed something was wrong but the code was actually fine (e.g., claiming a callable is not callable, claiming an attribute is missing when it exists), removing them is an IMPROVEMENT. Only classify removals as regression if the errors were genuinely catching real bugs.

KEY RULES:
1. If a new error ('+') correctly identifies a real bug in the source code, that is an IMPROVEMENT — pyrefly is catching something it should catch. Even if the code is buggy, pyrefly finding it is a good thing.
2. Keep your reasoning internally consistent. If your analysis describes a genuine type error, inconsistency, or bug in the code, that points toward "improvement" — not "regression". If your analysis describes the removed errors as false positives, inference failures, or incorrect checks, that points toward "improvement" — removing wrong errors means pyrefly got better. Only reason toward "regression" for removals if the errors were genuinely catching real bugs.
3. A "bad-override" where the child class truly has an inconsistent type signature vs the parent is a REAL bug — that is an improvement, not a regression.
4. MISSING-ATTRIBUTE PATTERN: When you see many `missing-attribute` errors across a well-known, well-tested project (e.g., mypy, discord.py, xarray, dulwich), and the errors claim attributes like `data`, `dims`, `fullname`, `parents`, etc. are missing from core classes, this is almost always a regression. These are fundamental attributes that the project uses extensively — they are defined somewhere (often via `__slots__` in parent classes, descriptors, or dynamic assignment). The type checker is failing to resolve them through the class hierarchy. Be especially skeptical when:
   - The same class has many "missing" attributes (suggests the checker can't see the class's attribute definitions)
   - The attributes are basic/fundamental (e.g., `data`, `name`, `tree`, `parents` on a `Commit` class)
   - The project is mature and well-tested (unlikely to have 50+ real bugs in core attribute access)
5. TYPE CHECKER LIMITATIONS: Be aware that type checkers have known blind spots. When errors arise from patterns the type checker cannot model (e.g., runtime-injected values, dynamic attribute assignment, C extension modules, conditional imports), those are false positives (regressions), not real bugs being caught. Similarly, when removing one concise error creates many scattered downstream errors for the same root cause, that is worse error reporting quality (regression), not an improvement.
6. NEVER/@_ INFERENCE FAILURES: When error messages contain `Never` or `@_` types, this almost always indicates a type inference failure rather than a real bug. These are regressions. Specific patterns:
   - `Expected class object, got 'Never'` → inference failure, REGRESSION
   - `list[Never]`, `Generator[Never, ...]`, `dict[str, Never]` → inference failure, REGRESSION
   - Arguments showing `Never` where a concrete type should be → inference failure, REGRESSION
   - Types changing FROM concrete types TO `Never`/`@_` → type resolution got WORSE, REGRESSION
   - `list[Unknown] | list[Never]` appearing where `list[SomeConcreteType]` is expected → REGRESSION
   NOTE: `Unknown` in error messages is different from `Never`. `Unknown` often means the code is untyped and pyrefly is correctly reporting that untyped values don't match typed parameters. An error like "Argument `dict[Unknown, Unknown] | Unknown` is not assignable to parameter X" may be a genuine type issue (the untyped value truly isn't compatible). Evaluate `Unknown` errors on their merits — they can be real bugs being caught.
7. TEST FILE ERRORS: Errors in test files (paths containing `/tests/`, `/test_`, `_test.py`, `conftest.py`) need case-by-case evaluation. Test code CAN have real type issues:
   - Calling `.equals()` on an `int` → real bug (method doesn't exist)
   - `missing-attribute` on a concrete class (not a mock) → likely real bug
   - Wrong return type from a fixture → real type issue
   But some test patterns produce false positives:
   - `not-callable` on values from fixtures/parametrize → often false positive
   - `missing-attribute` on mock objects → false positive
   - Errors from complex parametrize type inference → often false positive
   Evaluate each error individually rather than dismissing all test file errors.
8. TYPE STUB PROJECTS: Projects whose names end in `-stubs` (e.g., `django-stubs`, `boto3-stubs`) are type stub packages. They provide type annotations for other libraries and are extensively tested against mypy/pyright. New errors from pyrefly on these projects are almost always REGRESSIONS — they indicate pyrefly disagrees with established type checker behavior, not that the stubs are wrong.
9. WELL-KNOWN EXEMPT PATTERNS: Some code patterns are technically type violations but are universally accepted by the Python ecosystem. New errors flagging these patterns are REGRESSIONS because they create noise with no practical value:
   - Re-assigning `typing.TYPE_CHECKING` (e.g., `TYPE_CHECKING = True`) — this is a standard hack used by many projects and mypy/pyright explicitly allow it, even though `TYPE_CHECKING` is `Final[bool]` in typeshed
   - Re-assigning module-level constants in feature detection blocks (try/except) — common in projects like urllib3, ssl, etc.
   - Star imports that re-export `Final` names from other modules
   If a new error flags one of these patterns and mypy/pyright would NOT flag it, classify as REGRESSION.

IMPORTANT: Source code from the affected project will be fetched from GitHub and provided below the errors when available. You MUST analyze the actual code in depth to support your verdict. Do NOT just rephrase the error message. You must:
- Reference specific lines, variable names, class definitions, and method signatures from the source code
- Explain WHY the code is buggy (e.g. "class X defines __gt__ but not __lt__, so min() has no way to compare") or WHY it is correct (e.g. "the variable is actually of type Y because of the assignment on line N")
- If a method override is involved, describe what the parent class expects vs what the child provides
- If a missing method/protocol is involved, explain which method is missing and why it matters
If source code is NOT provided, state this in your reasoning and note that your confidence is lower.

REASONING QUALITY — CRITICAL:
Your explanations will be read by experienced engineers reviewing PRs. Factual errors in your reasoning will undermine trust in the classifier. Follow these rules:
- REACHABILITY: A line is ONLY unreachable if preceded by `return`, `raise`, `break`, `continue`, `sys.exit()`, or an unconditional infinite loop. A bare function call like `super().__gt__(other)` or `self.validate()` does NOT make subsequent code unreachable — execution continues to the next line. NEVER say "unreachable code" or "dead code" unless one of those flow-terminating statements is present. If the issue is that a return value is discarded (e.g., `super().__gt__(other)` without `return`), say "return value is discarded" — that is a different bug from unreachable code.
- DISTINGUISH TYPE-LEVEL vs RUNTIME impact. A `bad-override` where the return type is `bool | tuple` instead of `bool` is a Liskov substitution violation at the TYPE level, even if the buggy branch is rarely hit at runtime. Say "type-level violation" not "would cause runtime errors" unless you can trace an actual runtime crash path.
- IDENTIFY CASCADE ERRORS. When multiple errors stem from the same root cause (e.g., a bad-override on `__gt__` causes `min()` to fail its overload check), explain the root cause and note the cascade. Don't treat cascade errors as independent bugs.
- CHECK INHERITANCE. Before claiming something is missing or broken, check the parent class. NamedTuple inherits from tuple, so a NamedTuple subclass IS a tuple — `isinstance(x, tuple)` will be True. ABC subclasses inherit abstract methods. `__slots__` may be defined in a parent.
- DO NOT HALLUCINATE CODE BEHAVIOR. If you're not sure what a line does, say so. Don't claim a function "returns None" unless you can see that it actually does.
- KEEP REASONING CONCISE but ACCURATE. A short, correct explanation is better than a long, partly-wrong one.

You MUST cite the relevant typing spec section with a link (https://typing.readthedocs.io/en/latest/spec/...) when your verdict depends on type system rules. BEFORE applying a type system rule, verify from the spec that the rule applies in this specific context — check all preconditions and scope limitations of the rule.

CRITICAL — MYPY/PYRIGHT CROSS-CHECK: Each error is annotated with whether mypy and pyright also flag the same file:line (matched semantically — same issue, possibly different line/wording). Use these as one signal among many — they inform but do not override your code analysis.
- Errors marked [pyrefly-only]: Neither mypy nor pyright flag these. This suggests pyrefly may be too strict or has an inference bug, BUT pyrefly could also be correctly catching something others miss. Analyze the actual error to decide.
- Errors marked [mypy: yes] and/or [pyright: yes]: Co-reported by other checkers. This is strong evidence the error is a real bug in the user's code, not a pyrefly false positive. Do NOT classify co-reported errors as false positives unless you have specific evidence that all three checkers are wrong.
- IMPORTANT: Cross-check annotations apply to individual errors, NOT the overall project verdict. The net impact matters most: if a project removes 138 errors and adds 14 (even if pyrefly-only), that is a net improvement — the project has 124 fewer false positives. Do NOT let a few pyrefly-only additions override a large net reduction in errors.
- If no cross-check annotations are present, fall back to your training data knowledge of mypy/pyright behavior.

INTENTIONAL REGRESSIONS vs SPEC COMPLIANCE: Sometimes a PR intentionally introduces stricter behavior to comply with the typing spec. When the PR description states conformance intent AND cites a specific spec section:
- If the spec clearly supports the new check, classify as "improvement" — the projects producing new errors were relying on a bug in pyrefly. The old (lenient) behavior was the bug, not the new (correct) behavior.
- Frame it clearly with a concrete example pulled from the actual errors. Show the specific type pattern from the error messages (e.g., the actual types involved) so reviewers can immediately understand what changed.
- The mypy/pyright cross-check is informational, not decisive. Pyrefly may be ahead of other tools on spec compliance, and that's fine.
- Do NOT suggest reverting spec-compliant behavior in fix suggestions.

OUTPUT FORMAT:

Do NOT include a "verdict" field — your job in this pass is ONLY to analyze and reason. A separate step will assign the verdict based on your reasoning.

When errors are grouped into categories, provide reasoning for EACH category separately. Different categories within the same project may have different conclusions.

REQUESTING ADDITIONAL FILES:

If you cannot confidently determine the answer because you need to see source code from another file (e.g., a parent class definition, a module that defines __slots__, a base class with the overridden method), you may request those files instead of guessing. Respond with:
{"needs_files": ["path/to/parent_class.py", "path/to/base.py"]}

Use the project's file paths (e.g., "dulwich/objects.py", "discord/permissions.py"). Request at most 3 files. Only request files when you genuinely need them to verify the conclusion — if the pattern is clear enough (e.g., 100+ missing-attribute errors on core classes of a well-tested project), classify directly without requesting files.

If you have enough context, respond with your analysis. IMPORTANT: reason through the evidence fully:
{"spec_check": "which typing spec rule applies, with a link, AND what is its scope — which constructs does it apply to?", "runtime_behavior": "would this cause an actual runtime error or crash? trace the execution path", "mypy_pyright": "what do the mypy/pyright cross-check annotations show? summarize the data and its implications", "removal_assessment": "For removed errors ('-'): were they correct (catching real bugs) or incorrect (false positives/too strict)? State which and why. Write 'N/A' if there are no removed errors.", "pr_attribution": "Which specific change(s) in the pyrefly PR diff caused this project's errors to change? Reference specific files, functions, or code patterns from the diff. Write 'N/A' if no PR diff was provided.", "reason": "explanation — if a typing spec rule is relevant, include the link (e.g., https://typing.readthedocs.io/en/latest/spec/...) so reviewers can verify", "categories": [{"category": "short label", "reason": "explanation"}, ...]}

The "spec_check", "runtime_behavior", and "mypy_pyright" fields force you to think through the evidence. If the spec doesn't require the check here AND there's no runtime impact AND mypy/pyright wouldn't flag it, it's almost certainly "too strict". The "categories" field is optional — omit it for small diffs with no categories. When present, each entry should correspond to an error category from the input.

PR DIFF ATTRIBUTION:
When a pyrefly PR diff is provided, identify which specific code change(s) in the diff caused the primer errors to appear or disappear. IMPORTANT: Always use the FULL file path exactly as it appears in the diff (e.g., `pyrefly/lib/alt/answers.rs`, not just `answers.rs`). Reference the function/method name with parentheses (e.g., `overload_resolution()`). Be specific — e.g., "the change to `overload_resolution()` in `pyrefly/lib/alt/answers.rs` relaxed the constraint on callable types, which removed the false positive `not-callable` errors." If no PR diff is provided, write "N/A" for pr_attribution.

In the reason fields, keep explanations concise. When your analysis depends on a typing spec rule, you MUST include the relevant spec link in the reason field (e.g., "per the [typing spec](https://typing.readthedocs.io/en/latest/spec/generics.html#variance), variance inference is only required for protocols")."""


def _build_user_prompt(
    errors_text: str,
    source_context: Optional[str],
    change_type: str,
    structural_signals: Optional[str] = None,
    pyrefly_diff: Optional[str] = None,
    pr_description: Optional[str] = None,
) -> str:
    parts = [
        f"Change type: {change_type} ('+' = new error on PR branch, '-' = removed error)\n",
        f"Errors:\n{errors_text}\n",
    ]
    if pr_description:
        parts.append(f"PR description (author's stated intent):\n{pr_description}\n")
    if structural_signals:
        parts.append(f"\n{structural_signals}\n")
    if source_context:
        parts.append(
            f"Source code at error location (line marked with >>>):\n{source_context}\n"
        )
    else:
        parts.append("Source code: not available (could not fetch from GitHub)\n")
    if pyrefly_diff:
        parts.append(
            f"Pyrefly PR diff (the code changes that caused the primer results above):\n{pyrefly_diff}\n"
        )

    return "\n".join(parts)


# Keys expected in classifier JSON responses.  Used by _parse_classification
# to distinguish real classification output from random JSON the LLM quotes.
_CLASSIFICATION_KEYS = frozenset(
    {
        "verdict",
        "needs_files",
        "reason",
        "suggestions",
        "categories",
        "spec_check",
        "runtime_behavior",
        "mypy_pyright",
        "removal_assessment",
        "pr_attribution",
        "summary",
    }
)


def _parse_classification(text: str) -> dict:
    """Parse a classification JSON from the LLM response text.

    Strict variant: only accepts JSON objects containing at least one
    expected classification key.  This distinguishes classifier JSON
    from random JSON that may appear in the LLM's reasoning text
    (e.g., a code snippet the LLM quotes).

    Use ``llm_transport.parse_json()`` when you want generic JSON
    extraction without key validation.
    """

    def _is_classification(d: dict) -> bool:
        """Return True if *d* looks like a classifier response."""
        return any(k in d for k in _CLASSIFICATION_KEYS)

    # Try the full text first
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict) and _is_classification(parsed):
            return parsed
    except json.JSONDecodeError:
        pass

    # Strip markdown code fences
    stripped = re.sub(r"```(?:json)?\s*", "", text).strip()
    try:
        parsed = json.loads(stripped)
        if isinstance(parsed, dict) and _is_classification(parsed):
            return parsed
    except json.JSONDecodeError:
        pass

    # Find JSON objects by looking for { and balancing braces
    for i, ch in enumerate(text):
        if ch == "{":
            depth = 0
            for j in range(i, len(text)):
                if text[j] == "{":
                    depth += 1
                elif text[j] == "}":
                    depth -= 1
                    if depth == 0:
                        candidate = text[i : j + 1]
                        try:
                            parsed = json.loads(candidate)
                            if isinstance(parsed, dict) and _is_classification(parsed):
                                return parsed
                        except json.JSONDecodeError:
                            pass
                        break

    raise LLMError(f"Could not parse LLM response as classification JSON: {text}")


def classify_with_llm(
    errors_text: str,
    source_context: Optional[str] = None,
    change_type: str = "mixed",
    model: Optional[str] = None,
    structural_signals: Optional[str] = None,
    pyrefly_diff: Optional[str] = None,
    pr_description: Optional[str] = None,
) -> LLMResponse:
    """Pass 1: Send errors + context to the LLM for reasoning (no verdict).

    The LLM produces reasoning, pr_attribution, and per-category analysis,
    but does NOT assign a verdict. Call assign_verdict_with_llm() afterward
    to get the verdict based on the reasoning.

    Uses Llama API if LLAMA_API_KEY is set, otherwise Anthropic.
    Raises LLMError if the API call fails or the response is unparsable.
    """
    backend, api_key = _get_backend()
    if backend == "none":
        raise LLMError(
            "No API key found. Set LLAMA_API_KEY (Meta internal) "
            "or CLASSIFIER_API_KEY / ANTHROPIC_API_KEY."
        )

    system_prompt = _build_system_prompt()
    user_prompt = _build_user_prompt(
        errors_text,
        source_context,
        change_type,
        structural_signals,
        pyrefly_diff,
        pr_description,
    )

    print(
        f"Using {backend} backend for classification (pass 1: reasoning)",
        file=sys.stderr,
    )

    if backend == "llama":
        result = _call_llama_api(api_key, system_prompt, user_prompt, model)
    else:
        result = _call_anthropic_api(api_key, system_prompt, user_prompt, model)

    text = _extract_text_from_response(backend, result)
    classification = _parse_classification(text)

    # Check if the LLM is requesting additional files
    needs_files = classification.get("needs_files", [])
    if needs_files and isinstance(needs_files, list):
        return LLMResponse(
            verdict="",
            reason="",
            needs_files=[f for f in needs_files if isinstance(f, str)][:3],
            raw_response=result,
        )

    reason_val = classification.get("reason", "No reason provided")
    # The LLM sometimes returns a dict for "reason" instead of a string.
    # If so, serialize it so downstream code always sees a string.
    reason = json.dumps(reason_val) if isinstance(reason_val, dict) else str(reason_val)

    # Parse per-category reasoning (no verdicts in pass 1)
    categories: list[CategoryVerdict] = []
    for cat_data in classification.get("categories", []):
        categories.append(
            CategoryVerdict(
                category=cat_data.get("category", "unknown"),
                verdict="",  # verdict assigned in pass 2
                reason=cat_data.get("reason", ""),
            )
        )

    # Parse pr_attribution if present
    pr_attribution = classification.get("pr_attribution", "")

    return LLMResponse(
        verdict="",  # verdict assigned in pass 2
        reason=reason,
        categories=categories,
        pr_attribution=pr_attribution,
        raw_response=result,
    )


def _build_verdict_system_prompt() -> str:
    """Build the system prompt for pass 2: assigning a verdict from reasoning."""
    return """You are assigning a verdict based on reasoning about pyrefly type checker changes. Pyrefly is a Python type checker. You are evaluating whether pyrefly got BETTER or WORSE.

You will receive reasoning from a prior analysis pass. Your job is to read that reasoning and assign the correct verdict.

Rules:
- If the reasoning describes removed errors as false positives, inference failures, too strict, or incorrect → "improvement" (pyrefly got better by removing wrong errors)
- If the reasoning describes removed errors as catching real bugs, genuine type errors, or valid checks → "regression" (pyrefly lost a real check)
- If the reasoning describes new errors as correctly catching real bugs → "improvement" (pyrefly is now smarter)
- If the reasoning describes new errors as wrong, false positives, or too strict → "regression" (pyrefly got worse)
- If the reasoning describes only message wording changes → "neutral"
- SPEC COMPLIANCE: If the reasoning says new errors are correct per the typing spec and the PR is implementing spec-required behavior, classify as "improvement" even if mypy/pyright don't enforce the same rule yet. Spec compliance is the primary authority.
- TOO STRICT (without spec basis): If the reasoning says new errors have NO spec basis and mypy/pyright don't flag them → "regression".

Respond with JSON only:
{"verdict": "regression|improvement|neutral", "categories": [{"category": "short label", "verdict": "regression|improvement|neutral"}, ...]}

The "categories" field is optional — omit it if there are no categories in the reasoning. When present, each entry should match a category from the reasoning."""


def _build_verdict_prompt(reason: str, categories: list[CategoryVerdict]) -> str:
    """Build the user prompt for pass 2: the reasoning from pass 1."""
    parts = [f"Reasoning from analysis:\n{reason}\n"]
    if categories:
        parts.append("Per-category reasoning:")
        for cat in categories:
            parts.append(f"- {cat.category}: {cat.reason}")
    return "\n".join(parts)


def assign_verdict_with_llm(
    reason: str,
    categories: list[CategoryVerdict],
    model: Optional[str] = None,
) -> tuple[str, list[CategoryVerdict]]:
    """Pass 2: Assign a verdict based on the reasoning from pass 1.

    Makes a small, cheap API call (~500 tokens in, ~100 tokens out) that
    reads the reasoning and assigns verdicts. Returns (overall_verdict,
    categories_with_verdicts).
    """
    backend, api_key = _get_backend()
    if backend == "none":
        raise LLMError(
            "No API key found. Set LLAMA_API_KEY (Meta internal) "
            "or CLASSIFIER_API_KEY / ANTHROPIC_API_KEY."
        )

    system_prompt = _build_verdict_system_prompt()
    user_prompt = _build_verdict_prompt(reason, categories)

    print(f"Using {backend} backend for verdict assignment (pass 2)", file=sys.stderr)

    if backend == "llama":
        result = _call_llama_api(api_key, system_prompt, user_prompt, model)
    else:
        result = _call_anthropic_api(api_key, system_prompt, user_prompt, model)

    text = _extract_text_from_response(backend, result)
    parsed = _parse_classification(text)

    verdict = parsed.get("verdict", "").lower().strip()
    if verdict not in ("regression", "improvement", "neutral"):
        print(
            f"Warning: verdict pass returned unexpected verdict '{verdict}', "
            "treating as ambiguous",
            file=sys.stderr,
        )
        verdict = "neutral"

    # Merge per-category verdicts back into the category objects
    verdict_by_category: dict[str, str] = {}
    for cat_data in parsed.get("categories", []):
        cat_verdict = cat_data.get("verdict", "").lower().strip()
        if cat_verdict not in ("regression", "improvement", "neutral"):
            cat_verdict = "neutral"
        verdict_by_category[cat_data.get("category", "")] = cat_verdict

    updated_categories = []
    for cat in categories:
        updated_categories.append(
            CategoryVerdict(
                category=cat.category,
                verdict=verdict_by_category.get(cat.category, verdict),
                reason=cat.reason,
            )
        )

    return verdict, updated_categories


def _build_suggestion_system_prompt() -> str:
    """Build the system prompt for Pass 3: aggregate suggestion generation."""
    return """You are analyzing aggregate results from pyrefly's primer classifier. Pyrefly is a Python type checker written in Rust. You have the classification results for ALL projects affected by a PR, plus the PR's code diff.

Your job: identify which specific code change(s) in the PR caused regressions, and suggest concrete source code fixes.

ACTIONABILITY REQUIREMENTS — your suggestions MUST be specific:
- Name the exact function or method from the diff that needs changing (e.g., "In calculate_abstract_members() in class_metadata.rs")
- Describe what the fix looks like in pseudo-code (e.g., "add a guard: if field.is_synthesized() { include it in abstract_members }")
- State the expected outcome (e.g., "eliminates 7 bad-override errors across 3 projects")
- List which projects are affected and which error kinds would be fixed

BAD (vague): "Add a guard condition to the override checking logic"
GOOD (actionable): "In check_override_compatibility() in solver/subset.rs, add a short-circuit: when the overriding method has (*args: Any, **kwargs: Any), treat it as compatible with any parameter signature. This eliminates 8 bad-override errors across jax, bokeh, poetry, and artigraph."

CROSS-CHECK DATA: Each project's errors include cross-check stats showing how many
errors are also reported by mypy/pyright. Use this to prioritize:
- "pyrefly-only" errors (not in mypy or pyright) are the PRIMARY targets for fixes.
  These indicate pyrefly is enforcing a rule more strictly than the ecosystem standard.
- Errors co-reported by mypy/pyright are real bugs in the project code — do NOT
  suggest pyrefly changes for these.
- If ALL regression errors are co-reported by mypy/pyright, the regression is likely
  a real improvement being misclassified — say so instead of suggesting a fix.

Rules:
- Focus on pyrefly-only regressions. Co-reported errors are NOT pyrefly bugs.
- For intentional/conformance improvements: do NOT suggest reverting or weakening spec-compliant behavior. The ecosystem needs to adapt to the spec, not the other way around.
- If a rule is applied too broadly (e.g., a protocol-only check applied to regular classes), suggest narrowing its scope with a guard condition, naming the specific function.
- If type inference regressed, identify which inference path changed and suggest restoring the previous behavior for the affected cases.
- Reference specific pyrefly source files and function names from the diff.
- If no regressions exist, return empty suggestions.

Respond with JSON only:
{"summary": "one-sentence summary of the situation", "suggestions": [{"description": "In function_name() in path/to/file.rs, do X to fix Y", "files": ["path/to/file.rs"], "confidence": "high|medium|low", "reasoning": "Why this fixes the regressions + expected outcome: eliminates N errors across M projects", "affected_projects": ["project1", "project2"], "error_kinds_fixed": ["bad-override", "bad-assignment"]}]}

If there are no regressions, respond with:
{"summary": "No regressions detected.", "suggestions": []}"""


def _build_suggestion_user_prompt(
    result: "ClassificationResult",
    pyrefly_diff: str,
) -> str:
    """Build the user prompt for Pass 3: serialize classification results + diff.

    Uses forward reference to ClassificationResult to avoid circular imports.
    """
    parts: list[str] = []

    # Summary stats
    parts.append(
        f"Classification summary: {result.regressions} regression(s), "
        f"{result.improvements} improvement(s), {result.neutrals} neutral, "
        f"{result.ambiguous} ambiguous out of {result.total_projects} project(s).\n"
    )

    # Aggregate: regression error kinds and affected projects
    regression_kinds: set[str] = set()
    affected_projects: list[str] = []
    for c in result.classifications:
        if c.verdict in ("regression", "ambiguous"):
            affected_projects.append(c.project_name)
            for cat in c.categories:
                regression_kinds.add(cat.category)
    if regression_kinds:
        parts.append(f"Regression error kinds: {', '.join(sorted(regression_kinds))}")
    if affected_projects:
        parts.append(f"Affected projects: {', '.join(affected_projects)}")
    if regression_kinds or affected_projects:
        parts.append("")

    # Group by verdict, regressions first
    verdict_order = ["regression", "ambiguous", "improvement", "neutral"]
    for verdict in verdict_order:
        group = [c for c in result.classifications if c.verdict == verdict]
        if not group:
            continue
        parts.append(f"--- {verdict.upper()} ({len(group)}) ---")
        for c in group:
            parts.append(
                f"Project: {c.project_name} (+{c.added_count}/-{c.removed_count})"
            )
            parts.append(f"  Reason: {c.reason}")
            if c.pr_attribution and c.pr_attribution != "N/A":
                parts.append(f"  Attribution: {c.pr_attribution}")
            # Include cross-check stats for the project
            if c.cross_check_stats:
                parts.append(f"  Cross-check: {c.cross_check_stats}")
            if c.categories:
                for cat in c.categories:
                    parts.append(
                        f"  Category [{cat.verdict}] {cat.category}: {cat.reason}"
                    )
            parts.append("")

    # Pyrefly PR diff
    if pyrefly_diff:
        # Truncate diff if very large to stay within token limits
        max_diff = 10000
        if len(pyrefly_diff) > max_diff:
            diff_text = pyrefly_diff[:max_diff] + "\n[... diff truncated ...]"
        else:
            diff_text = pyrefly_diff
        parts.append(f"Pyrefly PR diff:\n{diff_text}")

    return "\n".join(parts)


def generate_suggestions(
    result: "ClassificationResult",
    pyrefly_diff: str,
    model: Optional[str] = None,
) -> "SuggestionResult":
    """Pass 3: Generate aggregate suggestions based on all classification results.

    Reads ALL per-project classifications and the pyrefly PR diff to produce
    actionable source code suggestions. Makes exactly one LLM call per PR.

    Skips the LLM call entirely if there are no regressions and no ambiguous
    results, returning an empty SuggestionResult.

    Uses forward references to avoid circular imports with classifier.py.
    """
    from .classifier import Suggestion, SuggestionResult

    # Skip LLM call if no regressions or ambiguous results
    has_regressions = result.regressions > 0
    if not has_regressions and result.ambiguous == 0:
        return SuggestionResult(
            suggestions=[],
            summary="No regressions detected.",
            has_regressions=False,
        )

    backend, api_key = _get_backend()
    if backend == "none":
        raise LLMError(
            "No API key found. Set LLAMA_API_KEY (Meta internal) "
            "or CLASSIFIER_API_KEY / ANTHROPIC_API_KEY."
        )

    system_prompt = _build_suggestion_system_prompt()
    user_prompt = _build_suggestion_user_prompt(result, pyrefly_diff)

    print(
        f"Using {backend} backend for suggestion generation (pass 3)",
        file=sys.stderr,
    )

    if backend == "llama":
        api_result = _call_llama_api(api_key, system_prompt, user_prompt, model)
    else:
        api_result = _call_anthropic_api(api_key, system_prompt, user_prompt, model)

    text = _extract_text_from_response(backend, api_result)
    parsed = _parse_classification(text)

    suggestions = []
    for s in parsed.get("suggestions", []):
        suggestions.append(
            Suggestion(
                description=s.get("description", ""),
                files=s.get("files", []),
                confidence=s.get("confidence", "low"),
                reasoning=s.get("reasoning", ""),
                affected_projects=s.get("affected_projects", []),
                error_kinds_fixed=s.get("error_kinds_fixed", []),
            )
        )

    return SuggestionResult(
        suggestions=suggestions,
        summary=parsed.get("summary", ""),
        has_regressions=has_regressions,
        raw_response=api_result,
    )
