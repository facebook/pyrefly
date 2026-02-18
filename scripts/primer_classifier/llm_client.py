# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""LLM client for classifying primer diff entries.

Supports two backends:
1. Meta's Llama API (native format at api.llama.com)
   - Set LLAMA_API_KEY to your Llama API key
2. Anthropic Claude API
   - Set CLASSIFIER_API_KEY or ANTHROPIC_API_KEY

Llama API is preferred when LLAMA_API_KEY is set. Falls back to Anthropic.
No pip dependencies — uses only urllib.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Optional

from .ssl_utils import get_ssl_context

MAX_RETRIES = 4
RETRY_BASE_DELAY = 2.0  # seconds, doubles each retry

# Llama API (native format)
LLAMA_API_URL = "https://api.llama.com/v1/chat/completions"
LLAMA_DEFAULT_MODEL = "Llama-4-Maverick-17B-128E-Instruct-FP8"

# Anthropic API
ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_DEFAULT_MODEL = "claude-sonnet-4-20250514"
ANTHROPIC_API_VERSION = "2023-06-01"


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
    needs_files: list[str] = field(default_factory=list)  # file paths the LLM wants to see
    raw_response: Optional[dict] = None


class LLMError(Exception):
    """Raised when the LLM API call fails."""

    pass


def _get_backend() -> tuple[str, str]:
    """Determine which backend to use and return (backend_name, api_key).

    Priority: LLAMA_API_KEY > CLASSIFIER_API_KEY > ANTHROPIC_API_KEY
    """
    llama_key = os.environ.get("LLAMA_API_KEY")
    if llama_key:
        return "llama", llama_key

    anthropic_key = os.environ.get("CLASSIFIER_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
    if anthropic_key:
        return "anthropic", anthropic_key

    return "none", ""



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

KEY RULES:
1. If a new error ('+') correctly identifies a real bug in the source code, that is an IMPROVEMENT — pyrefly is catching something it should catch. Even if the code is buggy, pyrefly finding it is a good thing.
2. BEFORE choosing your verdict, re-read your own analysis. If your explanation describes a genuine type error, inconsistency, or bug in the code, the verdict MUST be "improvement" — not "regression". Only choose "regression" if the code is actually correct and pyrefly is wrong.
3. A "bad-override" where the child class truly has an inconsistent type signature vs the parent is a REAL bug — that is an improvement, not a regression.
4. MISSING-ATTRIBUTE PATTERN: When you see many `missing-attribute` errors across a well-known, well-tested project (e.g., mypy, discord.py, xarray, dulwich), and the errors claim attributes like `data`, `dims`, `fullname`, `parents`, etc. are missing from core classes, this is almost always a REGRESSION. These are fundamental attributes that the project uses extensively — they are defined somewhere (often via `__slots__` in parent classes, descriptors, or dynamic assignment). The type checker is failing to resolve them through the class hierarchy. Be especially skeptical when:
   - The same class has many "missing" attributes (suggests the checker can't see the class's attribute definitions)
   - The attributes are basic/fundamental (e.g., `data`, `name`, `tree`, `parents` on a `Commit` class)
   - The project is mature and well-tested (unlikely to have 50+ real bugs in core attribute access)
5. NET DIRECTION MATTERS — THIS TAKES PRIORITY OVER OTHER RULES: When there are BOTH added (+) AND removed (-) errors, consider the overall direction:
   - If removals significantly outnumber additions (3:1 ratio or higher, e.g., 29 removed, 5 added), the overall verdict should be "improvement". The PR fixed many problems and the few new errors are minor side effects. Even if some new errors contain Never/Unknown types, the net impact is clearly positive.
   - If additions significantly outnumber removals, lean toward "regression" — the PR introduced more problems than it fixed.
   - Only classify a net-removal project as "regression" if removals are FEWER than additions AND the newly added errors are clearly wrong.
   - When old and new errors share the same file:line:kind but differ only in inferred types (e.g., `Unknown` -> `Any`, or `Value | Unknown | None` -> `Value | None`), that is a WORDING CHANGE — do NOT count it as a new error. Subtract these from the added count when assessing net direction.
6. TYPE CHECKER LIMITATIONS: Be aware that type checkers have known blind spots. When errors arise from patterns the type checker cannot model (e.g., runtime-injected values, dynamic attribute assignment, C extension modules, conditional imports), those are false positives (regressions), not real bugs being caught. Similarly, when removing one concise error creates many scattered downstream errors for the same root cause, that is worse error reporting quality (regression), not an improvement.
7. NEVER/@_ INFERENCE FAILURES — THIS IS CRITICAL: When error messages contain `Never` or `@_` types, this almost always indicates a type inference failure rather than a real bug. These are REGRESSIONS. Specific patterns:
   - `Expected class object, got 'Never'` → inference failure, REGRESSION
   - `list[Never]`, `Generator[Never, ...]`, `dict[str, Never]` → inference failure, REGRESSION
   - Arguments showing `Never` where a concrete type should be → inference failure, REGRESSION
   - Types changing FROM concrete types TO `Never`/`@_` → type resolution got WORSE, REGRESSION
   - `list[Unknown] | list[Never]` appearing where `list[SomeConcreteType]` is expected → REGRESSION
   NOTE: `Unknown` in error messages is different from `Never`. `Unknown` often means the code is untyped and pyrefly is correctly reporting that untyped values don't match typed parameters. An error like "Argument `dict[Unknown, Unknown] | Unknown` is not assignable to parameter X" may be a genuine type issue (the untyped value truly isn't compatible). Evaluate `Unknown` errors on their merits — they can be real bugs being caught.
8. TEST FILE ERRORS: Errors in test files (paths containing `/tests/`, `/test_`, `_test.py`, `conftest.py`) need case-by-case evaluation. Test code CAN have real type issues:
   - Calling `.equals()` on an `int` → real bug (method doesn't exist)
   - `missing-attribute` on a concrete class (not a mock) → likely real bug
   - Wrong return type from a fixture → real type issue
   But some test patterns produce false positives:
   - `not-callable` on values from fixtures/parametrize → often false positive
   - `missing-attribute` on mock objects → false positive
   - Errors from complex parametrize type inference → often false positive
   Evaluate each error individually rather than dismissing all test file errors.
9. TYPE STUB PROJECTS: Projects whose names end in `-stubs` (e.g., `django-stubs`, `boto3-stubs`) are type stub packages. They provide type annotations for other libraries and are extensively tested against mypy/pyright. New errors from pyrefly on these projects are almost always REGRESSIONS — they indicate pyrefly disagrees with established type checker behavior, not that the stubs are wrong.
10. WELL-KNOWN EXEMPT PATTERNS: Some code patterns are technically type violations but are universally accepted by the Python ecosystem. New errors flagging these patterns are REGRESSIONS because they create noise with no practical value:
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

CRITICAL — MYPY/PYRIGHT CROSS-CHECK: Before classifying a new error as "improvement", ask yourself: would mypy and pyright also flag this exact pattern? If they would NOT — if pyrefly is enforcing a rule more strictly than the ecosystem standard — then the error is a REGRESSION regardless of whether the code has a theoretical type issue. Pyrefly being stricter than established tools creates false positive noise. Use your training data knowledge of mypy/pyright behavior to make this determination.

OUTPUT FORMAT:

When errors are grouped into categories, provide a verdict for EACH category separately, plus an overall verdict. Different categories within the same project may have different verdicts (e.g., one category is a regression, another is an improvement).

REQUESTING ADDITIONAL FILES:

If you cannot confidently determine the verdict because you need to see source code from another file (e.g., a parent class definition, a module that defines __slots__, a base class with the overridden method), you may request those files instead of guessing. Respond with:
{"needs_files": ["path/to/parent_class.py", "path/to/base.py"]}

Use the project's file paths (e.g., "dulwich/objects.py", "discord/permissions.py"). Request at most 3 files. Only request files when you genuinely need them to verify the verdict — if the pattern is clear enough (e.g., 100+ missing-attribute errors on core classes of a well-tested project), classify directly without requesting files.

If you have enough context to classify, respond with the verdict:
{"spec_check": "which typing spec rule applies, with a link, AND what is its scope — which constructs does it apply to?", "runtime_behavior": "would this cause an actual runtime error or crash? trace the execution path", "mypy_pyright": "would mypy/pyright flag this? yes/no and why", "verdict": "regression|improvement|neutral", "reason": "explanation — if a typing spec rule is relevant, include the link (e.g., https://typing.readthedocs.io/en/latest/spec/...) so reviewers can verify", "categories": [{"category": "short label", "verdict": "regression|improvement|neutral", "reason": "explanation"}, ...]}

The "spec_check", "runtime_behavior", and "mypy_pyright" fields force you to think through the evidence before concluding. If the spec doesn't require the check here AND there's no runtime impact AND mypy/pyright wouldn't flag it, it's almost certainly "too strict" (regression). The "categories" field is optional — omit it for small diffs with no categories. When present, each entry should correspond to an error category from the input. The top-level "verdict" should reflect the overall assessment (if all categories agree, use that; if mixed, use the dominant direction or "regression" if any regressions are present).

In the reason fields, keep explanations concise. When your verdict depends on a typing spec rule, you MUST include the relevant spec link in the reason field (e.g., "per the [typing spec](https://typing.readthedocs.io/en/latest/spec/generics.html#variance), variance inference is only required for protocols")."""


def _build_user_prompt(
    errors_text: str,
    source_context: Optional[str],
    change_type: str,
    structural_signals: Optional[str] = None,
) -> str:
    parts = [
        f"Change type: {change_type} ('+' = new error on PR branch, '-' = removed error)\n",
        f"Errors:\n{errors_text}\n",
    ]
    if structural_signals:
        parts.append(f"\n{structural_signals}\n")
    if source_context:
        parts.append(f"Source code at error location (line marked with >>>):\n{source_context}\n")
    else:
        parts.append("Source code: not available (could not fetch from GitHub)\n")

    return "\n".join(parts)


def _call_llama_api(
    api_key: str,
    system_prompt: str,
    user_prompt: str,
    model: Optional[str],
) -> dict:
    """Call Meta's Llama API with retry on rate limiting."""
    payload = {
        "model": model or LLAMA_DEFAULT_MODEL,
        "temperature": 0,
        "max_completion_tokens": 2048,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }

    data = json.dumps(payload).encode("utf-8")
    ctx = get_ssl_context()
    last_error: Optional[Exception] = None

    for attempt in range(MAX_RETRIES + 1):
        req = urllib.request.Request(
            LLAMA_API_URL,
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=60, context=ctx) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace") if e.fp else ""
            if e.code == 429 and attempt < MAX_RETRIES:
                delay = RETRY_BASE_DELAY * (2 ** attempt)
                print(f"  Rate limited, retrying in {delay:.0f}s (attempt {attempt + 1}/{MAX_RETRIES})...", file=sys.stderr)
                time.sleep(delay)
                last_error = LLMError(f"Llama API returned {e.code}: {body}")
                continue
            raise LLMError(f"Llama API returned {e.code}: {body}") from e
        except urllib.error.URLError as e:
            raise LLMError(f"Llama API network error: {e.reason}") from e

    raise last_error or LLMError("Llama API failed after retries")


def _call_anthropic_api(
    api_key: str,
    system_prompt: str,
    user_prompt: str,
    model: Optional[str],
) -> dict:
    """Call the Anthropic Messages API."""
    payload = {
        "model": model or ANTHROPIC_DEFAULT_MODEL,
        "temperature": 0,
        "max_tokens": 2048,
        "system": system_prompt,
        "messages": [
            {"role": "user", "content": user_prompt},
        ],
    }

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        ANTHROPIC_API_URL,
        data=data,
        headers={
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": ANTHROPIC_API_VERSION,
        },
        method="POST",
    )

    try:
        ctx = get_ssl_context()
        with urllib.request.urlopen(req, timeout=60, context=ctx) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace") if e.fp else ""
        raise LLMError(f"Anthropic API returned {e.code}: {body}") from e
    except urllib.error.URLError as e:
        raise LLMError(f"Anthropic API network error: {e.reason}") from e


def _extract_text_from_response(backend: str, result: dict) -> str:
    """Extract the generated text from the API response."""
    try:
        if backend == "llama":
            # Llama native format: completion_message.content.text
            content = result["completion_message"]["content"]
            if isinstance(content, dict):
                return content["text"]
            return str(content)
        else:
            # Anthropic format: content[0].text
            return result["content"][0]["text"]
    except (KeyError, IndexError) as e:
        raise LLMError(f"Unexpected {backend} API response structure: {result}") from e


def _parse_classification(text: str) -> dict:
    """Parse the JSON classification from the LLM response text.

    Handles cases where the LLM wraps JSON in markdown fences or
    surrounds it with analysis text.
    """
    # Try the full text first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Strip markdown code fences
    stripped = re.sub(r"```(?:json)?\s*", "", text).strip()
    try:
        return json.loads(stripped)
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
                            if isinstance(parsed, dict) and (
                                "verdict" in parsed or "needs_files" in parsed
                            ):
                                return parsed
                        except json.JSONDecodeError:
                            pass
                        break

    raise LLMError(f"Could not parse LLM response as JSON: {text}")


def classify_with_llm(
    errors_text: str,
    source_context: Optional[str] = None,
    change_type: str = "mixed",
    model: Optional[str] = None,
    structural_signals: Optional[str] = None,
) -> LLMResponse:
    """Send errors + context to the LLM for classification.

    Uses Llama API if LLAMA_API_KEY is set, otherwise Anthropic.
    Raises LLMError if the API call fails or the response is unparseable.
    """
    backend, api_key = _get_backend()
    if backend == "none":
        raise LLMError(
            "No API key found. Set LLAMA_API_KEY (Meta internal) "
            "or CLASSIFIER_API_KEY / ANTHROPIC_API_KEY."
        )

    system_prompt = _build_system_prompt()
    user_prompt = _build_user_prompt(errors_text, source_context, change_type, structural_signals)

    print(f"Using {backend} backend for classification", file=sys.stderr)

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

    verdict = classification.get("verdict", "").lower().strip()
    reason = classification.get("reason", "No reason provided")

    if verdict not in ("regression", "improvement", "neutral"):
        print(
            f"Warning: LLM returned unexpected verdict '{verdict}', treating as ambiguous",
            file=sys.stderr,
        )
        verdict = "neutral"
        reason = f"[Ambiguous LLM verdict: '{classification.get('verdict')}']. {reason}"

    # Parse per-category verdicts if present
    categories: list[CategoryVerdict] = []
    for cat_data in classification.get("categories", []):
        cat_verdict = cat_data.get("verdict", "").lower().strip()
        if cat_verdict not in ("regression", "improvement", "neutral"):
            cat_verdict = "neutral"
        categories.append(CategoryVerdict(
            category=cat_data.get("category", "unknown"),
            verdict=cat_verdict,
            reason=cat_data.get("reason", ""),
        ))

    return LLMResponse(verdict=verdict, reason=reason, categories=categories, raw_response=result)


