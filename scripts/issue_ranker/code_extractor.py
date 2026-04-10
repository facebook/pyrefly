#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Extract Python code blocks from GitHub issue markdown bodies.

Tries three strategies in order:
1. Explicitly tagged ```python / ```py fences
2. Bare ``` fences that look like Python code
3. LLM-based extraction from the issue body text
"""

from __future__ import annotations

import logging
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from llm_transport import call_llm, LLMError

# Matches ```python or ```py fenced code blocks (optionally indented).
_PYTHON_FENCE_RE = re.compile(
    r"^[ \t]*```(?:python|py)\s*\n(.*?)^[ \t]*```",
    re.MULTILINE | re.DOTALL,
)

# Matches bare ``` fenced code blocks (no language tag).
_BARE_FENCE_RE = re.compile(
    r"^[ \t]*```\s*\n(.*?)^[ \t]*```",
    re.MULTILINE | re.DOTALL,
)

# Heuristic: does this look like Python code?
# Uses strong signals (def/class/import at start of line) to avoid
# matching error messages or config text in bare code fences.
_PYTHON_STRONG_RE = re.compile(
    r"^(?:def |class |import |from \w+ import )",
    re.MULTILINE,
)

HAIKU_MODEL = "claude-haiku-4-5-20251001"

# Regex to extract JSON from a ```json fence
_JSON_FENCE_RE = re.compile(
    r"```(?:json)?\s*\n(.*?)```",
    re.DOTALL,
)


def _parse_code_response(text: str) -> str:
    """Parse code from an LLM response that may be JSON or a code fence.

    Handles:
    - Clean JSON: {"code": "..."}
    - JSON in a fence: ```json {"code": "..."} ```
    - Bare code in a fence: ```python x = 1 ```
    - Plain text code
    """
    import json as json_mod

    # Try direct JSON parse
    try:
        parsed = json_mod.loads(text.strip())
        code = parsed.get("code", "")
        if code and code != "NO_CODE":
            return code
        return ""
    except (json_mod.JSONDecodeError, AttributeError):
        logging.debug(f"  _parse_code_response: direct JSON parse failed: {text[:80]}")

    # Try extracting JSON from a code fence
    json_match = _JSON_FENCE_RE.search(text)
    if json_match:
        try:
            parsed = json_mod.loads(json_match.group(1).strip())
            code = parsed.get("code", "")
            if code and code != "NO_CODE":
                return code
            return ""
        except (json_mod.JSONDecodeError, AttributeError):
            logging.debug(
                f"  _parse_code_response: fenced JSON parse failed: {text[:80]}"
            )

    # Try extracting code from a python/bare fence
    py_match = _PYTHON_FENCE_RE.search(text)
    if py_match:
        return py_match.group(1).strip()

    bare_match = _BARE_FENCE_RE.search(text)
    if bare_match:
        code = bare_match.group(1).strip()
        if _looks_like_python(code):
            return code

    logging.debug(f"  _parse_code_response: no code found in: {text[:120]}")
    return ""


_EXTRACT_PROMPT = """You are extracting a Python code snippet from a GitHub issue about a Python type checker.

IMPORTANT: Only extract code for type-checking issues (type errors, inference bugs, false positives/negatives). Do NOT try to create code for:
- Performance issues (memory, speed, etc.)
- IDE/language server features (hover, completions, go-to-def)
- Configuration or build issues
- Documentation issues

Read the issue text below and produce a minimal, self-contained Python code snippet that reproduces the type checking bug or demonstrates the type-related feature request. The snippet should:
- Be valid Python that can be type-checked
- Include type annotations where relevant
- Be as minimal as possible while still demonstrating the issue
- Include necessary imports (from typing, etc.)
- NOT include pip install commands, shell commands, or non-Python code
- If the code references a third-party library, include the import but don't worry about installing it

If the issue cannot be demonstrated with a Python code snippet, respond with exactly: NO_CODE

Respond with JSON:
{"code": "the python code here"}
or
{"code": "NO_CODE"}"""

_REPAIR_PROMPT = """You are fixing a Python code snippet extracted from a GitHub issue about a Python type checker.

The snippet may have problems that prevent it from being type-checked:
- Missing imports (add the necessary import statements)
- Syntax errors from bad copy-paste (fix them)
- Incomplete code (add minimal stubs to make it parseable)
- References to undefined names (add stub definitions)

You will also receive the issue title and body for context about what the code is supposed to demonstrate.

Your goal is to produce a self-contained snippet that a type checker can analyze. Keep changes minimal — only fix what's broken, don't rewrite the logic.

If the code is already valid or cannot be meaningfully repaired, return it unchanged.

Respond with JSON:
{"code": "the fixed python code here"}"""


def _looks_like_python(code: str) -> bool:
    """Heuristic check: does this code block look like Python?

    Uses strong signals (def/class/import at start of line) to avoid
    false positives on error messages or config text.
    """
    return bool(_PYTHON_STRONG_RE.search(code))


def extract_code_blocks(body: str, use_llm: bool = False) -> list[str]:
    """Extract Python code snippets from a markdown issue body.

    Strategy:
    1. Look for ```python / ```py fenced blocks
    2. Fall back to bare ``` blocks that look like Python
    3. If use_llm=True and no blocks found, ask an LLM to generate a repro

    Returns a list of code strings (may be empty).
    """
    # Strategy 1: Explicitly tagged Python fences
    blocks = []
    for match in _PYTHON_FENCE_RE.finditer(body):
        code = match.group(1).strip()
        if code:
            blocks.append(code)

    if blocks:
        return blocks

    # Strategy 2: Bare ``` fences that look like Python
    for match in _BARE_FENCE_RE.finditer(body):
        code = match.group(1).strip()
        if code and _looks_like_python(code):
            blocks.append(code)

    if blocks:
        return blocks

    # Strategy 3: LLM-based extraction
    if use_llm and body.strip():
        llm_code = _llm_extract_code(body)
        if llm_code:
            return [llm_code]

    return []


def _llm_extract_code(body: str) -> str | None:
    """Use an LLM to extract/generate a Python repro snippet from issue text."""
    try:
        user_prompt = f"Issue text:\n{body[:2000]}"
        text = call_llm(_EXTRACT_PROMPT, user_prompt, model=HAIKU_MODEL)
        code = _parse_code_response(text)
        return code if code else None
    except LLMError as e:
        logging.debug(f"LLM code extraction failed: {e}")

    return None


def repair_snippet(code: str, issue_title: str = "", issue_body: str = "") -> str:
    """Use an LLM to fix a broken code snippet (missing imports, syntax errors).

    Receives the issue title and body for context so the LLM understands
    what the code is supposed to demonstrate.

    Returns the repaired code, or the original if repair fails or isn't needed.
    """
    try:
        user_prompt = (
            f"Issue title: {issue_title}\n"
            f"Issue body:\n{issue_body[:1000]}\n\n"
            f"Code snippet to fix:\n```python\n{code}\n```"
        )
        text = call_llm(_REPAIR_PROMPT, user_prompt, model=HAIKU_MODEL)
        repaired = _parse_code_response(text)

        if repaired:
            return repaired
    except LLMError as e:
        logging.debug(f"LLM snippet repair failed: {e}")

    return code
