# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""LLM client for classifying primer diff entries.

Supports two backends:
1. Meta's Llama API (OpenAI-compatible endpoint at api.llama.com)
   - Set LLAMA_API_KEY to your Llama API key
   - Get a key at https://www.internalfb.com/metagen/tools/llm-api-keys
2. Anthropic Claude API
   - Set CLASSIFIER_API_KEY or ANTHROPIC_API_KEY

Llama API is preferred when LLAMA_API_KEY is set. Falls back to Anthropic.
No pip dependencies — uses only urllib.
"""

from __future__ import annotations

import json
import os
import re
import ssl
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Optional

MAX_RETRIES = 4
RETRY_BASE_DELAY = 2.0  # seconds, doubles each retry


def _get_ssl_context() -> ssl.SSLContext:
    """Get an SSL context that works on macOS (certifi or system certs)."""
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        pass
    # Fallback: try system default, then unverified as last resort
    ctx = ssl.create_default_context()
    try:
        # Test that the default context can verify
        urllib.request.urlopen("https://api.llama.com", timeout=5, context=ctx)
    except urllib.error.URLError:
        # macOS Python without certs installed — use unverified context
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
    except Exception:
        pass
    return ctx

# Llama API (native format)
LLAMA_API_URL = "https://api.llama.com/v1/chat/completions"
LLAMA_DEFAULT_MODEL = "Llama-4-Maverick-17B-128E-Instruct-FP8"

# Anthropic API
ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_DEFAULT_MODEL = "claude-sonnet-4-20250514"
ANTHROPIC_API_VERSION = "2023-06-01"


@dataclass
class LLMResponse:
    """Response from the LLM classification call."""

    verdict: str  # "regression", "improvement", "neutral"
    reason: str  # human-readable explanation
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

Classify as one of:
- "improvement": Pyrefly got better. This means:
  - New errors ('+') that correctly catch REAL bugs in the code (true positives — pyrefly is now smarter)
  - Removed errors ('-') that were wrong (false positives removed — pyrefly is now less noisy)
- "regression": Pyrefly got worse. This means:
  - New errors ('+') that are WRONG — the code is actually correct and pyrefly is flagging it incorrectly (false positives introduced)
  - Removed errors ('-') where the code actually had a bug that pyrefly used to catch but no longer does
- "neutral": Message wording changes with no behavioral impact

KEY RULE: If a new error ('+') correctly identifies a real bug in the source code, that is an IMPROVEMENT — pyrefly is catching something it should catch. Even if the code is buggy, pyrefly finding it is a good thing.

IMPORTANT: When source code is provided, you MUST analyze the actual code to support your verdict. Do NOT just rephrase the error message. Describe the specific bug or explain why the code is actually correct by referencing what the source code does.

If a typing spec section is relevant, include a link (https://typing.readthedocs.io/en/latest/spec/...). If the error contradicts what mypy/pyright accept, say so.

Respond with ONLY a JSON object (no markdown fences):
{"verdict": "regression|improvement|neutral", "reason": "explanation"}

In the reason field, separate the explanation for each error with \\n\\n (escaped newlines) so they render as distinct paragraphs."""


def _build_user_prompt(
    errors_text: str,
    source_context: Optional[str],
    change_type: str,
) -> str:
    parts = [
        f"Change type: {change_type} ('+' = new error on PR branch, '-' = removed error)\n",
        f"Errors:\n{errors_text}\n",
    ]
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
        "max_completion_tokens": 1024,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }

    data = json.dumps(payload).encode("utf-8")
    ctx = _get_ssl_context()
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
        "max_tokens": 1024,
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
        ctx = _get_ssl_context()
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
    """Parse the JSON classification from the LLM response text."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to extract JSON from markdown fences or surrounding text
    m = re.search(r"\{[^}]+\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass

    raise LLMError(f"Could not parse LLM response as JSON: {text}")


def classify_with_llm(
    errors_text: str,
    source_context: Optional[str] = None,
    change_type: str = "mixed",
    model: Optional[str] = None,
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
    user_prompt = _build_user_prompt(errors_text, source_context, change_type)

    print(f"Using {backend} backend for classification", file=sys.stderr)

    if backend == "llama":
        result = _call_llama_api(api_key, system_prompt, user_prompt, model)
    else:
        result = _call_anthropic_api(api_key, system_prompt, user_prompt, model)

    text = _extract_text_from_response(backend, result)
    classification = _parse_classification(text)

    verdict = classification.get("verdict", "").lower().strip()
    reason = classification.get("reason", "No reason provided")

    if verdict not in ("regression", "improvement", "neutral"):
        print(
            f"Warning: LLM returned unexpected verdict '{verdict}', treating as ambiguous",
            file=sys.stderr,
        )
        verdict = "neutral"
        reason = f"[Ambiguous LLM verdict: '{classification.get('verdict')}']. {reason}"

    return LLMResponse(verdict=verdict, reason=reason, raw_response=result)


