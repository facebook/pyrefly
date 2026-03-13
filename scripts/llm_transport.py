# @nolint
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Shared LLM transport layer for calling language model APIs.

Provides backend detection, API dispatch, text extraction, and JSON
parsing.  Used by both primer_classifier and issue_ranker so that neither
package depends on the other's internals.

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
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from primer_classifier.ssl_utils import get_ssl_context

# ── retry / rate-limit constants ─────────────────────────────────────
MAX_RETRIES = 4
RETRY_BASE_DELAY = 2.0  # seconds, doubles each retry

# ── Llama API (native format) ────────────────────────────────────────
LLAMA_API_URL = "https://api.llama.com/v1/chat/completions"
LLAMA_DEFAULT_MODEL = "Llama-4-Maverick-17B-128E-Instruct-FP8"

# ── Anthropic API ────────────────────────────────────────────────────
ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_DEFAULT_MODEL = "claude-sonnet-4-20250514"
ANTHROPIC_API_VERSION = "2023-06-01"


class LLMError(Exception):
    """Raised when the LLM API call fails."""

    pass


# ── backend detection ────────────────────────────────────────────────


def get_backend() -> tuple[str, str]:
    """Determine which backend to use and return (backend_name, api_key).

    Priority: LLAMA_API_KEY > CLASSIFIER_API_KEY > ANTHROPIC_API_KEY
    """
    llama_key = os.environ.get("LLAMA_API_KEY")
    if llama_key:
        return "llama", llama_key

    anthropic_key = os.environ.get("CLASSIFIER_API_KEY") or os.environ.get(
        "ANTHROPIC_API_KEY"
    )
    if anthropic_key:
        return "anthropic", anthropic_key

    return "none", ""


# ── low-level API calls ──────────────────────────────────────────────


def call_llama_api(
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
                delay = RETRY_BASE_DELAY * (2**attempt)
                print(
                    f"  Rate limited, retrying in {delay:.0f}s "
                    f"(attempt {attempt + 1}/{MAX_RETRIES})...",
                    file=sys.stderr,
                )
                time.sleep(delay)
                last_error = LLMError(f"Llama API returned {e.code}: {body}")
                continue
            raise LLMError(f"Llama API returned {e.code}: {body}") from e
        except urllib.error.URLError as e:
            raise LLMError(f"Llama API network error: {e.reason}") from e

    raise last_error or LLMError("Llama API failed after retries")


def call_anthropic_api(
    api_key: str,
    system_prompt: str,
    user_prompt: str,
    model: Optional[str],
    max_tokens: int = 2048,
    timeout: int = 120,
) -> dict:
    """Call the Anthropic Messages API with retry on transient errors."""
    payload = {
        "model": model or ANTHROPIC_DEFAULT_MODEL,
        "temperature": 0,
        "max_tokens": max_tokens,
        "system": system_prompt,
        "messages": [
            {"role": "user", "content": user_prompt},
        ],
    }

    data = json.dumps(payload).encode("utf-8")
    ctx = get_ssl_context()
    last_error: Optional[Exception] = None

    for attempt in range(MAX_RETRIES + 1):
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
            with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace") if e.fp else ""
            if e.code in (429, 529, 500, 502, 503) and attempt < MAX_RETRIES:
                delay = RETRY_BASE_DELAY * (2**attempt)
                print(
                    f"  Anthropic API {e.code}, retrying in {delay:.0f}s "
                    f"(attempt {attempt + 1}/{MAX_RETRIES})...",
                    file=sys.stderr,
                )
                time.sleep(delay)
                last_error = LLMError(f"Anthropic API returned {e.code}: {body}")
                continue
            raise LLMError(f"Anthropic API returned {e.code}: {body}") from e
        except urllib.error.URLError as e:
            if attempt < MAX_RETRIES:
                delay = RETRY_BASE_DELAY * (2**attempt)
                print(
                    f"  Anthropic API network error, retrying in {delay:.0f}s "
                    f"(attempt {attempt + 1}/{MAX_RETRIES})...",
                    file=sys.stderr,
                )
                time.sleep(delay)
                last_error = LLMError(f"Anthropic API network error: {e.reason}")
                continue
            raise LLMError(f"Anthropic API network error: {e.reason}") from e

    raise last_error or LLMError("Anthropic API failed after retries")


# ── response parsing ─────────────────────────────────────────────────


def extract_text(backend: str, result: dict) -> str:
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


def parse_json(text: str) -> dict:
    """Parse a JSON dict from LLM response text.

    Handles cases where the LLM wraps JSON in markdown fences or
    surrounds it with analysis text.  Returns the first valid JSON
    dict found in the text.
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
                            if isinstance(parsed, dict):
                                return parsed
                        except json.JSONDecodeError:
                            pass
                        break

    raise LLMError(f"Could not parse LLM response as JSON: {text}")


# ── convenience wrappers ─────────────────────────────────────────────


def call_llm(
    system_prompt: str,
    user_prompt: str,
    *,
    model: str | None = None,
    max_tokens: int = 2048,
    timeout: int = 120,
) -> str:
    """Call the LLM and return the extracted text response.

    Handles backend detection, API dispatch, and text extraction.
    The *model* parameter is only used for the Anthropic backend;
    Llama always uses its default model.
    Raises LLMError if no API key or the call fails.
    """
    backend, api_key = get_backend()
    if backend == "none":
        raise LLMError(
            "No API key found. Set LLAMA_API_KEY (Meta internal) "
            "or CLASSIFIER_API_KEY / ANTHROPIC_API_KEY."
        )

    if backend == "llama":
        result = call_llama_api(api_key, system_prompt, user_prompt, None)
    else:
        result = call_anthropic_api(
            api_key,
            system_prompt,
            user_prompt,
            model,
            max_tokens=max_tokens,
            timeout=timeout,
        )

    return extract_text(backend, result)


def call_llm_json(
    system_prompt: str,
    user_prompt: str,
    *,
    model: str | None = None,
    max_tokens: int = 2048,
    timeout: int = 120,
) -> dict:
    """Call the LLM and return the parsed JSON response.

    Handles backend detection, API dispatch, text extraction, and
    JSON parsing.
    Raises LLMError if no API key, the call fails, or JSON parsing
    fails.
    """
    text = call_llm(
        system_prompt,
        user_prompt,
        model=model,
        max_tokens=max_tokens,
        timeout=timeout,
    )
    return parse_json(text)
