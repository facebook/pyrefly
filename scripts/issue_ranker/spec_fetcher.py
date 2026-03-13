#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Fetch relevant typing spec sections on demand for LLM context.

Maps error kinds to spec URLs and fetches excerpts via urllib.
"""

from __future__ import annotations

import logging
import os
import re
import sys
import urllib.error
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from primer_classifier.ssl_utils import get_ssl_context

# Map of error kinds to relevant typing spec pages.
_SPEC_URLS: dict[str, str] = {
    "bad-override": "https://typing.readthedocs.io/en/latest/spec/class-compat.html",
    "bad-return": "https://typing.readthedocs.io/en/latest/spec/callables.html",
    "bad-assignment": "https://typing.readthedocs.io/en/latest/spec/generics.html",
    "missing-attribute": "https://typing.readthedocs.io/en/latest/spec/protocol.html",
    "not-callable": "https://typing.readthedocs.io/en/latest/spec/callables.html",
    "bad-argument": "https://typing.readthedocs.io/en/latest/spec/callables.html",
    "incompatible-override": "https://typing.readthedocs.io/en/latest/spec/class-compat.html",
    "variance": "https://typing.readthedocs.io/en/latest/spec/generics.html",
    "type-var": "https://typing.readthedocs.io/en/latest/spec/generics.html",
    "protocol": "https://typing.readthedocs.io/en/latest/spec/protocol.html",
    "overload": "https://typing.readthedocs.io/en/latest/spec/overload.html",
    "narrowing": "https://typing.readthedocs.io/en/latest/spec/narrowing.html",
}

_cache: dict[str, str] = {}


def get_spec_excerpt(error_kind: str) -> str:
    """Fetch a typing spec excerpt relevant to the given error kind.

    Returns an excerpt string, or empty string if no spec is mapped or
    fetch fails. Results are cached in-memory.
    """
    url = _SPEC_URLS.get(error_kind, "")
    if not url:
        return ""

    if url in _cache:
        return _cache[url]

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "pyrefly-ranker/1.0"})
        ctx = get_ssl_context()
        with urllib.request.urlopen(req, timeout=15, context=ctx) as resp:
            html = resp.read().decode("utf-8", errors="replace")
            # Extract a manageable excerpt — strip HTML tags naively
            text = re.sub(r"<[^>]+>", " ", html)
            text = re.sub(r"\s+", " ", text).strip()
            # Take first ~2000 chars as context
            excerpt = text[:2000]
            _cache[url] = excerpt
            return excerpt
    except (urllib.error.URLError, urllib.error.HTTPError, OSError) as e:
        logging.debug(f"Failed to fetch spec for {error_kind}: {e}")
        _cache[url] = ""
        return ""
