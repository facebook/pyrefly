# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Shared SSL context for HTTPS requests.

macOS system Python often lacks certificate bundles. This module provides
a lazily-initialized SSL context that tries (in order):
1. certifi's CA bundle (if installed)
2. System default certificates
3. Unverified context (with a warning) as a last resort
"""

from __future__ import annotations

import ssl
import sys
from typing import Optional


def _build_ssl_context() -> ssl.SSLContext:
    """Build an SSL context, trying certifi first, then system defaults."""
    # 1. Try certifi
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        pass

    # 2. Try system default
    ctx = ssl.create_default_context()
    # On macOS with stock Python the default store may be empty.
    # We detect that by checking whether any CA certs are loaded.
    stats = ctx.cert_store_stats()
    if stats["x509_ca"] > 0:
        return ctx

    # 3. Fall back to unverified (CI runners, sandboxed envs)
    print(
        "Warning: No CA certificates found. SSL verification disabled. "
        "Install certifi (`pip install certifi`) to fix this.",
        file=sys.stderr,
    )
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


_cached_ctx: Optional[ssl.SSLContext] = None


def get_ssl_context() -> ssl.SSLContext:
    """Return a lazily-initialized, cached SSL context."""
    global _cached_ctx
    if _cached_ctx is None:
        _cached_ctx = _build_ssl_context()
    return _cached_ctx
