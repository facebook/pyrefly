# @nolint
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Shared SSL context for HTTPS requests.

macOS system Python often lacks certificate bundles. This module provides
a lazily-initialized SSL context that tries (in order):
1. System default certificates
2. certifi's CA bundle (if installed)

Raises RuntimeError if no valid CA certificates can be found.
"""

from __future__ import annotations

import ssl
from typing import Optional


def _build_ssl_context() -> ssl.SSLContext:
    """Build a secure SSL context. Fails if verification cannot be established."""
    # 1. Try system default
    ctx = ssl.create_default_context()
    if ctx.cert_store_stats()["x509_ca"] > 0:
        return ctx

    # 2. If empty (common on macOS), try certifi
    try:
        import certifi

        ctx.load_verify_locations(cafile=certifi.where())
        return ctx
    except (ImportError, FileNotFoundError):
        pass

    # 3. Fail hard — never silently disable TLS verification
    raise RuntimeError(
        "Could not find a valid CA certificate store. "
        "Please install 'certifi' or ensure system certificates are mapped."
    )


_cached_ctx: Optional[ssl.SSLContext] = None


def get_ssl_context() -> ssl.SSLContext:
    """Return a lazily-initialized, cached SSL context."""
    global _cached_ctx
    if _cached_ctx is None:
        _cached_ctx = _build_ssl_context()
    return _cached_ctx
