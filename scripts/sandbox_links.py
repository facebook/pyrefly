#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Generate shareable sandbox links for the Pyrefly playground.

The playground uses lz-string compression with compressToEncodedURIComponent.
It compresses JSON {files: {filename: content}, activeFile: string}.

Usage:
    python sandbox_links.py <file.py>
    python sandbox_links.py --code "def foo(): pass"
    python sandbox_links.py --decode <URL>

Requires: pip install lzstring
"""

import argparse
import json
import sys
from urllib.parse import unquote, urlencode, urlparse


def _get_lzstring():
    try:
        import lzstring
    except ImportError:
        print("Error: lzstring package required. Install with: pip install lzstring")
        sys.exit(1)
    return lzstring


def compress_to_encoded_uri(text: str) -> str:
    """Compress text using lz-string's compressToEncodedURIComponent."""
    lz = _get_lzstring().LZString()
    return lz.compressToEncodedURIComponent(text)


def decompress_from_encoded_uri(compressed: str) -> str | None:
    """Decompress text using lz-string's decompressFromEncodedURIComponent."""
    lz = _get_lzstring().LZString()
    return lz.decompressFromEncodedURIComponent(compressed)


def generate_pyrefly_link(
    code: str,
    filename: str = "sandbox.py",
    version: str | None = None,
) -> str:
    """Generate a Pyrefly sandbox link for a single file."""
    return generate_pyrefly_link_multifile({filename: code}, filename, version)


def generate_pyrefly_link_multifile(
    files: dict[str, str],
    active_file: str = "sandbox.py",
    version: str | None = None,
) -> str:
    """
    Generate a Pyrefly sandbox link with multiple files.

    files: dict mapping filename to content (e.g. {"sandbox.py": "...", "pyrefly.toml": "..."})
    active_file: which file tab to show initially
    """
    project_state = {
        "files": files,
        "activeFile": active_file,
    }
    compressed = compress_to_encoded_uri(json.dumps(project_state))

    params = {"project": compressed}
    if version:
        params["version"] = version

    return f"https://pyrefly.org/sandbox/?{urlencode(params)}"


def decode_pyrefly_link(url: str) -> dict:
    """Decode a Pyrefly sandbox link back to its components."""
    parsed = urlparse(url)
    raw_query = parsed.query

    if "project=" in raw_query:
        idx = raw_query.index("project=") + len("project=")
        amp_idx = raw_query.find("&", idx)
        raw_project = raw_query[idx:] if amp_idx == -1 else raw_query[idx:amp_idx]
        decoded = unquote(raw_project)
        decompressed = decompress_from_encoded_uri(decoded)
        return json.loads(decompressed) if decompressed else {}
    elif "code=" in raw_query:
        idx = raw_query.index("code=") + len("code=")
        amp_idx = raw_query.find("&", idx)
        raw_code = raw_query[idx:] if amp_idx == -1 else raw_query[idx:amp_idx]
        decompressed = decompress_from_encoded_uri(unquote(raw_code))
        if decompressed is None:
            return {}
        return {"files": {"sandbox.py": decompressed}, "activeFile": "sandbox.py"}

    return {}


def main():
    parser = argparse.ArgumentParser(description="Generate Pyrefly sandbox links")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("file", nargs="?", help="Python file to encode")
    group.add_argument("--code", "-c", help="Python code string to encode")
    group.add_argument("--decode", "-d", metavar="URL", help="Decode a sandbox URL")

    parser.add_argument(
        "--filename",
        "-f",
        default="sandbox.py",
        help="Filename for Pyrefly (default: sandbox.py)",
    )

    args = parser.parse_args()

    if args.decode:
        result = decode_pyrefly_link(args.decode)
        print(json.dumps(result, indent=2))
        return

    if args.file:
        with open(args.file) as f:
            code = f.read()
    else:
        code = args.code

    print(generate_pyrefly_link(code, filename=args.filename))


if __name__ == "__main__":
    main()
