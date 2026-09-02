#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# update_revision - update the PyTorch benchmark pin for pyrefly.
# Usage: ./update_revision.sh <40-hex-commit>
#
# The pin lives in one place: benches/pytorch_pin.bzl (rev + tarball sha256).
# Internally, pyrefly/BUCK loads it to build the pytorch-bench-src http_archive
# fetched from Manifold, so CI needs no github egress. In OSS, benches/pytorch.rs
# parses the rev out of the same file and shallow-clones PyTorch at that commit.
# This script regenerates pytorch_pin.bzl and uploads the archive to Manifold.
#
# Must run on a machine with github.com egress (laptop, not devvm/Sandcastle).
# See benches/README.md for the full procedure.
set -euo pipefail

REV="${1:-}"
if [[ ! "$REV" =~ ^[0-9a-f]{40}$ ]]; then
  echo "Usage: $0 <40-hex-pytorch-commit>" >&2
  echo "Example: $0 0ab6f77f3478f367c8eab3e20a11594252356c81" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIN_BZL="$SCRIPT_DIR/pytorch_pin.bzl"

echo "==> Cloning pytorch at $REV (shallow, filter blob:none)"
TMPDIR=$(mktemp -d /tmp/pytorch-bench.XXXXXX)
trap 'rm -rf "$TMPDIR"' EXIT

# GitHub is blocked from devvm and Sandcastle by default. Detect failure early
# and tell the user to switch to a host with egress.
if ! git clone --filter=blob:none --depth 1 https://github.com/pytorch/pytorch.git "$TMPDIR/pytorch" 2> /tmp/git-clone-err.log; then
  echo "" >&2
  echo "ERROR: git clone from github.com failed." >&2
  cat /tmp/git-clone-err.log >&2
  echo "" >&2
  echo "This script must run on a machine with github.com egress." >&2
  echo "  * devvm, Sandcastle, and most corp hosts block outbound github by design." >&2
  echo "  * Use your laptop on VPN or home wifi, or an on-demand with egress allowed." >&2
  echo "" >&2
  echo "Rerun on laptop: $0 $REV" >&2
  exit 1
fi

if ! git -C "$TMPDIR/pytorch" fetch --depth 1 origin "$REV" 2> /tmp/git-fetch-err.log; then
  echo "" >&2
  echo "ERROR: git fetch $REV from github.com failed." >&2
  cat /tmp/git-fetch-err.log >&2
  echo "" >&2
  echo "Check the commit exists at https://github.com/pytorch/pytorch/commit/$REV" >&2
  echo "and that you are on a host with github egress (laptop, not devvm)." >&2
  exit 1
fi

git -C "$TMPDIR/pytorch" checkout "$REV"

TARBALL="pytorch-${REV}.tar.gz"
OUTPATH="$(pwd)/$TARBALL"
# gzip output isn't guaranteed byte-identical across git/zlib versions, but that
# doesn't matter here: we pin the sha256 of *this* tarball and upload the same
# bytes to Manifold; consumers fetch that exact artifact rather than regenerating it.
echo "==> Creating archive $OUTPATH via git archive (no .git)"
git -C "$TMPDIR/pytorch" archive --format=tar.gz --prefix=pytorch/ -o "$OUTPATH" "$REV"

if command -v sha256sum >/dev/null 2>&1; then
  SHA256=$(sha256sum "$OUTPATH" | awk '{print $1}')
else
  SHA256=$(shasum -a 256 "$OUTPATH" | awk '{print $1}')
fi
echo "==> sha256: $SHA256"

# Write the committed pin only after the tarball exists, so a failed clone/fetch
# can't leave pytorch_pin.bzl referencing a rev with no uploaded artifact.
echo "==> Updating $PIN_BZL"
# Rewrite just the two constant lines; python avoids BSD/GNU sed differences.
python3 - "$PIN_BZL" "$REV" "$SHA256" <<'PY'
import re, sys
path, rev, sha = sys.argv[1], sys.argv[2], sys.argv[3]
txt = open(path).read()
txt = re.sub(r'PYTORCH_BENCH_REV = "[0-9a-f]{40}"', 'PYTORCH_BENCH_REV = "%s"' % rev, txt)
txt = re.sub(r'PYTORCH_BENCH_SHA256 = "[0-9a-f]{64}"', 'PYTORCH_BENCH_SHA256 = "%s"' % sha, txt)
open(path, "w").write(txt)
PY

echo ""
echo "==> Uploading tarball to Manifold pyrefly_resources/tree/$TARBALL"
if command -v manifold >/dev/null 2>&1; then
  if manifold put "$OUTPATH" "pyrefly_resources/tree/$TARBALL" --overwrite 2>&1 | tee /tmp/manifold-put.log; then
    echo "Manifold upload succeeded."
  else
    echo "" >&2
    echo "Manifold upload failed. Tarball is ready at $OUTPATH — retry:" >&2
    echo "  manifold put $TARBALL pyrefly_resources/tree/$TARBALL --overwrite" >&2
    echo "  # or UI: https://www.internalfb.com/manifold/bucket/pyrefly_resources/tree" >&2
  fi
else
  echo "manifold CLI not found — skipping upload. Install via 'dnf install fb-manifold' or use the web UI."
fi

echo ""
echo "Next steps:"
echo "  1. Review the pin change:"
echo "     sl diff pyrefly/pyrefly/benches/pytorch_pin.bzl"
echo "  2. Verify the Buck target parses:"
echo "     buck2 uquery fbcode//pyrefly/pyrefly:pytorch-bench-src"
echo "  3. Commit and submit."
echo ""
echo "Done. Tarball at $OUTPATH  sha256 $SHA256"
