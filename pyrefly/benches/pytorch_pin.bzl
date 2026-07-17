# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Single source of truth for the PyTorch benchmark pin. `pyrefly/BUCK` loads these
# to build the internal `pytorch-bench-src` http_archive (fetched from Manifold, so
# CI needs no github egress); the OSS `cargo bench` path parses PYTORCH_BENCH_REV
# out of this file and shallow-clones PyTorch at that commit. Update with
# `benches/update_revision.sh` — do not edit by hand.
PYTORCH_BENCH_REV = "0ab6f77f3478f367c8eab3e20a11594252356c81"

# sha256 of the `git archive` tarball uploaded to Manifold for the rev above.
PYTORCH_BENCH_SHA256 = "4cdd89bdd1d54e1a902056fd63173dd561f923a92e992fecf1abe1ba1a893ba2"
