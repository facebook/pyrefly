# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# IMPORTANT: *Any* change to this file will kick off an upload of a new version of pyrefly to PyPI,
# and a new VSCode extension although the upload will likely fail if you haven't changed the version number.
#
# An automated PyPI/extension release happens once a week. To update the version for a manual release:
# * The version number is in the format "<major>.<minor>.<patch>".
# * Do exactly ONE of the following:
#   * Increase the patch number by 1 if the release contains only minor changes like bug fixes.
#   * Increase the minor number by 1 and set the patch number to 0 if the release contains major
#     changes like new features.
#   * Increase the major number by 1 and set the minor and patch numbers to 0 to indicate a
#     significant shift in the project. This should almost never happen.
# * Do not include leading zeroes or anything else extra.
# * After updating the version, run `arc autocargo -p pyrefly` to regenerate `Cargo.toml`
#   and put the resulting diff up for review. Once the diff lands, the new version should be
#   available on PyPI within a few hours.
VERSION = "0.51.2"
