# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# IMPORTANT: *Any* change to this file will kick off the workflows to publish new Pyrefly versions
# to PyPI and VSCode, although the workflows will likely either exit without uploading or fail to
# upload if you haven't changed the release number and pushed to the correct branch.
#
# Follow the release process at facebook/RELEASE.md to cut a new release. The process includes
# flows for dev previews, regular minor releases, and patch releases for urgent fixes.
#
# For documentation purposes, here is how the release pipeline works, although you should not cut a
# release manually unless absolutely necessary:
# * First, update the version number in this file. The format is "<major>.<minor>.<patch>" with an
#   optional `-dev.N` suffix. Check RELEASE.md for allowed version transitions.
# * After updating the version, run `arc autocargo -p pyrefly` to regenerate `Cargo.toml`
#   and put the resulting diff up for review. Once the diff lands and has been exported to GitHub:
#   * For a dev release, the publish workflows kick off when the commit lands on main.
#   * For a minor or patch release, the publish workflows kick off when the commit has been
#     cherry-picked onto a release branch.
# * The publish workflows are:
#   * https://github.com/facebook/pyrefly/actions/workflows/publish_to_pypi.yml
#   * https://github.com/facebook/pyrefly/actions/workflows/deploy_extension.yml
VERSION = "1.4.0-dev.1"
