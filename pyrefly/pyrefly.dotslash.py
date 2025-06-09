#!/usr/bin/env fbpython
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import dotslash

dotslash.export_fbcode_build(
    target="fbcode//pyrefly/pyrefly:fbpyrefly",
    oncall="pyre",
    install_platforms={
        dotslash.InstallPlatform.LINUX_AARCH64: {},
        dotslash.InstallPlatform.LINUX_X86_64: {},
        dotslash.InstallPlatform.MAC_AARCH64: {},
        dotslash.InstallPlatform.MAC_X86_64: {},
        dotslash.InstallPlatform.WINDOWS: {"sign": True},
    },
)
