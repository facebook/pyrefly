# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from . import bar
from .bar import Bar

Bar().foo
bar.Bar().foo
