# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# This import should fail because core.py is inside module_dir which has a config
# But we're outside that config, so this reference should NOT appear in results
# when searching for Symbol inside module_dir
from module_dir.core import Symbol

Symbol
