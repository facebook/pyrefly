/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

import useBaseUrl from '@docusaurus/useBaseUrl';

export const DEFAULT_SANDBOX_PROGRAM = `
from typing import *

def test(x: int):
    return x + 1

reveal_type(test(42))
`.trimStart();
