/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

import useBaseUrl from '@docusaurus/useBaseUrl';

/**
 * This sandbox allows you to experiment with Pyrefly, a fast Python type checker.
 *
 * Features:
 * - Write and run Python code directly in your browser
 * - Get real-time type checking feedback
 * - Use reveal_type() to inspect inferred types
 * - Explore Python type annotations
 *
 * Try modifying the example below or write your own code to see Pyrefly in action!
 */
export const DEFAULT_SANDBOX_PROGRAM = `
# Welcome to the Pyrefly Sandbox, this is a page
# where you can write, share and learn about typing in Python.
#
# Here are a couple features of this sandbox:
# - Get real-time type checking feedback
# - Use reveal_type() to inspect inferred types
# - experiment with the IDE features that we support (e.g. autocomplete, go-to-definition, hover, etc)
# - Write and run Python code directly in your browser

from typing import *
from utils import format_number

def test(x: int):
    return format_number(x)

# reveal_type will produce a type error that tells you the type Pyrefly has
# computed for the return (in this case, str)
reveal_type(test(42))
`.trimStart();
