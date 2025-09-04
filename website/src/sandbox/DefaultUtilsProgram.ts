/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * @format
 */

export const DEFAULT_UTILS_PROGRAM = `
# Utility functions

def format_number(x: int) -> str:
    return f"Number: {x}"
`.trimStart();