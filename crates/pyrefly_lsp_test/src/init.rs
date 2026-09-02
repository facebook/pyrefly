/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use anstream::ColorChoice;
use pyrefly_util::trace::init_tracing;

pub fn init_test() {
    ColorChoice::write_global(ColorChoice::Always);
    init_tracing(true, true);
}
