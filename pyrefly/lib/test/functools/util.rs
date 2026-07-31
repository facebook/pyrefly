/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use crate::test::util::TestEnv;

/// `functools` lives in the bundled typeshed, so (unlike the `attrs` tests) no extra stub
/// search-path is needed. This mirrors the per-library test-module layout and gives the
/// `functools` tests their own macro/namespace.
pub fn functools_env() -> TestEnv {
    TestEnv::new()
}

#[macro_export]
macro_rules! functools_testcase {
    (bug = $explanation:literal, $name:ident, $contents:literal,) => {
        #[test]
        fn $name() -> anyhow::Result<()> {
            $crate::test::util::testcase_for_macro(
                $crate::test::functools::util::functools_env(),
                $contents,
                file!(),
                line!(),
            )
        }
    };
    ($name:ident, $contents:literal,) => {
        #[test]
        fn $name() -> anyhow::Result<()> {
            $crate::test::util::testcase_for_macro(
                $crate::test::functools::util::functools_env(),
                $contents,
                file!(),
                line!() - 1,
            )
        }
    };
}
