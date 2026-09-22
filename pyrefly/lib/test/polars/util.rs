/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::env;

use crate::test::util::TestEnv;

/// Returns an environment containing minimal stubs with the real Polars qualified names.
pub fn env_with_polars_stubs() -> TestEnv {
    let path = env::var("POLARS_TEST_PATH").expect("POLARS_TEST_PATH must be set");
    TestEnv::new_with_site_package_paths(&[&path])
}

#[macro_export]
macro_rules! polars_testcase {
    (bug = $explanation:literal, $name:ident, $contents:literal,) => {
        #[test]
        fn $name() -> anyhow::Result<()> {
            $crate::test::util::testcase_for_macro(
                $crate::test::polars::util::env_with_polars_stubs(),
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
                $crate::test::polars::util::env_with_polars_stubs(),
                $contents,
                file!(),
                line!() - 1,
            )
        }
    };
}
