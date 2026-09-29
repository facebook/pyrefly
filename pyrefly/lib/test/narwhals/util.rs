/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::env;

use crate::test::util::TestEnv;

/// Returns an environment containing minimal stubs with the real Narwhals qualified names.
pub fn env_with_narwhals_stubs() -> TestEnv {
    let path = env::var("NARWHALS_TEST_PATH").expect("NARWHALS_TEST_PATH must be set");
    TestEnv::new_with_site_package_paths(&[&path])
}

/// Narwhals wraps a backend frame, so some behavior is only observable with a backend present.
pub fn env_with_narwhals_and_polars_stubs() -> TestEnv {
    let narwhals = env::var("NARWHALS_TEST_PATH").expect("NARWHALS_TEST_PATH must be set");
    let polars = env::var("POLARS_TEST_PATH").expect("POLARS_TEST_PATH must be set");
    TestEnv::new_with_site_package_paths(&[&narwhals, &polars])
}

#[macro_export]
macro_rules! narwhals_testcase {
    (bug = $explanation:literal, $name:ident, $contents:literal,) => {
        #[test]
        fn $name() -> anyhow::Result<()> {
            $crate::test::util::testcase_for_macro(
                $crate::test::narwhals::util::env_with_narwhals_stubs(),
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
                $crate::test::narwhals::util::env_with_narwhals_stubs(),
                $contents,
                file!(),
                line!() - 1,
            )
        }
    };
}
