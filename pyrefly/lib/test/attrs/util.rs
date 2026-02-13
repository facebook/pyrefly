/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use crate::test::util::TestEnv;

pub fn attrs_env() -> TestEnv {
    let path = std::env::var("ATTRS_TEST_PATH").expect("ATTRS_TEST_PATH must be set");
    TestEnv::new_with_site_package_path(&path)
}

#[macro_export]
macro_rules! attrs_testcase {
    (bug = $explanation:literal, $name:ident, $contents:literal,) => {
        #[test]
        fn $name() -> anyhow::Result<()> {
            $crate::test::util::testcase_for_macro(
                $crate::test::attrs::util::attrs_env(),
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
                $crate::test::attrs::util::attrs_env(),
                $contents,
                file!(),
                line!() - 1,
            )
        }
    };
}
