/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use crate::test::util::TestEnv;

pub fn factory_boy_env() -> TestEnv {
    let django_path = std::env::var("DJANGO_TEST_PATH").expect("DJANGO_TEST_PATH must be set");
    let factory_boy_path =
        std::env::var("FACTORY_BOY_TEST_PATH").expect("FACTORY_BOY_TEST_PATH must be set");
    TestEnv::new_with_site_package_paths(&[&django_path, &factory_boy_path])
}

#[macro_export]
macro_rules! factory_boy_testcase {
    (bug = $explanation:literal, $name:ident, $contents:literal,) => {
        #[test]
        fn $name() -> anyhow::Result<()> {
            $crate::test::util::testcase_for_macro(
                $crate::test::factory_boy::util::factory_boy_env(),
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
                $crate::test::factory_boy::util::factory_boy_env(),
                $contents,
                file!(),
                line!() - 1,
            )
        }
    };
}
