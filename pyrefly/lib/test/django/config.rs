/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use crate::test::util::TestEnv;
use crate::testcase;

fn django_env() -> TestEnv {
    let path = std::env::var("DJANGO_TEST_PATH").expect("DJANGO_TEST_PATH must be set");
    TestEnv::new_with_site_package_path(&path)
}

testcase!(
    bug = "we do not correctly pick up the _Getter descriptor",
    test_model,
    django_env(),
    r#"
from django.apps.config import AppConfig
from typing_extensions import assert_type

class FooConfig(AppConfig):
    name = "foo"
    default_auto_field = "django.db.models.BigAutoField"

assert_type( # E: assert_type(_Getter[str] | str, str)
    FooConfig.default_auto_field, str 
) 
"#,
);
