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
    bug = "Discover django models and discover the correct field type",
    test_model,
    django_env(),
    r#"
from typing import assert_type

from django.db import models # E: Could not find import of `django.db` 

class Person(models.Model):
    first_name = models.CharField(max_length=30)

p = Person(first_name="Alice")
assert_type(p.first_name, str) # E: assert_type(Any, str) failed
"#,
);
