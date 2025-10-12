/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use crate::django_testcase;

django_testcase!(
    test_model,
    r#"
from django.apps.config import AppConfig
from typing_extensions import assert_type

class FooConfig(AppConfig):
    name = "foo"
    default_auto_field = "django.db.models.BigAutoField"

assert_type( 
    FooConfig.default_auto_field, str 
) 
"#,
);
