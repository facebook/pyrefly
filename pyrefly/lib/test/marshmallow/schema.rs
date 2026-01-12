/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use crate::marshmallow_testcase;

marshmallow_testcase!(
    test_meta_override_without_inheritance,
    r#"
from marshmallow import Schema

class UserSchema(Schema):
    class Meta:
        strict = True
        ordered = True
"#,
);
