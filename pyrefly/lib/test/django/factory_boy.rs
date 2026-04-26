/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// TODO: Create a dedicated factory_boy_testcase! macro with factory-boy stubs.
use crate::django_testcase;

django_testcase!(
    bug = "https://github.com/facebook/pyrefly/issues/3214",
    test_create_returns_model,
    r#"
from typing import assert_type

from django.db import models
from factory.django import DjangoModelFactory  # E: Cannot find module `factory.django`

class User(models.Model):
    username = models.CharField(max_length=150)

class UserFactory(DjangoModelFactory):
    class Meta:
        model = User

    username = "testuser"

user = UserFactory.create()
assert_type(user, User)  # E: assert_type(Unknown, User) failed
"#,
);
