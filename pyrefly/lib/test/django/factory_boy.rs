/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use crate::factory_boy_testcase;

factory_boy_testcase!(
    test_create_returns_model,
    r#"
from typing import assert_type

from django.db import models
from factory.django import DjangoModelFactory

class User(models.Model):
    username = models.CharField(max_length=150)

class UserFactory(DjangoModelFactory):
    class Meta:
        model = User

    username = "testuser"

user = UserFactory.create()
assert_type(user, User)
"#,
);

factory_boy_testcase!(
    test_build_returns_model,
    r#"
from typing import assert_type

from django.db import models
from factory.django import DjangoModelFactory

class User(models.Model):
    username = models.CharField(max_length=150)

class UserFactory(DjangoModelFactory):
    class Meta:
        model = User

user = UserFactory.build()
assert_type(user, User)
"#,
);

factory_boy_testcase!(
    test_create_batch_returns_list,
    r#"
from typing import assert_type

from django.db import models
from factory.django import DjangoModelFactory

class User(models.Model):
    username = models.CharField(max_length=150)

class UserFactory(DjangoModelFactory):
    class Meta:
        model = User

users = UserFactory.create_batch(3)
assert_type(users, list[User])
"#,
);

factory_boy_testcase!(
    test_model_attribute_access,
    r#"
from django.db import models
from factory.django import DjangoModelFactory

class Document(models.Model):
    title = models.CharField(max_length=200)

class DocumentFactory(DjangoModelFactory):
    class Meta:
        model = Document

doc = DocumentFactory.create()
title = doc.title
"#,
);

factory_boy_testcase!(
    test_build_batch_returns_list,
    r#"
from typing import assert_type

from django.db import models
from factory.django import DjangoModelFactory

class User(models.Model):
    username = models.CharField(max_length=150)

class UserFactory(DjangoModelFactory):
    class Meta:
        model = User

users = UserFactory.build_batch(3)
assert_type(users, list[User])
"#,
);

factory_boy_testcase!(
    test_inherited_meta,
    r#"
from typing import assert_type

from django.db import models
from factory.django import DjangoModelFactory

class User(models.Model):
    username = models.CharField(max_length=150)

class UserFactory(DjangoModelFactory):
    class Meta:
        model = User

class AdminFactory(UserFactory):
    pass

admin = AdminFactory.create()
assert_type(admin, User)
"#,
);

factory_boy_testcase!(
    test_no_meta_class,
    r#"
from factory.django import DjangoModelFactory

class BarebonesFactory(DjangoModelFactory):
    pass

obj = BarebonesFactory.create()
"#,
);

factory_boy_testcase!(
    test_meta_without_model,
    r#"
from factory.django import DjangoModelFactory

class IncompleteFactory(DjangoModelFactory):
    class Meta:
        pass

obj = IncompleteFactory.create()
"#,
);
