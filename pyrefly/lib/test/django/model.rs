/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use crate::django_testcase;
use crate::test::django::util::django_env;
use crate::test::util::TestEnv;
use crate::testcase;

fn django_model_utils_env() -> TestEnv {
    let mut env = django_env();
    env.add(
        "model_utils.managers",
        r#"
from typing import Generic, TypeVar

from django.db import models

ModelT = TypeVar("ModelT", bound=models.Model, covariant=True)

class InheritanceQuerySet(models.QuerySet[ModelT]): ...

class InheritanceManager(models.Manager[ModelT]):
    def select_subclasses(self) -> InheritanceQuerySet[ModelT]: ...
"#,
    );
    env
}

testcase!(
    test_django_model_utils_custom_manager,
    django_model_utils_env(),
    r#"
from typing import assert_type

from django.db import models
from model_utils.managers import InheritanceManager, InheritanceQuerySet

class Place(models.Model):
    objects = InheritanceManager()

assert_type(Place.objects, InheritanceManager[Place])
assert_type(Place.objects.select_subclasses(), InheritanceQuerySet[Place])
"#,
);

django_testcase!(
    test_model,
    r#"
from typing import assert_type

from django.db import models

class Person(models.Model):
    first_name = models.CharField(max_length=30)

p = Person(first_name="Alice")
assert_type(p.first_name, str)
"#,
);

django_testcase!(
    test_model_admin_tuple,
    r#"
from django.contrib import admin
from django.db import models

class Person(models.Model):
    first_name = models.CharField(max_length=None)
    last_name = models.CharField(max_length=None)
    birthday = models.DateField()

class PersonFieldsetTupleAdmin(admin.ModelAdmin[Person]):
    fieldsets = (
        (
            "Personal Details",
            {
                "description": "Personal details of a person.",
                "fields": (("first_name", "last_name"), "birthday"),
            },
        ),
    )
"#,
);

django_testcase!(
    test_model_admin_list,
    r#"
from django.contrib import admin 
from django.db import models

class Person(models.Model):
    first_name = models.CharField(max_length=None)
    last_name = models.CharField(max_length=None)
    birthday = models.DateField()

class PersonFieldsetListAdmin(admin.ModelAdmin[Person]):
    fieldsets = [
        (
            "Personal Details",
            {
                "description": "Personal details of a person.",
                "fields": [["first_name", "last_name"], "birthday"],
            },
        )
    ]
"#,
);

django_testcase!(
    test_meta_override_without_inheritance,
    r#"
from django.db import models

class DateTimeMixin(models.Model):

    created_at = models.DateTimeField(auto_now_add=True, editable=False)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        abstract = True

class Invoice(DateTimeMixin):
    class Meta:
        verbose_name = "Invoice"
        verbose_name_plural = "Invoices"
"#,
);
