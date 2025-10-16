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
from typing import assert_type

from django.db import models

class Person(models.Model):
    first_name = models.CharField(max_length=30)

p = Person(first_name="Alice")
assert_type(p.first_name, str)
"#,
);

django_testcase!(
    bug = "We error on the definition of PersonFieldsetTupleAdmin, which is wrong.",
    test_model_admin_tuple,
    r#"
from django.contrib import admin
from django.db import models

class Person(models.Model):
    first_name = models.CharField(max_length=None)
    last_name = models.CharField(max_length=None)
    birthday = models.DateField()

class PersonFieldsetTupleAdmin(admin.ModelAdmin[Person]):
    fieldsets = ( # E: `tuple[tuple[Literal['Personal Details'], dict[str, str | tuple[tuple[str, str], str]]]]` is not assignable to attribute `fieldsets` with type `list[tuple[_StrPromise | str | None, TypedDict[_FieldOpts]]] | tuple[()] | tuple[tuple[_StrPromise | str | None, TypedDict[_FieldOpts]], ...] | None`
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
