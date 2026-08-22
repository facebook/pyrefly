/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use crate::django_testcase;

django_testcase!(
    test_meta_override_without_inheritance,
    r#"
from django.db import models
from rest_framework import serializers

class MyModel(models.Model):
    name = models.CharField(max_length=100)

class MySerializer(serializers.ModelSerializer):
    class Meta:
        model = MyModel
        fields = ["name"]

class GenericSerializer(serializers.ModelSerializer[MyModel]):
    class Meta:
        model = MyModel
        fields = ["name"]
"#,
);
