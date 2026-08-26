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

class ChildSerializer(MySerializer):
    class Meta:
        fields = "__all__"
"#,
);

django_testcase!(
    test_declared_fields,
    r#"
from rest_framework import serializers

class FieldNameSerializer(serializers.Serializer):
    label = serializers.CharField()
    source = serializers.CharField()
    context = serializers.CharField()
    data = serializers.CharField()

class NestedSerializer(serializers.Serializer):
    child = FieldNameSerializer()
"#,
);

django_testcase!(
    test_declared_field_suppression_is_scoped,
    r#"
from rest_framework import serializers

class BadSerializer(serializers.Serializer):
    label = 42  # E: `Literal[42]` is not assignable to attribute `label`

class NotSerializer(serializers.CharField):
    label = serializers.CharField()  # E: `CharField` is not assignable to attribute `label`

class ParentSerializer(serializers.Serializer):
    class Meta: ...

class ChildSerializer(ParentSerializer):
    class Meta:  # E: overrides parent class `ParentSerializer` in an inconsistent manner
        ...
"#,
);
