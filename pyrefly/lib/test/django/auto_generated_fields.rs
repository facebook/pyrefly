/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use crate::django_testcase;

django_testcase!(
    test_auto_generated_id_field,
    r#"
from typing import assert_type

from django.db import models

class Reporter(models.Model):
    name = models.CharField(max_length=100)

reporter = Reporter()
assert_type(reporter.id, int)
assert_type(reporter.pk, int)
"#,
);

django_testcase!(
    test_existing_field,
    r#"
from typing import assert_type

from django.db import models

class Reporter(models.Model):
    id : str = "id"

reporter = Reporter()
assert_type(reporter.id, str)
"#,
);

django_testcase!(
    test_custom_pk,
    r#"
from typing import assert_type
from django.db import models
from uuid import UUID

class Article(models.Model):
    uuid = models.UUIDField(primary_key=True)

class B(Article):
    pass

article = Article()
article.id # E: Object of class `Article` has no attribute `id`
assert_type(article.uuid, UUID)
assert_type(article.pk, UUID)

article2 = B()
article2.id # E: Object of class `B` has no attribute `id`
assert_type(article2.uuid, UUID)
assert_type(article2.pk, UUID)
"#,
);

django_testcase!(
    test_abstract_model_charfield_pk,
    r#"
from typing import assert_type
from django.db import models

class StrIdMixin(models.Model):
    id = models.CharField(max_length=36, primary_key=True)
    class Meta:
        abstract = True

class StrIdChildModel(StrIdMixin):
    name = models.CharField(max_length=100)

child = StrIdChildModel()
assert_type(child.id, str)
assert_type(child.pk, str)
"#,
);

// Multiple abstract mixins: the one with primary_key=True must win,
// even if a later base has no custom PK (regression test for #2218).
django_testcase!(
    test_abstract_model_charfield_pk_multiple_mixins,
    r#"
from typing import assert_type
from django.db import models

class StrIdMixin(models.Model):
    id = models.CharField(max_length=36, primary_key=True)
    class Meta:
        abstract = True

class AuditMixin(models.Model):
    created_by = models.CharField(max_length=100)
    class Meta:
        abstract = True

class ConcreteModel(StrIdMixin, AuditMixin):
    name = models.CharField(max_length=100)

obj = ConcreteModel()
assert_type(obj.id, str)
assert_type(obj.pk, str)
"#,
);
