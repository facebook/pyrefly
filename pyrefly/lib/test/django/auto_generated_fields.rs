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
