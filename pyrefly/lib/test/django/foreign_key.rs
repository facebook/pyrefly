/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use crate::django_testcase;

django_testcase!(
    bug = "infer ForeignKey field type and support chained access, as well as support _id suffix",
    test_foreign_key_basic,
    r#"
from typing import assert_type

from django.db import models

class Reporter(models.Model):
    full_name = models.CharField(max_length=70)

class Article(models.Model):
    reporter = models.ForeignKey(Reporter, on_delete=models.CASCADE)

article = Article()
assert_type(article.reporter, Reporter) # E: assert_type(Any, Reporter) failed 
assert_type(article.reporter.full_name, str) # E: assert_type(Any, str) failed 
assert_type(article.reporter_id, int) # E: assert_type(Any, int) failed # E: Object of class `Article` has no attribute `reporter_id`
"#,
);

django_testcase!(
    bug = "We should take into account the nullability of the foreign key",
    test_foreign_key_nullable,
    r#"
from typing import assert_type

from django.db import models

class Reporter(models.Model): ...

class Article(models.Model):
    reporter = models.ForeignKey(Reporter, null=True, on_delete=models.CASCADE)

article = Article()
assert_type(article.reporter,  Reporter | None) # E: assert_type(Any, Reporter | None) failed 

"#,
);
