/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use crate::django_testcase;

django_testcase!(
    bug = "Reverse relations not yet implemented",
    test_foreign_key_reverse_default_name,
    r#"
from django.db import models

class Reporter(models.Model):
    full_name = models.CharField(max_length=70)

class Article(models.Model):
    reporter = models.ForeignKey(Reporter, on_delete=models.CASCADE)

reporter = Reporter()
# Default reverse name is <model_lowercase>_set
reporter.article_set  # E: `Reporter` has no attribute `article_set`
"#,
);

django_testcase!(
    bug = "Reverse relations not yet implemented",
    test_foreign_key_reverse_custom_name,
    r#"
from django.db import models

class Author(models.Model):
    name = models.CharField(max_length=100)

class Book(models.Model):
    author = models.ForeignKey(Author, on_delete=models.CASCADE, related_name='written_books')

author = Author()
# Custom related_name should be used instead of default
author.written_books  # E: `Author` has no attribute `written_books`
"#,
);
