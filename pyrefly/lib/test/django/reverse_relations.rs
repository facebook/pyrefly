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

django_testcase!(
    test_foreign_key_reverse_disabled,
    r#"
from django.db import models

class Author(models.Model):
    name = models.CharField(max_length=100)

class Book(models.Model):
    # related_name='+' disables the reverse accessor entirely
    author = models.ForeignKey(Author, on_delete=models.CASCADE, related_name='+')

author = Author()
# No reverse accessor should exist
author.book_set  # E: `Author` has no attribute `book_set`
"#,
);

// Self-referential FK creates reverse accessor on the same model
django_testcase!(
    bug = "Reverse relations not yet implemented",
    test_foreign_key_reverse_self_reference,
    r#"
from django.db import models

class Person(models.Model):
    name = models.CharField(max_length=100)
    # Self-referential FK: a person can have a parent who is also a Person
    parent = models.ForeignKey('self', null=True, on_delete=models.CASCADE)

person = Person()
person.person_set  # E: `Person` has no attribute `person_set`
"#,
);
