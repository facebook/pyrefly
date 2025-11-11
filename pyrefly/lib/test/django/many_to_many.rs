/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use crate::django_testcase;

django_testcase!(
    bug = "Add basic support for ManyToMany. Also, book.authors should be  ManyRelatedManager[Author, Book_authors]; potentially v2",
    test_basic,
    r#"
from django.db.models.query import QuerySet
from typing import assert_type
from django.db import models

class Author(models.Model):
    name = models.CharField(max_length=100)


class Book(models.Model):
    title = models.CharField(max_length=200)
    authors = models.ManyToManyField(Author, related_name='books')

book = Book()
assert_type(book.authors.all(), QuerySet[Author, Author]) # E: assert_type(Any, QuerySet[Author, Author]) 
"#,
);
