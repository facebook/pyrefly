/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use crate::django_testcase;

django_testcase!(
    bug = "Add basic support for ManyToMany - need to detect ManyToManyField and return ManyRelatedManager;
    Add basic QuerySet type inference and Manager method support",
    test_basic,
    r#"
from django.db.models.query import QuerySet
from django.db.models.fields.related_descriptors import ManyRelatedManager
from typing import assert_type
from django.db import models

class Author(models.Model):
    name = models.CharField(max_length=100)


class Book(models.Model):
    title = models.CharField(max_length=200)
    authors = models.ManyToManyField(Author, related_name='books')

book = Book()
assert_type(book.authors, ManyRelatedManager[Author, models.Model]) # E: assert_type(Any, ManyRelatedManager[Author, Model])
assert_type(book.authors.all(), QuerySet[Author, Author]) # E: assert_type(Any, QuerySet[Author, Author]) 

assert_type(book.authors.filter(name="Bob"), QuerySet[Author, Author]) # E: assert_type(Any, QuerySet[Author, Author])
assert_type(book.authors.create(name="Alice"), Author) # E: assert_type(Any, Author)

book.authors.add("wrong type")  
"#,
);
