/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use crate::django_testcase;
use crate::test::django::util::django_env;
use crate::test::util::TestEnv;
use crate::testcase;

django_testcase!(
    test_foreign_key_basic,
    r#"
from typing import assert_type

from django.db import models

class Reporter(models.Model):
    full_name = models.CharField(max_length=70)

class Article(models.Model):
    reporter = models.ForeignKey(Reporter, on_delete=models.CASCADE)

article = Article()
assert_type(article.reporter, Reporter)
assert_type(article.reporter.full_name, str)
assert_type(article.reporter_id, int)

class B(Article):
    pass

b = B()

assert_type(b.reporter, Reporter)
assert_type(b.reporter.full_name, str)
assert_type(b.reporter_id, int)
"#,
);

django_testcase!(
    test_foreign_key_nullable,
    r#"
from typing import assert_type

from django.db import models

class Reporter(models.Model): ...

class Article(models.Model):
    reporter = models.ForeignKey(Reporter, null=True, on_delete=models.CASCADE)

article = Article()
assert_type(article.reporter, Reporter | None)
assert_type(article.reporter_id, int | None)
"#,
);

django_testcase!(
    test_foreign_key_string_literal,
    r#"
from typing import assert_type
from django.db import models

class Article(models.Model):
    reporter = models.ForeignKey('Reporter', on_delete=models.CASCADE)

class Reporter(models.Model):
    full_name = models.CharField(max_length=70)

article = Article()
assert_type(article.reporter, Reporter)
assert_type(article.reporter.full_name, str)
assert_type(article.reporter_id, int)
"#,
);

fn django_env_with_model_import() -> TestEnv {
    let mut env = django_env();
    env.add(
        "reporter",
        r#"
from django.db import models

class Reporter(models.Model):
    full_name = models.CharField(max_length=70)
"#,
    );
    env
}

testcase!(
    test_foreign_key_string_literal_imported,
    django_env_with_model_import(),
    r#"
from typing import assert_type, TYPE_CHECKING
from django.db import models

if TYPE_CHECKING:
    from .reporter import Reporter

class Article(models.Model):
    reporter = models.ForeignKey('Reporter', on_delete=models.CASCADE)

article = Article()
assert_type(article.reporter, Reporter)
assert_type(article.reporter.full_name, str)
assert_type(article.reporter_id, int)
"#,
);

django_testcase!(
    test_foreign_key_self_reference,
    r#"
from typing import assert_type

from django.db import models

class Person(models.Model):
    name = models.CharField(max_length=100)
    parent = models.ForeignKey('self', null=True, on_delete=models.CASCADE)

person = Person()
assert_type(person.parent, Person | None)
assert_type(person.parent_id, int | None)
if person.parent:
    assert_type(person.parent.name, str)
"#,
);

django_testcase!(
    test_foreign_key_custom_pk,
    r#"
from typing import assert_type
from uuid import UUID

from django.db import models

class Reporter(models.Model):
    uuid = models.UUIDField(primary_key=True)
    full_name = models.CharField(max_length=70)

class Reporter2(models.Model):
    pass

class Article(models.Model):
    reporter = models.ForeignKey(Reporter, on_delete=models.CASCADE)

class B(Article):
    pass

article = Article()
assert_type(article.reporter, Reporter)
assert_type(article.reporter_id, UUID)

b = B()
assert_type(b.reporter, Reporter)
assert_type(b.reporter_id, UUID)
"#,
);

django_testcase!(
    test_foreign_key_in_function,
    r#"
from django.db import models

def test_through_db_table_mutually_exclusive(self):
    class Child(models.Model):
        pass

    class Through(models.Model):
        referred = models.ForeignKey(Child, on_delete=models.CASCADE)
"#,
);
