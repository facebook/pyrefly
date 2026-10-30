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

// https://github.com/facebook/pyrefly/issues/5157
django_testcase!(
    test_foreign_key_keyword_target,
    r#"
from typing import assert_type
from uuid import UUID
from django.db import models
from django.db.models.fields.related_descriptors import RelatedManager

class Author(models.Model):
    class Meta:
        app_label = "testapp"

class Book(models.Model):
    positional = models.ForeignKey("testapp.Author", models.CASCADE, related_name="positional_books")
    author = models.ForeignKey(on_delete=models.CASCADE, to="testapp.Author", related_name="books")
    editor = models.ForeignKey(to=Author, on_delete=models.CASCADE, null=True, related_name="edited_books")
    reviewer = models.ForeignKey(to="Author", on_delete=models.CASCADE, related_name="reviewed_books")
    sequel = models.ForeignKey(to="self", on_delete=models.CASCADE, null=True)

    class Meta:
        app_label = "testapp"

book = Book()
assert_type(book.positional, Author)
assert_type(book.positional_id, int)
assert_type(book.author, Author)
assert_type(book.author_id, int)
assert_type(book.editor, Author | None)
assert_type(book.editor_id, int | None)
assert_type(book.reviewer, Author)
assert_type(book.reviewer_id, int)
assert_type(book.sequel, Book | None)
assert_type(book.sequel_id, int | None)
assert_type(Author().books, RelatedManager[Book])
assert_type(Author().edited_books, RelatedManager[Book])
assert_type(Author().reviewed_books, RelatedManager[Book])
assert_type(book.book_set, RelatedManager[Book])

class Account(models.Model):
    id = models.UUIDField(primary_key=True)

class Profile(models.Model):
    account = models.ForeignKey(to=Account, on_delete=models.CASCADE)

assert_type(Profile().account_id, UUID)
"#,
);

django_testcase!(
    test_one_to_one_keyword_target,
    r#"
from typing import assert_type
from django.db import models

class Author(models.Model): ...

class Profile(models.Model):
    author = models.OneToOneField(to=Author, on_delete=models.CASCADE)

class Draft(models.Model):
    author = models.OneToOneField(to="Author", on_delete=models.CASCADE, null=True)

assert_type(Profile().author, Author)
assert_type(Profile().author_id, int)
assert_type(Draft().author, Author | None)
assert_type(Draft().author_id, int | None)
assert_type(Author().profile, Profile)
assert_type(Author().draft, Draft)
"#,
);

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

testcase!(
    test_foreign_key_app_label_string_literal,
    django_env_with_model_import(),
    r#"
from typing import assert_type, TYPE_CHECKING
from django.db import models

if TYPE_CHECKING:
    from .reporter import Reporter

class Article(models.Model):
    reporter = models.ForeignKey('news.Reporter', on_delete=models.CASCADE)

article = Article()
assert_type(article.reporter, Reporter)
assert_type(article.reporter.full_name, str)
assert_type(article.reporter_id, int)
"#,
);

testcase!(
    test_foreign_key_module_alias,
    django_env_with_model_import(),
    r#"
from typing import assert_type
from django.db import models
import reporter as reporter_models

class Article(models.Model):
    reporter = models.ForeignKey(reporter_models.Reporter, on_delete=models.CASCADE)

article = Article()
assert_type(article.reporter, reporter_models.Reporter)
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
    test_one_to_one_field_id,
    r#"
from typing import assert_type

from django.db import models

class Reporter(models.Model):
    full_name = models.CharField(max_length=70)

class Article(models.Model):
    reporter = models.OneToOneField(Reporter, on_delete=models.CASCADE)

article = Article()
assert_type(article.reporter, Reporter)
assert_type(article.reporter.full_name, str)
assert_type(article.reporter_id, int)
"#,
);

django_testcase!(
    test_one_to_one_field_nullable_id,
    r#"
from typing import assert_type

from django.db import models

class Reporter(models.Model): ...

class Article(models.Model):
    reporter = models.OneToOneField(Reporter, null=True, on_delete=models.CASCADE)

article = Article()
assert_type(article.reporter, Reporter | None)
assert_type(article.reporter_id, int | None)
"#,
);

django_testcase!(
    test_one_to_one_field_custom_pk,
    r#"
from typing import assert_type
from uuid import UUID

from django.db import models

class Reporter(models.Model):
    uuid = models.UUIDField(primary_key=True)

class Article(models.Model):
    reporter = models.OneToOneField(Reporter, on_delete=models.CASCADE)

article = Article()
assert_type(article.reporter, Reporter)
assert_type(article.reporter_id, UUID)
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
