/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use crate::binding::binding::KeyClassSynthesizedFields;
use crate::django_testcase;
use crate::test::django::util::django_env;
use crate::test::util::TestEnv;
use crate::test::util::get_class;
use crate::testcase;

// Cross-module reverse relations: when the FK target is in a different module,
// reverse relations cannot be synthesized yet because our current analysis only scans the current module.
fn django_env_with_separate_models() -> TestEnv {
    let mut env = django_env();
    env.add(
        "author",
        r#"
from django.db import models

class Author(models.Model):
    name = models.CharField(max_length=100)
"#,
    );
    env
}

fn django_env_with_module(name: &str, code: &str) -> TestEnv {
    let mut env = django_env();
    env.add(name, code);
    env
}

fn django_env_without_auto_field() -> TestEnv {
    let mut env = TestEnv::new();
    env.add("django", "");
    env.add("django.db", "");
    env.add(
        "django.db.models",
        r#"
from django.db.models.base import Model
"#,
    );
    env.add("django.db.models.base", "class Model: pass");
    env
}

testcase!(
    test_missing_auto_field_does_not_panic,
    django_env_without_auto_field(),
    r#"
from django.db import models

class Author(models.Model):
    pass

Author()
"#,
);

django_testcase!(
    test_foreign_key_reverse_default_name,
    r#"
from django.db import models
from django.db.models.fields.related_descriptors import RelatedManager
from typing import assert_type

class Reporter(models.Model):
    full_name = models.CharField(max_length=70)

class Article(models.Model):
    reporter = models.ForeignKey(Reporter, on_delete=models.CASCADE)

reporter = Reporter()
# Default reverse name is <model_lowercase>_set
assert_type(reporter.article_set, RelatedManager[Article])
"#,
);

django_testcase!(
    test_foreign_key_reverse_custom_name,
    r#"
from django.db import models
from django.db.models.fields.related_descriptors import RelatedManager
from typing import assert_type

class Author(models.Model):
    name = models.CharField(max_length=100)

class Book(models.Model):
    author = models.ForeignKey(Author, on_delete=models.CASCADE, related_name='written_books')

author = Author()
# Custom related_name should be used instead of default
assert_type(author.written_books, RelatedManager[Book])
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

// An attribute whose name is not an identifier is unreachable from Python code, so the
// only way to observe one is to look at the synthesized fields directly.
#[test]
fn test_foreign_key_reverse_invalid_identifier_synthesizes_nothing() {
    let mut env = django_env();
    env.add(
        "main",
        r#"
from django.db import models

class Author(models.Model): ...

class Book(models.Model):
    author = models.ForeignKey(Author, on_delete=models.CASCADE, related_name='written_books')

class Magazine(models.Model):
    author = models.ForeignKey(Author, on_delete=models.CASCADE, related_name='123articles')
"#,
    );
    let (state, handle_for) = env.to_state();
    let handle = handle_for("main");
    let author = get_class("Author", &handle, &state);
    let solutions = state.transaction().get_solutions(&handle).unwrap();
    let fields = solutions.get(&KeyClassSynthesizedFields(author.index()));
    let mut names = fields
        .fields()
        .map(|(name, _)| name.as_str())
        .collect::<Vec<_>>();
    names.sort();
    // `123articles` is dropped entirely rather than synthesized under an unusable name.
    assert_eq!(names, vec!["id", "pk", "written_books"]);
}

// A related name that is not a valid identifier names an attribute that can never be
// accessed, so no reverse relation is created at all -- not even under the default name.
django_testcase!(
    test_foreign_key_reverse_invalid_identifier,
    r#"
from django.db import models

class Author(models.Model):
    name = models.CharField(max_length=100)

class Book(models.Model):
    author = models.ForeignKey(Author, on_delete=models.CASCADE, related_name=' written_books ')

class Magazine(models.Model):
    author = models.ForeignKey(Author, on_delete=models.CASCADE, related_name='123articles')

class Poem(models.Model):
    author = models.ForeignKey(Author, on_delete=models.CASCADE, related_name='written.poems')

author = Author()
author.written_books  # E: `Author` has no attribute `written_books`
author.book_set  # E: `Author` has no attribute `book_set`
author.magazine_set  # E: `Author` has no attribute `magazine_set`
author.poem_set  # E: `Author` has no attribute `poem_set`
"#,
);

django_testcase!(
    test_foreign_key_reverse_unknown_app_label,
    r#"
from django.db import models

class Author(models.Model): ...

class Book(models.Model):
    author = models.ForeignKey(
        Author,
        on_delete=models.CASCADE,
        related_name='%(app_label)s_books',
    )

author = Author()
author.main_books  # E: `Author` has no attribute `main_books`
"#,
);

testcase!(
    test_foreign_key_reverse_app_label,
    django_env_with_module(
        "myapp.models",
        r#"
from django.db import models
from django.db.models.fields.related_descriptors import RelatedManager
from typing import assert_type

class Author(models.Model): ...

class Book(models.Model):
    author = models.ForeignKey(
        Author,
        on_delete=models.CASCADE,
        related_name='%(app_label)s_%(class)s_books',
    )

author = Author()
assert_type(author.myapp_book_books, RelatedManager[Book])
"#
    ),
    "",
);

// Self-referential FK creates reverse accessor on the same model
django_testcase!(
    test_foreign_key_reverse_self_reference,
    r#"
from django.db import models
from django.db.models.fields.related_descriptors import RelatedManager
from typing import assert_type

class Person(models.Model):
    name = models.CharField(max_length=100)
    # Self-referential FK: a person can have a parent who is also a Person
    parent = models.ForeignKey('self', null=True, on_delete=models.CASCADE)

person = Person()
assert_type(person.person_set, RelatedManager[Person])
"#,
);

django_testcase!(
    test_foreign_key_reverse_unicode_default_name,
    r#"
from django.db import models
from django.db.models.fields.related_descriptors import RelatedManager
from typing import assert_type

class Reporter(models.Model): ...

class ÜberBook(models.Model):
    reporter = models.ForeignKey(Reporter, on_delete=models.CASCADE)

reporter = Reporter()
assert_type(reporter.überbook_set, RelatedManager[ÜberBook])
"#,
);

testcase!(
    bug = "Cross-module reverse relations not supported",
    test_foreign_key_reverse_cross_module,
    django_env_with_separate_models(),
    r#"
from django.db import models
from .author import Author

class Book(models.Model):
    author = models.ForeignKey(Author, on_delete=models.CASCADE)

# Author is defined in a different module, so reverse relation won't be synthesized
author = Author()
author.book_set  # E: `Author` has no attribute `book_set`
"#,
);

// OneToOneField reverse relation: returns single object (not a manager like FK)
// Default name is just the lowercase model name without `_set`
django_testcase!(
    test_one_to_one_reverse_default_name,
    r#"
from django.db import models
from typing import assert_type

class Place(models.Model):
    name = models.CharField(max_length=50)

class Restaurant(models.Model):
    place = models.OneToOneField(Place, on_delete=models.CASCADE)

place = Place()
# OneToOne reverse is just the lowercase model name (no _set suffix)
assert_type(place.restaurant, Restaurant)
"#,
);

// ManyToManyField reverse relation: returns a manager like FK
django_testcase!(
    test_many_to_many_reverse_default_name,
    r#"
from django.db import models
from django.db.models.fields.related_descriptors import ManyRelatedManager
from typing import assert_type

class Tag(models.Model):
    name = models.CharField(max_length=50)

class Article(models.Model):
    tags = models.ManyToManyField(Tag)

tag = Tag()
# ManyToMany default reverse name is <model_lowercase>_set
assert_type(tag.article_set, ManyRelatedManager[Article, models.Model])
"#,
);

django_testcase!(
    test_one_to_one_reverse_custom_name,
    r#"
from django.db import models
from typing import assert_type

class Place(models.Model):
    name = models.CharField(max_length=50)

class Restaurant(models.Model):
    place = models.OneToOneField(Place, on_delete=models.CASCADE, related_name='dining_spot')

place = Place()
assert_type(place.dining_spot, Restaurant)
"#,
);

django_testcase!(
    test_one_to_one_reverse_disabled,
    r#"
from django.db import models

class Place(models.Model):
    name = models.CharField(max_length=50)

class Restaurant(models.Model):
    place = models.OneToOneField(Place, on_delete=models.CASCADE, related_name='+')

place = Place()
# No reverse accessor should exist
place.restaurant  # E: `Place` has no attribute `restaurant`
"#,
);

django_testcase!(
    test_many_to_many_reverse_custom_name,
    r#"
from django.db import models
from django.db.models.fields.related_descriptors import ManyRelatedManager
from typing import assert_type

class Tag(models.Model):
    name = models.CharField(max_length=50)

class Article(models.Model):
    tags = models.ManyToManyField(Tag, related_name='tagged_articles')

tag = Tag()
assert_type(tag.tagged_articles, ManyRelatedManager[Article, models.Model])
"#,
);

django_testcase!(
    test_many_to_many_reverse_disabled,
    r#"
from django.db import models

class Tag(models.Model):
    name = models.CharField(max_length=50)

class Article(models.Model):
    tags = models.ManyToManyField(Tag, related_name='+')

tag = Tag()
# No reverse accessor should exist
tag.article_set  # E: `Tag` has no attribute `article_set`
"#,
);

// Self-referential ManyToMany is symmetrical by default, meaning no reverse accessor
// is created because the relation is bidirectional through the same field
django_testcase!(
    test_many_to_many_self_reference_symmetrical,
    r#"
from django.db import models

class Person(models.Model):
    name = models.CharField(max_length=100)
    # Symmetrical M2M: friends is accessible from both sides via the same field
    friends = models.ManyToManyField('self')

person = Person()
# No person_set because symmetrical=True (default for self-referential M2M)
person.person_set  # E: `Person` has no attribute `person_set`
"#,
);

django_testcase!(
    test_many_to_many_self_reference_asymmetrical,
    r#"
from django.db import models
from django.db.models.fields.related_descriptors import ManyRelatedManager
from typing import assert_type

class Person(models.Model):
    name = models.CharField(max_length=100)
    # Asymmetrical M2M: followers vs following relationship
    following = models.ManyToManyField('self', symmetrical=False, related_name='followers')

person = Person()
# With symmetrical=False, reverse accessor is created
assert_type(person.followers, ManyRelatedManager[Person, models.Model])
"#,
);

django_testcase!(
    test_many_to_many_self_reference_dynamic_symmetry,
    r#"
from django.db import models
from django.db.models.fields.related_descriptors import ManyRelatedManager
from typing import assert_type

symmetrical = False

class Person(models.Model):
    following = models.ManyToManyField('self', symmetrical=symmetrical, related_name='followers')

person = Person()
assert_type(person.followers, ManyRelatedManager[Person, models.Model])
"#,
);
