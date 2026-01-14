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
    bug = "Reverse relations not yet implemented",
    test_one_to_one_reverse_default_name,
    r#"
from django.db import models

class Place(models.Model):
    name = models.CharField(max_length=50)

class Restaurant(models.Model):
    place = models.OneToOneField(Place, on_delete=models.CASCADE)

place = Place()
# OneToOne reverse is just the lowercase model name (no _set suffix)
place.restaurant  # E: `Place` has no attribute `restaurant`
"#,
);

// ManyToManyField reverse relation: returns a manager like FK
django_testcase!(
    bug = "Reverse relations not yet implemented",
    test_many_to_many_reverse_default_name,
    r#"
from django.db import models

class Tag(models.Model):
    name = models.CharField(max_length=50)

class Article(models.Model):
    tags = models.ManyToManyField(Tag)

tag = Tag()
# ManyToMany default reverse name is <model_lowercase>_set
tag.article_set  # E: `Tag` has no attribute `article_set`
"#,
);

django_testcase!(
    bug = "Reverse relations not yet implemented",
    test_one_to_one_reverse_custom_name,
    r#"
from django.db import models

class Place(models.Model):
    name = models.CharField(max_length=50)

class Restaurant(models.Model):
    place = models.OneToOneField(Place, on_delete=models.CASCADE, related_name='dining_spot')

place = Place()
place.dining_spot  # E: `Place` has no attribute `dining_spot`
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
    bug = "Reverse relations not yet implemented",
    test_many_to_many_reverse_custom_name,
    r#"
from django.db import models

class Tag(models.Model):
    name = models.CharField(max_length=50)

class Article(models.Model):
    tags = models.ManyToManyField(Tag, related_name='tagged_articles')

tag = Tag()
tag.tagged_articles  # E: `Tag` has no attribute `tagged_articles`
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
    bug = "Reverse relations not yet implemented",
    test_many_to_many_self_reference_asymmetrical,
    r#"
from django.db import models

class Person(models.Model):
    name = models.CharField(max_length=100)
    # Asymmetrical M2M: followers vs following relationship
    following = models.ManyToManyField('self', symmetrical=False, related_name='followers')

person = Person()
# With symmetrical=False, reverse accessor is created
person.followers  # E: `Person` has no attribute `followers`
"#,
);
