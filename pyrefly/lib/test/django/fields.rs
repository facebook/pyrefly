/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use crate::django_testcase;

// https://github.com/facebook/pyrefly/issues/5322
django_testcase!(
    test_field_expression_assignment,
    r#"
from typing import assert_type
from django.db import models
from django.db.models import F, Value

class Book(models.Model):
    is_draft = models.BooleanField()
    pages = models.IntegerField()
    title = models.CharField(max_length=100)
    optional_pages = models.IntegerField(null=True)
    ordinary: bool = False

def update(book: Book) -> None:
    book.is_draft = F("other")
    assert_type(book.is_draft, bool)
    book.is_draft = Value(True)
    assert_type(book.is_draft, bool)
    book.pages = F("pages") + 1
    assert_type(book.pages, int)
    book.title = Value("New title")
    assert_type(book.title, str)
    book.optional_pages = F("pages")
    assert_type(book.optional_pages, int | None)
    book.optional_pages = None
    book.is_draft = "invalid"  # E: is not assignable to attribute `is_draft`
    book.is_draft = None  # E: is not assignable to attribute `is_draft`
    book.ordinary = F("other")  # E: is not assignable to attribute `ordinary`

class SpecialBook(Book):
    pass

def update_subclass(book: SpecialBook) -> None:
    book.is_draft = F("other")
    assert_type(book.is_draft, bool)
"#,
);

django_testcase!(
    test_textfield_nullable,
    r#"
from django.db import models

class Group(models.Model):
    name = models.TextField()

class Customer(models.Model):
    name = models.TextField(null=True)

def test():
    c = Customer.objects.create()
    c.name = None
"#,
);

django_testcase!(
    test_get_foo_display,
    r#"
from typing import assert_type

from django.db import models

class Person(models.Model):
    SHIRT_SIZES = {
        "S": "Small",
        "M": "Medium",
        "L": "Large",
    }
    name = models.CharField(max_length=60)
    shirt_size = models.CharField(max_length=2, choices=SHIRT_SIZES)

p = Person(name="Fred Flintstone", shirt_size="L")
p.save()
assert_type(p.shirt_size, str) 
assert_type(p.get_shirt_size_display(), str)
"#,
);

django_testcase!(
    test_charfield_choices_inline_tuple,
    r#"
from typing import assert_type, Literal

from django.db import models

class Card(models.Model):
    suit = models.CharField(
        max_length=100,
        choices=(
            ("CLUBS", "Clubs"),
            ("SPADES", "Spades"),
            ("HEARTS", "Hearts"),
            ("DIAMONDS", "Diamonds"),
        ),
    )

card = Card(suit="CLUBS")
assert_type(card.suit, Literal["CLUBS", "SPADES", "HEARTS", "DIAMONDS"])
"#,
);

// We can consider narrowing the type further
// but it's unclear if it's worth doing so since
// django stubs give type Any for the aggregate values
django_testcase!(
    test_int_field,
    r#"
from django.db import models
from typing import assert_type, Any

class Default(models.Model):
    int_field = models.IntegerField(default=0)

total_sum_typed = Default.objects.aggregate(
        total=models.Sum("int_field", output_field=models.IntegerField())
)
assert_type(total_sum_typed, dict[str, Any]) 
"#,
);

// Mapping a union of Django field types to their Python types rebuilds the union through the
// solver, so a literal is absorbed by the class the field maps to.
django_testcase!(
    test_field_union_absorbs_literal,
    r#"
from typing import reveal_type
from django.db import models

def flag() -> bool: ...

class M(models.Model):
    x = models.CharField(max_length=10) if flag() else "A"

def probe(m: M) -> None:
    reveal_type(m.x)  # E: revealed type: str
"#,
);
