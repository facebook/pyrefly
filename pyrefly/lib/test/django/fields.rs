/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use crate::django_testcase;
use crate::test::django::util::django_env;
use crate::testcase;

testcase!(
    test_field_instance_contract_compatibility,
    django_env().enable_missing_override_decorator_error(),
    r#"
from typing import Protocol
from django.db import models

class HasName(Protocol):
    name: str

class ReadOnlyName(Protocol):
    @property
    def name(self) -> str: ...

class AnyName(Protocol):
    name: object

class Base(models.Model):
    name = models.TextField()
    class Meta:
        abstract = True

class Contact(Base):
    name = models.CharField(max_length=200)

def accept_name(value: HasName) -> None: ...
def accept_read_only_name(value: ReadOnlyName) -> None: ...

accept_name(Contact())
accept_read_only_name(Contact())
accept_name(Base())
accept_any_name: AnyName = Contact()  # E: not assignable

class WrongOverride(Base):
    name = models.IntegerField()  # E: inconsistent

class WrongName(models.Model):
    name = models.IntegerField()

accept_name(WrongName())  # E: not assignable

class PlainBase(models.Model):
    name: str

class FieldChild(PlainBase):
    name = models.CharField(max_length=200)

class FieldBase(models.Model):
    name: models.CharField[str, str] = models.CharField(max_length=200)

class PlainChild(FieldBase):
    name: str = "Alice"

class WideSetter(models.Model):
    name: models.CharField[str | int, str] = models.CharField(max_length=200)

class NarrowSetter(WideSetter):
    name: models.CharField[str, str] = models.CharField(max_length=200)  # E: inconsistent
"#,
);

django_testcase!(
    test_field_explicit_type_arguments,
    r#"
from typing import Any, Literal, assert_type
from django.db import models
from django.db.models.fields import _FieldDescriptor

class Contact(models.Model):
    name: models.CharField[str, str] = models.CharField(max_length=200)
    code = models.CharField[str, str](max_length=20)
    permissive: models.CharField[Any, str] = models.CharField(max_length=200)
    literal = models.CharField[str, Literal["fixed"]](max_length=20)

def update(contact: Contact) -> None:
    contact.name = "Alice"
    contact.name = 123  # E: not assignable
    contact.code = 123  # E: not assignable
    contact.permissive = 123

assert_type(Contact().name, str)
assert_type(Contact.name, _FieldDescriptor[models.CharField[str, str]])
assert_type(Contact.code, _FieldDescriptor[models.CharField[str, str]])
assert_type(Contact.permissive, _FieldDescriptor[models.CharField[Any, str]])
assert_type(Contact().literal, Literal["fixed"])
"#,
);

django_testcase!(
    test_field_access_on_class_and_instance,
    r#"
from typing import assert_type

from django.db import models
from django.db.models.expressions import Combinable
from django.db.models.fields import _FieldDescriptor

class Contact(models.Model):
    address = models.CharField(max_length=200)

contact = Contact(address="123 Fake St")
assert_type(contact.address, str)
assert_type(
    Contact.address,
    _FieldDescriptor[models.CharField[str | int | Combinable, str]],
)
assert_type(Contact.address.field.max_length, int | None)
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
