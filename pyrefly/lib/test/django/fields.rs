/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use crate::django_testcase;

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

django_testcase!(
    test_queryset_as_manager_preserves_custom_methods,
    r#"
from django.db import models

class NotificationQuerySet(models.QuerySet["Notification"]):
    def resolved(self):
        return self.filter(resolved_at__isnull=False)

class Notification(models.Model):
    resolved_at = models.DateTimeField(null=True, blank=True)
    objects = NotificationQuerySet.as_manager()

Notification.objects.resolved()
Notification.objects.all().resolved()
"#,
);

django_testcase!(
    test_queryset_manager_type,
    r#"
from typing import assert_type
from django.db import models

class Q(models.QuerySet["M"]):
    def active(self) -> int:
        return 1

    def _private(self) -> int:
        return 1

    def excluded(self) -> int:
        return 1
    excluded.queryset_only = True  # type: ignore[missing-attribute]

    def _included(self) -> int:
        return 1
    _included.queryset_only = False  # type: ignore[missing-attribute]

    @property
    def label(self) -> str:
        return "label"

    @classmethod
    def class_method(cls) -> int:
        return 1

class M(models.Model):
    objects = Q.as_manager()

def accept_manager(manager: models.Manager[M]) -> None: ...

accept_manager(M.objects)
assert_type(M.objects, models.Manager[M])
accept_manager(M.objects.db_manager())
assert_type(M.objects.db_manager().active(), int)
assert_type(M.objects.get(), M)
assert_type(M.objects.create(), M)
assert_type(M.objects.get_queryset(), Q)
assert_type(M.objects.all(), Q)
assert_type(M.objects.filter().active(), int)
assert_type(M.objects.active(), int)
assert_type(M.objects._included(), int)
M.objects.delete()  # E: has no attribute `delete`
M.objects._private()  # E: has no attribute `_private`
M.objects.excluded()  # E: has no attribute `excluded`
M.objects.label  # E: has no attribute `label`
M.objects.class_method()  # E: has no attribute `class_method`
M.objects.all().delete()
iter(M.objects)  # E: No matching overload found
"#,
);

django_testcase!(
    test_queryset_manager_model_binding,
    r#"
from typing import Generic, TypeVar, assert_type
from django.db import models

T = TypeVar("T", bound=models.Model)

class Q(models.QuerySet[T], Generic[T]):
    def active(self) -> int:
        return 1

class BareQ(models.QuerySet):
    def active(self) -> int:
        return 1

class Base(models.Model):
    objects = Q.as_manager()
    records = BareQ.as_manager()

    class Meta:
        abstract = True

class Child(Base): ...

assert_type(Base.objects.get(), Base)
assert_type(Child.objects.get(), Child)
assert_type(Child.records.get(), Child)
assert_type(Child.objects.active(), int)
assert_type(Child.objects.all().get(), Child)
assert_type(Child.objects.get_queryset(), Q[Child])
assert_type(Child.objects.filter().get(), Child)
assert_type(Child.records.all().get(), Child)

class FixedQ(models.QuerySet["FixedBase"]):
    def active(self) -> int:
        return 1

    def accept_base(self, other: "FixedBase") -> "FixedBase":
        return other

class FixedBase(models.Model):
    objects = FixedQ.as_manager()

    class Meta:
        abstract = True

class FixedChild(FixedBase): ...

assert_type(FixedChild.objects.get(), FixedChild)
assert_type(FixedChild.objects.all().get(), FixedChild)
assert_type(FixedChild.objects.accept_base(FixedBase()), FixedBase)

class Plain(models.Model):
    objects = models.QuerySet.as_manager()

assert_type(Plain.objects.get(), Plain)
assert_type(Plain.objects.all().get(), Plain)

R = TypeVar("R")

class RowQ(models.QuerySet[T, R], Generic[T, R]):
    def active(self) -> int:
        return 1

class RowModel(models.Model):
    objects = RowQ.as_manager()

assert_type(RowModel.objects.get(), RowModel)
assert_type(RowModel.objects.all().get(), RowModel)
"#,
);

django_testcase!(
    test_queryset_manager_overrides,
    r#"
from django.db import models

class Q1(models.QuerySet):
    def first_method(self) -> int:
        return 1

class Q2(models.QuerySet):
    def second_method(self) -> int:
        return 2

class Base(models.Model):
    records = Q1.as_manager()

class Child(Base):
    records = Q2.as_manager()

Child.records.second_method()

class Parent(models.Model):
    objects: int = 1  # E: overrides ClassVar of the same name in parent class `Model`

class Invalid(Parent):
    objects: str = "bad"  # E: overrides parent class `Parent` in an inconsistent manner

class Pretender:
    @classmethod
    def as_manager(cls) -> str:
        return "not a manager"

class Ordinary(models.Model):
    records = Pretender.as_manager()

Ordinary.records.upper()
"#,
);
