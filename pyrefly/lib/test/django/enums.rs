/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use crate::django_testcase;

django_testcase!(
    test_int_choices,
    r#"
from django.db.models import IntegerChoices

def get_choices_using_property(choices: type[IntegerChoices]) -> list[tuple[int, str]]:
    return choices.choices
"#,
);

django_testcase!(
    test_int_choices_enums,
    r#"
from django.db.models import IntegerChoices
from django.utils.translation import gettext_lazy as _
from typing_extensions import assert_type

class Suit(IntegerChoices):
    DIAMOND = 1, _("Diamond")
    SPADE = 2, _("Spade")
    HEART = 3, _("Heart")
    CLUB = 4, _("Club")

assert_type(Suit.CLUB._value_, int)
assert_type(Suit.CLUB.value, int)
assert_type(Suit.values, list[int])
"#,
);

django_testcase!(
    test_enum_iterable,
    r#"
import enum

from django.db.models import TextChoices

from typing_extensions import assert_type

class Gender(TextChoices):
    MALE = "M"
    FEMALE = "F"
    NOT_SPECIFIED = "X"

    __empty__ = "(Undeclared)"

class Medal(TextChoices):
    GOLD = enum.auto()
    SILVER = enum.auto()
    BRONZE = enum.auto()

# Assertions for mixing multiple choices types with consistent base types - only `TextChoices`.
x1 = (Medal, Gender)
assert_type([member.label for choices in x1 for member in choices], list[str])
assert_type([member.value for choices in x1 for member in choices], list[str])
"#,
);

django_testcase!(
    test_enum_choices,
    r#"

import enum
from typing import Any, Literal

from django.db.models import Choices
from typing_extensions import assert_type

class BaseEmptyChoices(Choices):
    __empty__ = "Python's None"

class VoidChoices(BaseEmptyChoices):
    ABYSS = enum.auto()
    CHASM = enum.auto()

assert_type(VoidChoices.names, list[str])
assert_type(VoidChoices.labels, list[str])
assert_type(VoidChoices.values, list[int | None])
assert_type(VoidChoices.choices, list[tuple[int | None, str]])
assert_type(VoidChoices.ABYSS, Literal[VoidChoices.ABYSS])
assert_type(VoidChoices.ABYSS.name, Literal["ABYSS"])
assert_type(VoidChoices.ABYSS.label, str)
assert_type(VoidChoices.ABYSS.value, int)
assert_type(VoidChoices.ABYSS.do_not_call_in_templates, Literal[True])
assert_type(VoidChoices.__empty__, str)
"#,
);

django_testcase!(
    test_enum_union,
    r#"
from typing_extensions import assert_type, Any, Literal
from django.db.models import Choices
from django.utils.translation import gettext_lazy as _

class Suit(Choices):
    DIAMOND = 1, _("Diamond")
    SPADE = "2", _("Spade")

def test(suit: Suit):
    assert_type(suit._value_, tuple[Literal[1]] | tuple[Literal['2']])
    assert_type(suit.value, tuple[Literal[1]] | tuple[Literal['2']])

assert_type(Suit.DIAMOND._value_, tuple[Literal[1]])
assert_type(Suit.DIAMOND.value, tuple[Literal[1]])
assert_type(Suit.values, list[tuple[Literal[1]] | tuple[Literal['2']]])
"#,
);

django_testcase!(
    test_overwrite_value,
    r#"
from django.db.models import Choices
from typing import Any, assert_type

class A(Choices):
    X = 1
    _value_: str

class B(Choices):
    X = 1
    @property
    def value(self) -> str: ...

assert_type(A.X._value_, str)
assert_type(A.X.value, str)
assert_type(A.values, list[str])
assert_type(B.X._value_, int)
assert_type(B.X.value, str)
assert_type(B.values, list[str])
    "#,
);

django_testcase!(
    test_enum_gettext_lazy,
    r#"
from typing import assert_type
from django.db.models import IntegerChoices
from django.utils.translation import gettext_lazy as _
from django.utils.functional import _StrPromise

class A(IntegerChoices):
    A = 1

class B(IntegerChoices):
    B = 1, _("B")

assert_type(A.choices, list[tuple[int, str]])
assert_type(B.choices, list[tuple[int, _StrPromise]])

"#,
);

django_testcase!(
    test_enum_empty,
    r#"
from typing import assert_type

from django.db.models import IntegerChoices
from django.utils.translation import gettext_lazy as _

class BaseChoices(IntegerChoices):
    __empty__ = _("(Unknown)")

class DerivedChoices(BaseChoices):
    B = 2, "B"

assert_type(DerivedChoices.values, list[int | None])

"#,
);

django_testcase!(
    test_enum_value,
    r#"
from typing import assert_type
from django.db.models import IntegerChoices
from django.utils.translation import gettext_lazy as _
class A(IntegerChoices):
    A = 1, _("A")
assert_type(A.A.value, int)

"#,
);

django_testcase!(
    test_enum_auto,
    r#"
import enum

from django.db.models import TextChoices
from typing_extensions import assert_type

class Medal(TextChoices):
    GOLD = enum.auto()
    SILVER = enum.auto()
    BRONZE = enum.auto()

assert_type(Medal.choices, list[tuple[str, str]])
assert_type(Medal.GOLD.label, str)
assert_type(Medal.GOLD.value, str)

"#,
);

django_testcase!(
    test_enum_auto_with_gettext_lazy,
    r#"
import enum

from django.db.models import TextChoices

from django.utils.functional import _StrOrPromise
from django.utils.translation import gettext_lazy as _
from typing_extensions import assert_type

class Medal(TextChoices):
    GOLD = enum.auto()
    SILVER = enum.auto(), _("B")
    BRONZE = enum.auto()

assert_type(Medal.choices, list[tuple[str, _StrOrPromise]])
assert_type(Medal.GOLD.label, _StrOrPromise)
assert_type(Medal.SILVER.label, _StrOrPromise)
assert_type(Medal.GOLD.value, str)
assert_type(Medal.SILVER.value, str)
"#,
);

django_testcase!(
    test_auto_in_tuple,
    r#"
import enum
from typing import assert_type
from django.db.models import Choices
class E(Choices):
    X = (enum.auto(), "labelX")
assert_type(E.X.value, tuple[int])
    "#,
);

django_testcase!(
    test_empty_with_type,
    r#"
from django.db.models import IntegerChoices
from django.utils.functional import _StrOrPromise
from django.utils.translation import gettext_lazy as _
from typing_extensions import assert_type

class VehicleWithEmpty(IntegerChoices):
    CAR = 1, "Carriage"
    TRUCK = 2
    JET_SKI = 3

    __empty__ = _("Unknown")

assert_type(VehicleWithEmpty.labels, list[_StrOrPromise])
"#,
);

django_testcase!(
    test_float,
    r#"
from django.db.models import Choices
from typing_extensions import assert_type

class Constants(float, Choices):
    PI = 3.141592653589793, "π"
    TAU = 6.283185307179586, "τ"

assert_type(Constants.PI.value, float)
assert_type(Constants.values, list[float])
"#,
);

django_testcase!(
    test_enum_class_iteration,
    r#"
from django.db.models import TextChoices
from typing_extensions import TypeVar

T_Choices = TypeVar("T_Choices", bound=TextChoices)

def get_choice_labels(choices: type[T_Choices]) -> list[str]:
    return [choice.label for choice in choices]
"#,
);

django_testcase!(
    test_empty_lazy_translation_widening,
    r#"
from django.db.models import IntegerChoices
from django.utils.functional import _StrPromise
from django.utils.translation import gettext_lazy as _
from typing_extensions import assert_type

class Vehicle(IntegerChoices):
    CAR = 1
    __empty__ = _("Unknown")

assert_type(Vehicle.__empty__, _StrPromise)
"#,
);

django_testcase!(
    test_three_element_tuple_member,
    r#"
from typing import assert_type, Literal
from django.db.models import Choices
class A(Choices):
    X = 1, 2, "ALabel"
assert_type(A.X.value, tuple[Literal[1], Literal[2]])
assert_type(A.X.label, str)
    "#,
);

django_testcase!(
    test_override_properties,
    r#"
from typing import Any
from django.db.models import TextChoices
class A(TextChoices):
    @property
    def label(self) -> Any: ...
    @property
    def value(self) -> Any: ...
    "#,
);
