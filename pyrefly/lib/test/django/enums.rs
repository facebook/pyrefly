/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use crate::test::util::TestEnv;
use crate::testcase;

fn django_env() -> TestEnv {
    let path = std::env::var("DJANGO_TEST_PATH").expect("DJANGO_TEST_PATH must be set");
    TestEnv::new_with_site_package_path(&path)
}

testcase!(
    test_int_choices,
    django_env(),
    r#"
from django.db.models import IntegerChoices

def get_choices_using_property(choices: type[IntegerChoices]) -> list[tuple[int, str]]:
    return choices.choices
"#,
);

testcase!(
    test_int_choices_enums,
    django_env(),
    r#"
from django.db.models import IntegerChoices
from django.utils.translation import gettext_lazy as _
from typing_extensions import assert_type

class Suit(IntegerChoices):
    DIAMOND = 1, _("Diamond")
    SPADE = 2, _("Spade")
    HEART = 3, _("Heart")
    CLUB = 4, _("Club")

assert_type(Suit.CLUB.value, int)
"#,
);

testcase!(
    bug = "1- fix the enum inheriting from multiple classes bug and 2- label types are not correct",
    test_enum_iterable,
    django_env(),
    r#"
import enum

from django.db.models import TextChoices
from django.utils.functional import _StrOrPromise

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
assert_type([member.label for choices in x1 for member in choices], list[_StrOrPromise]) # E: Type `type[Gender]` is not iterable  # E: Type `type[Medal]` is not iterable # E: assert_type(list[str], list[_StrPromise | str])
assert_type([member.value for choices in x1 for member in choices], list[str]) # E: Type `type[Gender]` is not iterable  # E: Type `type[Medal]` is not iterable
"#,
);

testcase!(
    test_enum_choices,
    django_env(),
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
assert_type(VoidChoices.values, list[Any | None]) 
assert_type(VoidChoices.choices, list[tuple[Any | None, str]]) 
assert_type(VoidChoices.ABYSS, Literal[VoidChoices.ABYSS])
assert_type(VoidChoices.ABYSS.name, Literal["ABYSS"])
assert_type(VoidChoices.ABYSS.label, str) 
assert_type(VoidChoices.ABYSS.value, Any)
assert_type(VoidChoices.ABYSS.do_not_call_in_templates, Literal[True])
assert_type(VoidChoices.__empty__, str)
"#,
);

testcase!(
    test_enum_union,
    django_env(),
    r#"
from django.utils.functional import _StrOrPromise
from typing_extensions import assert_type, Any
from django.db.models import Choices
from django.utils.translation import gettext_lazy as _

class Suit(Choices):
    DIAMOND = 1, _("Diamond")
    SPADE = "2", _("Spade")

assert_type(Suit.values, list[Any])

"#,
);

testcase!(
    test_enum_gettext_lazy,
    django_env(),
    r#"
from typing import assert_type
from django.db.models import IntegerChoices
from django.utils.translation import gettext_lazy as _
from django.utils.functional import _StrOrPromise

class A(IntegerChoices):
    A = 1

class B(IntegerChoices):
    B = 1, _("B")

assert_type(A.choices, list[tuple[int, str]])
assert_type(B.choices, list[tuple[int, _StrOrPromise | str]])

"#,
);

testcase!(
    test_enum_empty,
    django_env(),
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

testcase!(
    test_enum_value,
    django_env(),
    r#"
from typing import assert_type
from django.db.models import IntegerChoices
from django.utils.translation import gettext_lazy as _
class A(IntegerChoices):
    A = 1, _("A")
assert_type(A.A.value, int)

"#,
);

testcase!(
    bug = "process labels correctly. They should have type str.",
    test_enum_auto,
    django_env(),
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

testcase!(
    test_enum_auto_with_gettext_lazy,
    django_env(),
    r#"
import enum

from django.db.models import TextChoices

from django.utils.functional import _StrPromise
from django.utils.translation import gettext_lazy as _
from typing_extensions import assert_type

class Medal(TextChoices):
    GOLD = enum.auto()
    SILVER = enum.auto(), _("B")
    BRONZE = enum.auto()

assert_type(Medal.choices, list[tuple[str, str | _StrPromise]])
assert_type(Medal.GOLD.label, (str | _StrPromise))
assert_type(Medal.SILVER.label, (str | _StrPromise))
assert_type(Medal.GOLD.value, str)
"#,
);
