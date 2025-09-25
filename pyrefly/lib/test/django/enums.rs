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
    bug = "mypy does not error on this definition, however pyright does.",
    test_int_choices,
    django_env(),
    r#"
from django.db.models import IntegerChoices

def get_choices_using_property(choices: type[IntegerChoices]) -> list[tuple[int, str]]:
    return choices.choices # E: Returned type `list[tuple[int, _StrPromise | str]]` is not assignable to declared return type `list[tuple[int, str]]`
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
    bug = "Properly interpret TextChoices properly as enums so that this test does not raise these false positives.",
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
assert_type([member.label for choices in x1 for member in choices], list[_StrOrPromise]) 
assert_type([member.value for choices in x1 for member in choices], list[str]) 
"#,
);

testcase!(
    bug = "Add full support for django enum Choices. Currently, not all cases are supported.",
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
assert_type(VoidChoices.labels, list[str]) # E: assert_type(list[_StrPromise | str], list[str])
assert_type(VoidChoices.values, list[Any | None]) # E: assert_type(list[Any], list[Any | None])
assert_type(VoidChoices.choices, list[tuple[Any | None, str]]) # E: assert_type(list[tuple[Any, _StrPromise | str]]
assert_type(VoidChoices.ABYSS, Literal[VoidChoices.ABYSS])
assert_type(VoidChoices.ABYSS.name, Literal["ABYSS"])
assert_type(VoidChoices.ABYSS.label, str) # E: assert_type(_StrPromise | str, str) failed 
assert_type(VoidChoices.ABYSS.value, Any) # E: assert_type(auto, Any)
assert_type(VoidChoices.ABYSS.do_not_call_in_templates, Literal[True])
assert_type(VoidChoices.__empty__, str)
"#,
);
