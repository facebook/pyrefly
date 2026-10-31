/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use crate::test::util::TestEnv;
use crate::testcase;

fn sqlalchemy_env() -> TestEnv {
    let mut env = TestEnv::new();
    env.add_with_path(
        "sqlalchemy",
        "sqlalchemy/__init__.py",
        "from . import orm\n",
    );
    env.add_with_path(
        "sqlalchemy.sql",
        "sqlalchemy/sql/__init__.py",
        "from . import sqltypes\nfrom . import type_api\n",
    );
    env.add_with_path(
        "sqlalchemy.sql.type_api",
        "sqlalchemy/sql/type_api.py",
        r#"
from typing import Generic, TypeVar

_T = TypeVar("_T")

class TypeEngine(Generic[_T]):
    ...

class TypeDecorator(TypeEngine[_T], Generic[_T]):
    ...
"#,
    );
    env.add_with_path(
        "sqlalchemy.sql.sqltypes",
        "sqlalchemy/sql/sqltypes.py",
        r#"
from enum import Enum as PythonEnum
from typing import Generic, TypeVar
from .type_api import TypeEngine

_T = TypeVar("_T")

class String(TypeEngine[str]):
    ...

class Integer(TypeEngine[int]):
    ...

class Float(TypeEngine[float]):
    ...

class GenericType(TypeEngine[_T], Generic[_T]):
    ...

class Enum(TypeEngine[str | PythonEnum]):
    def __init__(self, enum_class: type[PythonEnum]) -> None:
        ...
"#,
    );
    env.add_with_path(
        "sqlalchemy.orm.base",
        "sqlalchemy/orm/base.py",
        r#"
from typing import Generic, TypeVar

_T = TypeVar("_T")

class Mapped(Generic[_T]):
    ...

class _MappedAnnotationBase(Mapped[_T]):
    ...

class _DeclarativeMapped(_MappedAnnotationBase[_T]):
    ...
"#,
    );
    env.add_with_path(
        "sqlalchemy.orm.properties",
        "sqlalchemy/orm/properties.py",
        r#"
from typing import Generic, TypeVar
from .base import _DeclarativeMapped

_T = TypeVar("_T")

class MappedColumn(_DeclarativeMapped[_T]):
    column: object

    def __init__(self) -> None:
        ...
"#,
    );
    env.add_with_path(
        "sqlalchemy.orm",
        "sqlalchemy/orm/__init__.py",
        r#"
from typing import Any
from sqlalchemy.sql.type_api import TypeEngine
from .base import Mapped
from .properties import MappedColumn

__all__ = ["Mapped", "MappedColumn", "mapped_column"]

def mapped_column(
    type_: TypeEngine[Any] | type[TypeEngine[Any]] | None = None,
    *args: Any,
    **kw: Any,
) -> MappedColumn[Any]:
    return MappedColumn()
"#,
    );
    env
}

testcase!(
    test_sqlalchemy_mapped_column_infers_type,
    sqlalchemy_env(),
    r#"
from typing import assert_type
from sqlalchemy.orm import MappedColumn, mapped_column
from sqlalchemy.sql.sqltypes import Integer, String

class Model:
    name = mapped_column(String())
    quantity = mapped_column(Integer)
    sku = mapped_column(String(), nullable=False)
    pk = mapped_column(Integer, primary_key=True)
    nullable_pk = mapped_column(Integer, primary_key=True, nullable=True)

assert_type(Model.name, MappedColumn[str | None])
assert_type(Model.quantity, MappedColumn[int | None])
assert_type(Model.sku, MappedColumn[str])
assert_type(Model.pk, MappedColumn[int])
assert_type(Model.nullable_pk, MappedColumn[int | None])

def make_column() -> MappedColumn[str | None]:
    result = mapped_column(String())
    result.column
    return result
"#,
);

testcase!(
    test_sqlalchemy_mapped_column_substitutes_generic_types,
    sqlalchemy_env(),
    r#"
from typing import assert_type, reveal_type
from sqlalchemy.orm import MappedColumn, mapped_column
from sqlalchemy.sql.sqltypes import GenericType
from sqlalchemy.sql.type_api import TypeDecorator

assert_type(mapped_column(GenericType[float]()), MappedColumn[float | None])
assert_type(mapped_column(TypeDecorator[str]()), MappedColumn[str | None])
reveal_type(mapped_column(GenericType))  # E: revealed type: MappedColumn[Unknown | None]
"#,
);

testcase!(
    test_sqlalchemy_mapped_column_infers_enum_class,
    sqlalchemy_env(),
    r#"
from enum import Enum as PythonEnum
from typing import assert_type
from sqlalchemy.orm import MappedColumn, mapped_column
from sqlalchemy.sql.sqltypes import Enum

class Color(PythonEnum):
    RED = "red"

assert_type(mapped_column(Enum(Color)), MappedColumn[Color | None])
"#,
);

testcase!(
    test_sqlalchemy_mapped_column_respects_explicit_annotation,
    sqlalchemy_env(),
    r#"
from typing import Literal, assert_type
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql.sqltypes import Float, Integer, String

Key = Literal["api_key", "api_secret"]

class Model:
    key: Mapped[Key] = mapped_column(String(), nullable=False)
    quantity: Mapped[int] = mapped_column(Integer, primary_key=True)
    ratio: Mapped[float] = mapped_column(Float(), nullable=False)
    maybe_ratio: Mapped[float | None] = mapped_column(Float())

assert_type(Model.key, Mapped[Key])
assert_type(Model.quantity, Mapped[int])
assert_type(Model.ratio, Mapped[float])
assert_type(Model.maybe_ratio, Mapped[float | None])
"#,
);
