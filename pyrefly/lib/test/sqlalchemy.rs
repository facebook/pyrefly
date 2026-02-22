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
"#,
    );
    env.add_with_path(
        "sqlalchemy.sql.sqltypes",
        "sqlalchemy/sql/sqltypes.py",
        r#"
from .type_api import TypeEngine

class String(TypeEngine[str]):
    ...

class Integer(TypeEngine[int]):
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
from .properties import MappedColumn

__all__ = ["MappedColumn", "mapped_column"]

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

assert_type(Model.name, MappedColumn[str | None])
assert_type(Model.quantity, MappedColumn[int | None])
assert_type(Model.sku, MappedColumn[str])
assert_type(Model.pk, MappedColumn[int])
"#,
);
