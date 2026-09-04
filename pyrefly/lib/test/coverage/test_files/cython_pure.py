# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Tests for Cython pure-Python mode: C-level declarations don't exist at
# runtime, so they're excluded from coverage counting.

import cython
import cython as cy
import cython.parallel
import numba
from cython import cclass, cfunc, declare

# C globals and C types are excluded, also when re-assigned later.
c_global = cython.declare(cython.int, 0)
c_reassigned = cython.declare(object, None)
c_reassigned = 1
c_annotated: object = cython.declare(object, None)
c_imported = declare(cython.int, 0)
CPtr = cython.typedef(cython.pointer[cython.int])

# Narrowing `cython`, a common pure Python mode idiom, keeps it resolvable.
if cython.compiled:
    pass
c_after_narrow = cython.declare(cython.int, 0)

# A bare `cython.X` alias is a plain variable.
py_alias = cython.declare


@cython.cfunc
def c_function(x: cython.int) -> cython.int:
    return x


# Aliased and imported spellings resolve through the imports.
@cy.cfunc
def c_aliased_function(x: cython.int) -> cython.int:
    return x


@cfunc
def c_imported_function(x: cython.int) -> cython.int:
    return x


@cclass
class ImportedExtension:
    def __dealloc__(self):
        pass


@numba.cfunc("int32(int32)")
def numba_function(x):
    return x


@cython.cclass
class Extension:
    width: cython.int

    def __cinit__(self, width: int):
        pass

    def __init__(self, width: int):
        self.width = width
        self.ptr = cython.NULL

    def __dealloc__(self):
        pass

    def __getbuffer__(self, view, flags):
        pass

    def method(self, x):
        return x


@cython.cclass
class DynamicExtension:
    __dict__: dict
    declared: cython.int

    def __init__(self):
        self.declared = 0
        self.dynamic = 1


@cython.cclass
class DynamicChild(DynamicExtension):
    def __init__(self):
        self.declared = 1
        self.dynamic = 2
        self.inherited_dynamic = 1


@cython.total_ordering
class OrderedExtension:
    def __dealloc__(self):
        pass


@cython.internal
@cython.cclass
class Internal:
    def method(self, x):
        return x


class Plain:
    def __cinit__(self):
        pass

    def __init__(self):
        self.ptr = None
