# Copyright (c) Vispy Development Team. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.

from . import BaseGLProxy, _copy_gl_functions
from ._constants import *

class DummyProxy(BaseGLProxy):
    def __call__(self, funcname, returns, *args): ...

# Instantiate proxy and inject functions
_proxy = ...
