from typing_extensions import Self

from sympy.core.function import Function, UndefinedFunction

class lerchphi(Function):
    def fdiff(self, argindex=...): ...

class polylog(Function):
    @classmethod
    def eval(cls, s, z) -> type[UndefinedFunction] | Self | None: ...
    def fdiff(self, argindex=...): ...

class zeta(Function):
    @classmethod
    def eval(cls, s, a=...) -> Self | None: ...
    def fdiff(self, argindex=...): ...

class dirichlet_eta(Function):
    @classmethod
    def eval(cls, s, a=...) -> Self | type[UndefinedFunction] | None: ...

class riemann_xi(Function):
    @classmethod
    def eval(cls, s) -> None: ...

class stieltjes(Function):
    @classmethod
    def eval(cls, n, a=...) -> None: ...
