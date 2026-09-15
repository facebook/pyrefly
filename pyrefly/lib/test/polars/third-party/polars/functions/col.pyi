from polars.expr.expr import Expr

class Col:
    def __call__(self, *names: str) -> Expr: ...
    def __getattr__(self, name: str) -> Expr: ...

col: Col
