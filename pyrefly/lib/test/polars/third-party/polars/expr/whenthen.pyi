from polars.expr.expr import Expr

class When:
    def then(self, statement: object) -> Then: ...

class Then(Expr):
    def otherwise(self, statement: object) -> Expr: ...
