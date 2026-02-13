from sympy.codegen.abstract_nodes import List as AbstractList
from sympy.codegen.ast import Token

class List(AbstractList): ...

class NumExprEvaluate(Token):
    _fields = ...
