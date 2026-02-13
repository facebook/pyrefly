from typing_extensions import Self

from sympy.polys.domains.domain import Domain

class SimpleDomain(Domain):
    is_Simple: bool = True
    def inject(self, *gens) -> Self: ...
