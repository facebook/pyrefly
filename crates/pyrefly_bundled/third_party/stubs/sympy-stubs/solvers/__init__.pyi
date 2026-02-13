from sympy.core.assumptions import check_assumptions, failing_assumptions
from sympy.core.singleton import S
from sympy.solvers.decompogen import decompogen
from sympy.solvers.deutils import ode_order
from sympy.solvers.diophantine import diophantine
from sympy.solvers.inequalities import (
    reduce_abs_inequalities,
    reduce_abs_inequality,
    reduce_inequalities,
    solve_poly_inequality,
    solve_rational_inequalities,
    solve_univariate_inequality,
)
from sympy.solvers.ode import checkodesol, classify_ode, dsolve, homogeneous_order
from sympy.solvers.pde import checkpdesol, classify_pde, pde_separate, pde_separate_add, pde_separate_mul, pdsolve
from sympy.solvers.polysys import solve_poly_system, solve_triangulated
from sympy.solvers.recurr import rsolve, rsolve_hyper, rsolve_poly, rsolve_ratio
from sympy.solvers.solvers import (
    checksol,
    det_quick,
    inv_quick,
    nsolve,
    solve,
    solve_linear,
    solve_linear_system,
    solve_linear_system_LU,
    solve_undetermined_coeffs,
)
from sympy.solvers.solveset import linear_eq_to_matrix, linsolve, nonlinsolve, solveset, substitution

Complexes = ...
__all__ = [
    "solve",
    "solve_linear_system",
    "solve_linear_system_LU",
    "solve_undetermined_coeffs",
    "nsolve",
    "solve_linear",
    "checksol",
    "det_quick",
    "inv_quick",
    "check_assumptions",
    "failing_assumptions",
    "diophantine",
    "rsolve",
    "rsolve_poly",
    "rsolve_ratio",
    "rsolve_hyper",
    "checkodesol",
    "classify_ode",
    "dsolve",
    "homogeneous_order",
    "solve_poly_system",
    "solve_triangulated",
    "pde_separate",
    "pde_separate_add",
    "pde_separate_mul",
    "pdsolve",
    "classify_pde",
    "checkpdesol",
    "ode_order",
    "reduce_inequalities",
    "reduce_abs_inequality",
    "reduce_abs_inequalities",
    "solve_poly_inequality",
    "solve_rational_inequalities",
    "solve_univariate_inequality",
    "decompogen",
    "solveset",
    "linsolve",
    "linear_eq_to_matrix",
    "nonlinsolve",
    "substitution",
    "Complexes",
]
