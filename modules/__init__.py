"""
Package modules — Interpolation Numérique
==========================================
Regroupe toutes les fonctions d'interpolation selon le cours.

Lagrange  : base_lagrange, interpolation_lagrange, lagrange_sympy, lagrange_scipy
Spline    : calculer_zi, position, evaluer_spline, spline_sympy, spline_scipy
"""

from .lagrange import (
    base_lagrange,
    interpolation_lagrange,
    lagrange_sympy,
    lagrange_scipy,
)

from .spline import (
    calculer_zi,
    position,
    evaluer_spline,
    spline_sympy,
    spline_scipy,
)

__all__ = [
    # Lagrange
    "base_lagrange",
    "interpolation_lagrange",
    "lagrange_sympy",
    "lagrange_scipy",
    # Spline
    "calculer_zi",
    "position",
    "evaluer_spline",
    "spline_sympy",
    "spline_scipy",
]
