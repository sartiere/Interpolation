"""
Module Lagrange — Interpolation Polynomiale de Lagrange
=======================================================
Cours d'Analyse Numérique — Master 1 GI

Formules du cours :
    L_i(x) = ∏(j≠i)  (x - x_j) / (x_i - x_j)     [Algorithme 1]
    P(x)   = Σ(j=0..n) y_j · L_j(x)               [Algorithme 2]

Contenu du module :
    base_lagrange(x, i, x_points)              → L_i(x) numérique (NumPy)
    interpolation_lagrange(x, x_points, y_pts) → P(x)   numérique (NumPy)
    lagrange_sympy(x_points, y_points)         → (P_sym, x_sym, bases)  symbolique
    lagrange_scipy(x_points, y_points)         → numpy.poly1d            scipy
"""

import numpy as np
import sympy as sp
from scipy.interpolate import lagrange as _scipy_lagrange


# ── Algorithme 1 : base de Lagrange L_i(x) ──────────────────────────────────

def base_lagrange(x, i, x_points):
    """
    Calcule la i-ième base de Lagrange L_i(x).

    Algorithme 1 (cours) :
        L = 1
        pour j = 0 à n :
            si j ≠ i : L = L · (x - x_j) / (x_i - x_j)
        retourner L

    Paramètres
    ----------
    x        : float ou np.ndarray — point(s) d'évaluation
    i        : int                 — indice de la base
    x_points : array-like          — nœuds d'interpolation [x_0, ..., x_n]

    Retourne
    --------
    L_i(x) : float ou np.ndarray
    """
    n = len(x_points)
    L = 1.0
    for j in range(n):
        if j != i:
            L *= (x - x_points[j]) / (x_points[i] - x_points[j])
    return L


# ── Algorithme 2 : polynôme d'interpolation P(x) ────────────────────────────

def interpolation_lagrange(x, x_points, y_points):
    """
    Calcule le polynôme d'interpolation de Lagrange P(x).

    Algorithme 2 (cours) :
        P = 0
        pour j = 0 à n :
            L_j = 1
            pour k = 0 à n :
                si k ≠ j : L_j = L_j · (x - x_k) / (x_j - x_k)
            P = P + y_j · L_j
        retourner P

    Propriété : P(x_i) = y_i  pour tout i.

    Paramètres
    ----------
    x        : float ou np.ndarray — point(s) d'évaluation
    x_points : array-like          — abscisses [x_0, ..., x_n]
    y_points : array-like          — ordonnées [y_0, ..., y_n]

    Retourne
    --------
    P(x) : float ou np.ndarray
    """
    n = len(x_points)
    if len(y_points) != n:
        raise ValueError("x_points et y_points doivent avoir la même taille.")

    if isinstance(x, (list, np.ndarray)):
        return np.array([interpolation_lagrange(xi, x_points, y_points) for xi in x])

    P = 0.0
    for j in range(n):
        Lj = 1.0
        for k in range(n):
            if k != j:
                Lj *= (x - x_points[k]) / (x_points[j] - x_points[k])
        P += y_points[j] * Lj
    return P


# ── SymPy : calcul symbolique exact ─────────────────────────────────────────

def _to_sym(val):
    """Convertit un nombre en entier SymPy si possible, sinon en Float."""
    v = float(val)
    return sp.Integer(int(v)) if v == int(v) else sp.Float(v, 15)


def lagrange_sympy(x_points, y_points):
    """
    Calcule symboliquement P(x) par la méthode de Lagrange.

    Suit exactement les Algorithmes 1 & 2 du cours avec SymPy.
    Calcul exact (valeurs entières → Rational, sinon Float 15 chiffres).

    Paramètres
    ----------
    x_points : liste de float — abscisses
    y_points : liste de float — ordonnées

    Retourne
    --------
    P_sym  : sp.Expr   — polynôme développé P(x)
    x_sym  : sp.Symbol — symbole x
    bases  : list[sp.Expr] — liste des L_i(x) développés
    """
    x = sp.Symbol('x')
    n = len(x_points)

    xr = [_to_sym(xi) for xi in x_points]
    yr = [_to_sym(yi) for yi in y_points]

    bases = []
    P = sp.Integer(0)

    for j in range(n):
        # Algorithme 1 : L_j(x) symbolique
        Lj = sp.Integer(1)
        for k in range(n):
            if k != j:
                Lj *= (x - xr[k]) / (xr[j] - xr[k])
        bases.append(sp.expand(Lj))

        # Algorithme 2 : P += y_j * L_j
        P += yr[j] * Lj

    return sp.expand(P), x, bases


# ── SciPy : scipy.interpolate.lagrange ──────────────────────────────────────

def lagrange_scipy(x_points, y_points):
    """
    Calcule P(x) via scipy.interpolate.lagrange.

    scipy applique la même formule mathématique que le cours :
        P(x) = Σ y_j · L_j(x)

    Paramètres
    ----------
    x_points : array-like — abscisses
    y_points : array-like — ordonnées

    Retourne
    --------
    poly : numpy.poly1d — coefficients décroissants du polynôme
    """
    return _scipy_lagrange(
        np.array(x_points, dtype=float),
        np.array(y_points, dtype=float)
    )
