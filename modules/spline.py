"""
Module Spline Quadratique
=========================
Cours d'Analyse Numérique — Master 1 GI

Formule directe du cours (forme centrée) :
    S_i(x) = (z_{i+1} - z_i) / (2·h_i) · (x - x_i)²
             + z_i · (x - x_i)
             + y_i,          x ∈ [x_i, x_{i+1}]

    avec  c_i = y_i,  b_i = z_i,  a_i = (z_{i+1} - z_i) / (2·h_i)
    → pas de calcul explicite de a_i, b_i, c_i séparément.

    Condition initiale : z_0 = S'(x_0) = 0

    Récurrence (Algorithme 3, Étape 1) :
        z_{i+1} = 2·(y_{i+1} - y_i) / h_i − z_i

    Évaluation (Algorithme 4) :
        trouver i tel que x ∈ [x_i, x_{i+1}]
        S(x) = (z_{i+1} - z_i) / (2·h_i) · (x - x_i)² + z_i·(x - x_i) + y_i

Contenu du module :
    calculer_zi(x_pts, y_pts, z0)              → z[0..n]             NumPy
    position(x, x_pts)                         → indice i             NumPy
    evaluer_spline(x, x_pts, y_pts, z)         → S(x)                NumPy
    spline_sympy(x_pts, y_pts, z0)             → (morceaux, z_sym)   SymPy
    spline_scipy(x_pts, y_pts, z0)             → PPoly               SciPy
"""

import numpy as np
import sympy as sp
from scipy.interpolate import PPoly


# ── Algorithme 3, Étape 1 : calcul des z_i ──────────────────────────────────

def calculer_zi(x_points, y_points, z0=0.0):
    """
    Calcule les dérivées aux nœuds z_i par la relation de récurrence.

    Algorithme 3 – Étape 1 (cours) :
        z_0 = z0   (condition initiale : S'(x_0) = 0)
        pour i = 0 à n-1 :
            h_i      = x_{i+1} − x_i
            z_{i+1}  = 2·(y_{i+1} − y_i) / h_i − z_i

    Paramètres
    ----------
    x_points : array-like — abscisses [x_0, ..., x_n]
    y_points : array-like — ordonnées [y_0, ..., y_n]
    z0       : float      — condition initiale S'(x_0) (défaut 0)

    Retourne
    --------
    z : np.ndarray shape (n+1,) — dérivées aux nœuds
    """
    n = len(x_points) - 1
    z = np.zeros(n + 1)
    z[0] = z0
    for i in range(n):
        hi = x_points[i + 1] - x_points[i]
        z[i + 1] = 2.0 * (y_points[i + 1] - y_points[i]) / hi - z[i]
    return z


# ── Algorithme 4 : trouver l'intervalle contenant x ─────────────────────────

def position(x, x_points):
    """
    Trouve l'indice i tel que x ∈ [x_i, x_{i+1}].

    Algorithme 4 (cours) :
        i = 0
        tant que i < n-1 et x > x_{i+1} : i = i + 1
        retourner i

    Paramètres
    ----------
    x        : float      — point d'évaluation
    x_points : array-like — nœuds

    Retourne
    --------
    i : int — indice du morceau
    """
    n = len(x_points) - 1
    if x < x_points[0] or x > x_points[n]:
        raise ValueError(
            f"x = {x} hors de l'intervalle [{x_points[0]}, {x_points[n]}]"
        )
    i = 0
    while i < n - 1 and x > x_points[i + 1]:
        i += 1
    return i


# ── Algorithme 4 : évaluation de S(x) ───────────────────────────────────────

def evaluer_spline(x, x_points, y_points, z):
    """
    Évalue S(x) directement à partir des z_i, sans calculer a_i/b_i/c_i.

    Formule du cours (forme centrée) :
        S_i(x) = (z_{i+1} - z_i) / (2·h_i) · (x - x_i)²
                 + z_i · (x - x_i)
                 + y_i

    Algorithme 4 (cours) :
        trouver i tel que x ∈ [x_i, x_{i+1}]
        poser dx = x - x_i,  h_i = x_{i+1} - x_i
        S = (z_{i+1} - z_i) / (2·h_i) · dx² + z_i·dx + y_i
        retourner S

    Paramètres
    ----------
    x        : float ou array-like
    x_points : array-like — nœuds
    y_points : array-like — ordonnées aux nœuds
    z        : array-like — dérivées aux nœuds z_i = S'(x_i)

    Retourne
    --------
    S(x) : float ou np.ndarray
    """
    if isinstance(x, (list, np.ndarray)):
        return np.array([evaluer_spline(xi, x_points, y_points, z) for xi in x])
    i  = position(x, x_points)
    hi = x_points[i + 1] - x_points[i]
    dx = x - x_points[i]
    return (z[i + 1] - z[i]) / (2.0 * hi) * dx**2 + z[i] * dx + y_points[i]


# ── SymPy : calcul symbolique exact ─────────────────────────────────────────

def _to_sym(val):
    """Entier SymPy si val est entier, sinon Float 15 chiffres."""
    v = float(val)
    return sp.Integer(int(v)) if v == int(v) else sp.Float(v, 15)


def spline_sympy(x_points, y_points, z0=0):
    """
    Calcule symboliquement la spline quadratique.

    Suit exactement les Algorithmes 3 & 4 du cours avec SymPy.
    Forme globale : S_i(x) = a_i·x² + b_i·x + c_i

    Paramètres
    ----------
    x_points : liste de float
    y_points : liste de float
    z0       : condition initiale (défaut 0)

    Retourne
    --------
    morceaux    : list[(S_i_expr, x_i, x_{i+1})]  — polynômes par morceau
    z_sym       : list[sp.Expr]                    — dérivées aux nœuds (exacts)
    S_piecewise : sp.Piecewise                     — fonction complète S(x)
    """
    x = sp.Symbol('x')
    n = len(x_points) - 1

    xr = [_to_sym(xi) for xi in x_points]
    yr = [_to_sym(yi) for yi in y_points]

    # ── Étape 1 : z_i symboliques ──────────────────────────────────────────
    z = [None] * (n + 1)
    z[0] = _to_sym(z0)
    for i in range(n):
        hi      = xr[i + 1] - xr[i]
        z[i + 1] = sp.Integer(2) * (yr[i + 1] - yr[i]) / hi - z[i]

    # ── Étape 2 : S_i(x) directement en fonction de z_i, y_i, x_i ────────
    morceaux = []
    for i in range(n):
        hi = xr[i + 1] - xr[i]
        xi = xr[i]
        yi = yr[i]

        # Formule directe du cours (forme centrée) :
        # S_i(x) = (z_{i+1} - z_i)/(2·h_i)·(x - x_i)² + z_i·(x - x_i) + y_i
        Si = sp.expand(
            (z[i + 1] - z[i]) / (2 * hi) * (x - xi)**2
            + z[i] * (x - xi)
            + yi
        )
        morceaux.append((Si, xr[i], xr[i + 1]))

    # ── Construction Piecewise S(x) ────────────────────────────────────────
    pieces = []
    for i, (Si, xi, xi1) in enumerate(morceaux):
        cond = ((x >= xi) & (x < xi1)) if i < n - 1 else ((x >= xi) & (x <= xi1))
        pieces.append((Si, cond))
    S_piecewise = sp.Piecewise(*pieces)

    return morceaux, z, S_piecewise


# ── SciPy : PPoly ────────────────────────────────────────────────────────────

def spline_scipy(x_points, y_points, z0=0.0):
    """
    Construit la spline quadratique comme un scipy.interpolate.PPoly.

    Les z_i et a_i sont calculés selon l'algorithme du cours.
    PPoly évalue chaque morceau sous forme centrée :
        c[0,i]·(x−x_i)² + c[1,i]·(x−x_i) + c[2,i]
    avec  c[0,i] = a_i,  c[1,i] = z_i,  c[2,i] = y_i
    (forme centrée ≡ forme globale du cours).

    Paramètres
    ----------
    x_points : array-like
    y_points : array-like
    z0       : float — condition initiale (défaut 0)

    Retourne
    --------
    PPoly — objet scipy évaluable via ppoly(x)
    """
    x_arr = np.array(x_points, dtype=float)
    y_arr = np.array(y_points, dtype=float)
    n = len(x_arr) - 1

    z = calculer_zi(x_arr, y_arr, float(z0))

    a_coef = np.array([
        (z[i + 1] - z[i]) / (2.0 * (x_arr[i + 1] - x_arr[i]))
        for i in range(n)
    ])

    # PPoly forme centrée : [a_i, z_i, y_i]
    coeffs = np.array([a_coef, z[:-1], y_arr[:-1]])
    return PPoly(coeffs, x_arr)
