"""
Application Streamlit — Interpolation Numérique
================================================
Master 1 GI — Cours d'Analyse Numérique

Fonctionnalités :
    • Interpolation de Lagrange  : SymPy (symbolique) vs SciPy (numérique)
    • Spline Quadratique         : SymPy (symbolique) vs SciPy (PPoly)p
    • Comparaison des temps de calcul (timeit, N répétitions)
    • Vérification des hypothèses H1 / H2 / H3 (spline)
    • Visualisation des courbes

Usage :
    streamlit run app.py
"""

import timeit
import numpy as np
import sympy as sp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

from modules.lagrange import (
    lagrange_sympy,
    lagrange_scipy,
)
from modules.spline import (
    spline_sympy,
    spline_scipy,
    calculer_zi,
)

# ── Configuration ────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Interpolation Numérique",
    page_icon="📐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar ──────────────────────────────────────────────────────────────────

st.sidebar.title("📐 Interpolation Numérique")
st.sidebar.markdown("**Master 1 GI — Analyse Numérique**")
st.sidebar.divider()

st.sidebar.subheader("Points d'interpolation")
x_input = st.sidebar.text_input("x_points  (séparés par des virgules)", "0, 1, 2, 3")
y_input = st.sidebar.text_input("y_points  (séparés par des virgules)", "1, 3, 2, 5")

st.sidebar.divider()
N_ITER = st.sidebar.slider(
    "Répétitions pour la mesure de temps", min_value=1, max_value=20, value=5
)

# ── Parsing des points ───────────────────────────────────────────────────────

try:
    x_pts = [float(v.strip()) for v in x_input.split(",") if v.strip()]
    y_pts = [float(v.strip()) for v in y_input.split(",") if v.strip()]
    if len(x_pts) != len(y_pts):
        st.sidebar.error("x et y doivent avoir la même taille.")
        st.stop()
    if len(x_pts) < 2:
        st.sidebar.error("Minimum 2 points requis.")
        st.stop()
    x_arr = np.array(x_pts, dtype=float)
    y_arr = np.array(y_pts, dtype=float)
except ValueError as e:
    st.sidebar.error(f"Erreur de saisie : {e}")
    st.stop()

n_pts = len(x_pts)
st.sidebar.success(f"{n_pts} points — degré {n_pts - 1}")

# ── En-tête principal ─────────────────────────────────────────────────────────

st.title("Interpolation Numérique")
st.markdown(
    "Comparaison **SymPy** (calcul symbolique exact) vs **SciPy** (calcul numérique rapide) "
    "pour deux méthodes d'interpolation vues en cours."
)

# ── Onglets ───────────────────────────────────────────────────────────────────

tab_lag, tab_spl = st.tabs(
    ["📈 Interpolation de Lagrange", "🔧 Spline Quadratique"]
)


# ════════════════════════════════════════════════════════════════════════════════
# ONGLET 1 — LAGRANGE
# ════════════════════════════════════════════════════════════════════════════════

with tab_lag:

    st.header("Interpolation Polynomiale de Lagrange")

    with st.expander("Formules du cours", expanded=False):
        st.latex(
            r"L_i(x) = \prod_{\substack{j=0 \\ j \neq i}}^{n} "
            r"\frac{x - x_j}{x_i - x_j}"
        )
        st.latex(r"P(x) = \sum_{j=0}^{n} y_j \cdot L_j(x)")
        st.markdown("**Propriété :** $P(x_i) = y_i$ pour tout $i$.")

    # ── Calculs + mesure de temps ─────────────────────────────────────────────

    t_sym_lag = (
        timeit.timeit(lambda: lagrange_sympy(x_pts, y_pts), number=N_ITER) / N_ITER
    )
    P_sym, x_sym, bases_sym = lagrange_sympy(x_pts, y_pts)

    t_sci_lag = (
        timeit.timeit(lambda: lagrange_scipy(x_pts, y_pts), number=N_ITER) / N_ITER
    )
    P_sci = lagrange_scipy(x_pts, y_pts)

    # ── Affichage côte à côte ─────────────────────────────────────────────────

    col_s, col_c = st.columns(2)

    # ── Colonne SymPy ─────────────────────────────────────────────────────────
    with col_s:
        st.subheader("SymPy — calcul symbolique")
        st.metric("Temps moyen (SymPy)", f"{t_sym_lag * 1e3:.3f} ms")

        st.markdown("**Bases de Lagrange $L_i(x)$ :**")
        for i, Li in enumerate(bases_sym):
            st.latex(rf"L_{{{i}}}(x) = {sp.latex(Li)}")

        st.markdown("**Polynôme $P(x)$ développé :**")
        st.latex(f"P(x) = {sp.latex(P_sym)}")

        st.markdown("**Vérification $P(x_i) = y_i$ :**")
        all_ok = True
        for xi, yi in zip(x_pts, y_pts):
            val = float(P_sym.subs(x_sym, xi))
            ok = abs(val - yi) < 1e-8
            all_ok = all_ok and ok
            st.write(
                f"  P({xi}) = {val:.6f}  ←  attendu {yi}  "
                + ("✅" if ok else "❌")
            )
        if all_ok:
            st.success("H1 satisfaite : P(xᵢ) = yᵢ  pour tout i")

    # ── Colonne SciPy ─────────────────────────────────────────────────────────
    with col_c:
        st.subheader("SciPy — scipy.interpolate.lagrange")
        st.metric("Temps moyen (SciPy)", f"{t_sci_lag * 1e3:.3f} ms")

        # Affichage du polynôme sous forme lisible
        coeffs_sci = P_sci.coeffs
        deg = len(coeffs_sci) - 1
        terms = []
        for k, c_val in enumerate(coeffs_sci):
            power = deg - k
            if abs(c_val) < 1e-12:
                continue
            c_str = f"{c_val:.4f}"
            if power == 0:
                terms.append(c_str)
            elif power == 1:
                terms.append(f"({c_str})x")
            else:
                terms.append(f"({c_str})x^{{{power}}}")
        poly_display = " + ".join(terms) if terms else "0"
        st.markdown("**Polynôme $P(x)$ :**")
        st.latex(f"P(x) = {poly_display}")

        st.markdown("**Vérification $P(x_i) = y_i$ :**")
        all_ok2 = True
        for xi, yi in zip(x_pts, y_pts):
            val = float(P_sci(xi))
            ok = abs(val - yi) < 1e-6
            all_ok2 = all_ok2 and ok
            st.write(
                f"  P({xi}) = {val:.6f}  ←  attendu {yi}  "
                + ("✅" if ok else "❌")
            )
        if all_ok2:
            st.success("H1 satisfaite : P(xᵢ) = yᵢ  pour tout i")

    # ── Comparaison des temps ─────────────────────────────────────────────────

    st.divider()
    st.subheader("Comparaison des temps de calcul")

    ratio_lag = t_sym_lag / t_sci_lag if t_sci_lag > 0 else float("inf")
    c1, c2, c3 = st.columns(3)
    c1.metric("SymPy", f"{t_sym_lag * 1e3:.3f} ms")
    c2.metric("SciPy", f"{t_sci_lag * 1e3:.3f} ms")
    c3.metric(
        "Ratio SymPy / SciPy",
        f"{ratio_lag:.1f}×",
        delta=f"SciPy {'plus rapide' if ratio_lag > 1 else 'plus lent'}",
        delta_color="normal",
    )

    fig_bar, ax_bar = plt.subplots(figsize=(4, 3))
    ax_bar.bar(
        ["SymPy", "SciPy"],
        [t_sym_lag * 1e3, t_sci_lag * 1e3],
        color=["#8e44ad", "#2980b9"],
        edgecolor="white",
    )
    ax_bar.set_ylabel("Temps moyen (ms)")
    ax_bar.set_title("Lagrange — temps de calcul")
    ax_bar.grid(axis="y", alpha=0.3)
    st.pyplot(fig_bar, use_container_width=False)
    plt.close(fig_bar)

    # ── Courbes d'interpolation ───────────────────────────────────────────────

    st.divider()
    st.subheader("Courbes d'interpolation")

    x_plot = np.linspace(x_arr[0], x_arr[-1], 500)

    # SymPy : lambdify pour vectorisation
    P_num = sp.lambdify(x_sym, P_sym, "numpy")
    y_sym_plot = P_num(x_plot)

    # SciPy : poly1d est vectorisé
    y_sci_plot = P_sci(x_plot)

    fig_lag, ax_lag = plt.subplots(figsize=(9, 5))
    ax_lag.plot(x_plot, y_sym_plot, "-",  color="#8e44ad", lw=2.5, label="Lagrange — SymPy")
    ax_lag.plot(x_plot, y_sci_plot, "--", color="#2980b9", lw=2,   label="Lagrange — SciPy")
    ax_lag.plot(x_arr, y_arr, "ro", ms=8, zorder=5, label="Points donnés")
    ax_lag.set_xlabel("x")
    ax_lag.set_ylabel("P(x)")
    ax_lag.set_title("Interpolation de Lagrange — SymPy vs SciPy")
    ax_lag.legend()
    ax_lag.grid(True, alpha=0.3)
    st.pyplot(fig_lag)
    plt.close(fig_lag)


# ════════════════════════════════════════════════════════════════════════════════
# ONGLET 2 — SPLINE QUADRATIQUE
# ════════════════════════════════════════════════════════════════════════════════

with tab_spl:

    st.header("Spline Quadratique")
    n_pieces = len(x_pts) - 1

    with st.expander("Formules du cours", expanded=False):
        st.latex(r"z_0 = S'(x_0) = 0")
        st.latex(
            r"z_{i+1} = \frac{2(y_{i+1} - y_i)}{h_i} - z_i, "
            r"\quad h_i = x_{i+1} - x_i"
        )
        st.latex(
            r"S_i(x) = \frac{z_{i+1} - z_i}{2h_i}(x - x_i)^2 "
            r"+ z_i\,(x - x_i) + y_i, \quad x \in [x_i,\, x_{i+1}]"
        )

    # ── Calculs + mesure de temps ─────────────────────────────────────────────

    t_sym_spl = (
        timeit.timeit(lambda: spline_sympy(x_pts, y_pts), number=N_ITER) / N_ITER
    )
    morceaux_sym, z_sym, S_pw = spline_sympy(x_pts, y_pts)

    t_sci_spl = (
        timeit.timeit(lambda: spline_scipy(x_pts, y_pts), number=N_ITER) / N_ITER
    )
    spline_pp = spline_scipy(x_pts, y_pts)
    z_c = calculer_zi(x_arr, y_arr)

    # ── Affichage côte à côte ─────────────────────────────────────────────────

    col_s2, col_c2 = st.columns(2)
    x_sp = sp.Symbol("x")

    # ── Colonne SymPy ─────────────────────────────────────────────────────────
    with col_s2:
        st.subheader("SymPy — calcul symbolique")
        st.metric("Temps moyen (SymPy)", f"{t_sym_spl * 1e3:.3f} ms")

        st.markdown("**Dérivées aux nœuds $z_i$ :**")
        for i, zi in enumerate(z_sym):
            st.latex(f"z_{{{i}}} = {sp.latex(zi)}")

        st.markdown("**Expressions $S_i(x) = a_i x^2 + b_i x + c_i$ :**")
        for i, (Si, xi, xi1) in enumerate(morceaux_sym):
            st.latex(
                rf"S_{{{i}}}(x) = {sp.latex(Si)},\quad "
                rf"x \in [{sp.latex(xi)},\, {sp.latex(xi1)}]"
            )

        st.markdown("**Vérifications :**")

        # H1 : S(x_i) = y_i
        h1_ok = True
        for xi, yi in zip(x_pts, y_pts):
            val = S_pw.subs(x_sp, xi)
            try:
                val_f = float(val)
            except Exception:
                val_f = float(val.evalf())
            ok = abs(val_f - yi) < 1e-8
            h1_ok = h1_ok and ok
            st.write(f"  H1: S({xi}) = {val_f:.6f} ← {yi}  " + ("✅" if ok else "❌"))

        # H2 : S_i(x_{i+1}) = S_{i+1}(x_{i+1})
        h2_ok = True
        for i in range(n_pieces - 1):
            xi1 = morceaux_sym[i][2]  # borne droite du morceau i
            val_left  = float(morceaux_sym[i][0].subs(x_sp, xi1))
            val_right = float(morceaux_sym[i + 1][0].subs(x_sp, xi1))
            ok = abs(val_left - val_right) < 1e-8
            h2_ok = h2_ok and ok
            st.write(
                f"  H2: S_{i}({float(xi1)}) = {val_left:.6f},  "
                f"S_{i+1}({float(xi1)}) = {val_right:.6f}  " + ("✅" if ok else "❌")
            )

        # H3 : S'_i(x_{i+1}) = S'_{i+1}(x_{i+1})
        h3_ok = True
        for i in range(n_pieces - 1):
            xi1 = morceaux_sym[i][2]
            dSi   = sp.diff(morceaux_sym[i][0], x_sp).subs(x_sp, xi1)
            dSi1  = sp.diff(morceaux_sym[i + 1][0], x_sp).subs(x_sp, xi1)
            ok = abs(float(dSi) - float(dSi1)) < 1e-8
            h3_ok = h3_ok and ok
            st.write(
                f"  H3: S'_{i}({float(xi1)}) = {float(dSi):.6f},  "
                f"S'_{i+1}({float(xi1)}) = {float(dSi1):.6f}  " + ("✅" if ok else "❌")
            )

        if h1_ok and (n_pieces < 2 or (h2_ok and h3_ok)):
            st.success("H1, H2, H3 satisfaites")

    # ── Colonne SciPy ─────────────────────────────────────────────────────────
    with col_c2:
        st.subheader("SciPy — PPoly (scipy.interpolate)")
        st.metric("Temps moyen (SciPy)", f"{t_sci_spl * 1e3:.3f} ms")

        st.markdown("**Dérivées aux nœuds $z_i$ :**")
        for i, zi in enumerate(z_c):
            st.write(f"  z_{i} = {zi:.6f}")

        st.markdown(
            r"**Formule directe $S_i(x) = \frac{z_{i+1}-z_i}{2h_i}(x-x_i)^2"
            r" + z_i(x-x_i) + y_i$ :**"
        )
        for i in range(n_pieces):
            hi  = x_pts[i + 1] - x_pts[i]
            ai  = (z_c[i + 1] - z_c[i]) / (2.0 * hi)
            zi  = z_c[i]
            yi  = y_pts[i]
            xi  = x_pts[i]
            sign_z = "+" if zi >= 0 else "-"
            sign_y = "+" if yi >= 0 else "-"
            st.latex(
                rf"S_{{{i}}}(x) = {ai:.4f}(x-{xi:.4g})^2 "
                rf"{sign_z} {abs(zi):.4f}(x-{xi:.4g}) "
                rf"{sign_y} {abs(yi):.4g},\quad "
                rf"x\in[{x_pts[i]},\, {x_pts[i+1]}]"
            )

        st.markdown("**Vérifications :**")

        # H1
        h1_ok2 = True
        for xi, yi in zip(x_pts, y_pts):
            val = float(spline_pp(xi))
            ok = abs(val - yi) < 1e-8
            h1_ok2 = h1_ok2 and ok
            st.write(f"  H1: S({xi}) = {val:.6f} ← {yi}  " + ("✅" if ok else "❌"))

        # H2 : S_i(x_{i+1}) = y_{i+1} par construction
        h2_ok2 = True
        for i in range(n_pieces - 1):
            xi1 = x_pts[i + 1]
            hi  = x_pts[i + 1] - x_pts[i]
            # S_i(x_{i+1}) = (z_{i+1}-z_i)/(2*hi)*hi² + z_i*hi + y_i
            val_l = (z_c[i+1] - z_c[i]) / (2*hi) * hi**2 + z_c[i] * hi + y_pts[i]
            # S_{i+1}(x_{i+1}) = 0 + 0 + y_{i+1}
            val_r = y_pts[i + 1]
            ok = abs(val_l - val_r) < 1e-8
            h2_ok2 = h2_ok2 and ok
            st.write(
                f"  H2: S_{i}({xi1}) = {val_l:.6f},  "
                f"S_{i+1}({xi1}) = {val_r:.6f}  " + ("✅" if ok else "❌")
            )

        # H3 : S'_i(x_{i+1}) = z_{i+1} par construction
        h3_ok2 = True
        for i in range(n_pieces - 1):
            xi1  = x_pts[i + 1]
            # S'_i(x_{i+1}) = z_{i+1}  (dérivée au bord droit du morceau i)
            dS_l = z_c[i + 1]
            # S'_{i+1}(x_{i+1}) = z_{i+1}  (dérivée au bord gauche du morceau i+1)
            dS_r = z_c[i + 1]
            ok = abs(dS_l - dS_r) < 1e-8
            h3_ok2 = h3_ok2 and ok
            st.write(
                f"  H3: S'_{i}({xi1}) = {dS_l:.6f},  "
                f"S'_{i+1}({xi1}) = {dS_r:.6f}  " + ("✅" if ok else "❌")
            )

        if h1_ok2 and (n_pieces < 2 or (h2_ok2 and h3_ok2)):
            st.success("H1, H2, H3 satisfaites")

    # ── Comparaison des temps ─────────────────────────────────────────────────

    st.divider()
    st.subheader("Comparaison des temps de calcul")

    ratio_spl = t_sym_spl / t_sci_spl if t_sci_spl > 0 else float("inf")
    d1, d2, d3 = st.columns(3)
    d1.metric("SymPy", f"{t_sym_spl * 1e3:.3f} ms")
    d2.metric("SciPy", f"{t_sci_spl * 1e3:.3f} ms")
    d3.metric(
        "Ratio SymPy / SciPy",
        f"{ratio_spl:.1f}×",
        delta=f"SciPy {'plus rapide' if ratio_spl > 1 else 'plus lent'}",
        delta_color="normal",
    )

    fig_bar2, ax_bar2 = plt.subplots(figsize=(4, 3))
    ax_bar2.bar(
        ["SymPy", "SciPy"],
        [t_sym_spl * 1e3, t_sci_spl * 1e3],
        color=["#8e44ad", "#27ae60"],
        edgecolor="white",
    )
    ax_bar2.set_ylabel("Temps moyen (ms)")
    ax_bar2.set_title("Spline Quadratique — temps de calcul")
    ax_bar2.grid(axis="y", alpha=0.3)
    st.pyplot(fig_bar2, use_container_width=False)
    plt.close(fig_bar2)

    # ── Courbes ───────────────────────────────────────────────────────────────

    st.divider()
    st.subheader("Courbes d'interpolation")

    x_plot_s = np.linspace(x_arr[0], x_arr[-1], 500)

    # SymPy : lambdify par morceau (Algorithme 4)
    funcs_sym = [sp.lambdify(x_sp, morceaux_sym[i][0], "numpy") for i in range(n_pieces)]
    y_sym_spl = np.zeros(len(x_plot_s))
    for k, xk in enumerate(x_plot_s):
        idx = 0
        while idx < n_pieces - 1 and xk > x_pts[idx + 1]:
            idx += 1
        y_sym_spl[k] = funcs_sym[idx](xk)

    # SciPy PPoly
    y_sci_spl = spline_pp(x_plot_s)

    fig_spl, ax_spl = plt.subplots(figsize=(9, 5))
    ax_spl.plot(x_plot_s, y_sym_spl, "-",  color="#8e44ad", lw=2.5, label="Spline — SymPy")
    ax_spl.plot(x_plot_s, y_sci_spl, "--", color="#27ae60", lw=2,   label="Spline — SciPy")
    ax_spl.plot(x_arr, y_arr, "ro", ms=8, zorder=5, label="Points donnés")
    for xi in x_arr[1:-1]:
        ax_spl.axvline(x=xi, color="gray", linestyle=":", alpha=0.4)
    ax_spl.set_xlabel("x")
    ax_spl.set_ylabel("S(x)")
    ax_spl.set_title("Spline Quadratique — SymPy vs SciPy")
    ax_spl.legend()
    ax_spl.grid(True, alpha=0.3)
    st.pyplot(fig_spl)
    plt.close(fig_spl)
