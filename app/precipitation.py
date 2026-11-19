"""
Precipitation & Solubility tab: microalloy precipitate thermodynamics for the current schedule.

Published austenite solubility products (log10 of the wt%-product), a 1:1-molar precipitation
balance, a sequential nitride-then-carbide engineering model for the undissolved Nb/Ti/V/Al
fraction, a C-N-Me ternary stability map and Zener pinning / pancaked-austenite screening
indicators.

Caveat carried through to the UI: these are paraequilibrium screening estimates from published
wt%-product fits (literature spread of roughly +-200 K on solution temperatures), not
Thermo-Calc calls.
"""

import numpy as np
import plotly.graph_objects as go
import streamlit as st

# ----------------------------------------------------------------------------- constants
ATOMIC_MASS = {"Nb": 92.906, "Ti": 47.867, "V": 50.942, "Al": 26.982,
               "C": 12.011, "N": 14.007}

# Austenite solubility products, log10([Me][X]) = A - B/T with T in Kelvin and both
# concentrations in wt%. Sources as cited per entry.
SOLUBILITY = {
    "NbC": {"A": 3.42, "B": 7900.0, "metal": "Chem_Nb", "inter": "Chem_C",
            "label": "NbC", "source": "Irvine et al., 1967"},
    "NbN": {"A": 3.70, "B": 10800.0, "metal": "Chem_Nb", "inter": "Chem_N",
            "label": "NbN", "source": "Irvine et al., 1967"},
    "TiC": {"A": 4.40, "B": 10475.0, "metal": "Chem_Ti", "inter": "Chem_C",
            "label": "TiC", "source": "austenite fit, cf. Wada & Pehlke family"},
    "TiN": {"A": 0.32, "B": 7780.0, "metal": "Chem_Ti", "inter": "Chem_N",
            "label": "TiN", "source": "Wada & Pehlke, 1977"},
    "VC": {"A": 6.72, "B": 9500.0, "metal": "Chem_V", "inter": "Chem_C",
           "label": "VC", "source": "Irvine et al., 1967"},
    "VN": {"A": 3.63, "B": 8250.0, "metal": "Chem_V", "inter": "Chem_N",
           "label": "VN", "source": "Irvine et al., 1967"},
    "AlN": {"A": 1.03, "B": 6770.0, "metal": "Chem_Al", "inter": "Chem_N",
            "label": "AlN", "source": "austenite fit, cf. Davern & Edmonds family"},
}

# Per metal, its precipitate systems in precipitation order: nitrides (esp. TiN) are far
# stabler and precipitate first, so the sequential nitride-then-carbide order is the standard
# engineering approximation for the coupled equilibrium.
METAL_SYSTEMS = {
    "Nb": ("NbN", "NbC"),
    "Ti": ("TiN", "TiC"),
    "V": ("VN", "VC"),
    "Al": ("AlN",),  # aluminium carbide is not stable in austenite; AlN only
}

RHO_MATRIX = 7.87   # g/cm3, austenite
RHO_PPT = {"NbC": 7.83, "NbN": 8.40, "TiN": 5.40, "TiC": 4.90,
           "VC": 5.70, "VN": 6.10, "AlN": 3.26}  # g/cm3
GAMMA_GB = 0.5      # J/m2, austenite grain-boundary energy

# wt% axis maxima used to scale the ternary simplex
TERNARY_MAX = {"C": 0.35, "N": 0.02, "Nb": 0.09, "Ti": 0.06, "V": 0.12}

INK, HAIR, GRID = "#0b0d10", "#e5e7eb", "#f1f2f4"
ACCENT = "#1746c8"   # GRP brand blue, same token as the app stylesheet


def _shade(hex_color, factor):
    """Lighter (factor > 0, toward white) or darker (factor < 0, toward black) step of a hex."""
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
    if factor >= 0:
        r, g, b = (round(c + (255 - c) * factor) for c in (r, g, b))
    else:
        r, g, b = (round(c * (1 + factor)) for c in (r, g, b))
    return "#{:02x}{:02x}{:02x}".format(r, g, b)


ACCENT_DARK = _shade(ACCENT, -0.45)
# Semantic state tokens, matching the pill colours used across the app.
SEMANTIC_OK = "#065f46"
SEMANTIC_WARN = "#92400e"
SEMANTIC_BAD = "#991b1b"
WARN_LINE = "#b45309"
# Categorical series palette anchored on the accent plus the semantic counter-hues
# (the former violet/pink picks are gone); order matters for the solubility chart.
SERIES = [ACCENT, SEMANTIC_BAD, SEMANTIC_OK, WARN_LINE, ACCENT_DARK, SEMANTIC_WARN,
          _shade(ACCENT, 0.35)]
METAL_COLOR = {"Nb": SERIES[0], "Ti": SERIES[1], "V": SERIES[2], "Al": SERIES[3]}


# ----------------------------------------------------------------------------- pure model
def _stoich_ratio(sysd):
    """Interstitial-to-metal mass ratio r = X_atmass / M_atmass for a precipitate system."""
    return ATOMIC_MASS[sysd["inter"][len("Chem_"):]] / ATOMIC_MASS[sysd["metal"][len("Chem_"):]]


def solution_temperature(M0, X0, A, B):
    """Temperature (K) at which the system just fully dissolves; None when unsolvable.

    T_sol = B / (A - log10(M0*X0)). Empty chemistry (P <= 0) or a product so large that
    log10(P) >= A (no finite positive solution) returns None.
    """
    try:
        P = float(M0) * float(X0)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(P) or P <= 0.0:
        return None
    denom = float(A) - np.log10(P)
    if not np.isfinite(denom) or denom <= 0.0:
        return None
    return float(B) / denom


def _precipitated(M0, X0, K, r):
    """Precipitated metal s (wt%) of the 1:1-molar balance; broadcasts over arrays.

    With totals (M0, X0), stoichiometric ratio r and solubility product K, the dissolved
    product must satisfy (M0 - s)(X0 - r*s) = K, i.e.
        r*s^2 - (r*M0 + X0)*s + (M0*X0 - K) = 0.
    The physical root lies in [0, min(M0, X0/r)] (the dissolved product decreases
    monotonically from M0*X0 to 0 over that interval, so the root is unique). It is taken in
    the numerically stable rationalised form s = 2c / (a + sqrt(a^2 - 4rc)) to avoid the
    catastrophic cancellation of (a - sqrt(D)) near T_sol where K -> M0*X0.
    """
    M0 = np.asarray(M0, dtype=float)
    X0 = np.asarray(X0, dtype=float)
    K = np.asarray(K, dtype=float)
    r = float(r)
    s_max = np.minimum(M0, X0 / r)
    a = r * M0 + X0
    c = M0 * X0 - K  # <= 0 means the steel can hold everything dissolved
    # (rM - X)^2 >= 0 guarantees a non-negative discriminant whenever c > 0
    disc = a * a - 4.0 * r * np.maximum(c, 0.0)
    denom = a + np.sqrt(np.maximum(disc, 0.0))
    s = np.where(c > 0.0, 2.0 * np.maximum(c, 0.0) / np.maximum(denom, 1e-300), 0.0)
    s = np.clip(s, 0.0, s_max)
    if np.ndim(s) == 0:
        return float(s)
    return s


def system_undissolved(label, M0, X0, T_C):
    """Undissolved metal fraction [0, 1] for one precipitate system's 1:1 balance."""
    sysd = SOLUBILITY[label]
    T = np.asarray(T_C, dtype=float) + 273.15
    K = 10.0 ** (sysd["A"] - sysd["B"] / T)
    s = _precipitated(M0, X0, K, _stoich_ratio(sysd))
    M0a = np.asarray(M0, dtype=float)
    frac = np.where(M0a > 0.0, s / np.where(M0a > 0.0, M0a, 1.0), 0.0)
    frac = np.clip(frac, 0.0, 1.0)
    if np.ndim(frac) == 0:
        return float(frac)
    return frac


def sequential_precipitated(metal, T_C, M0, C0, N0):
    """wt% of metal precipitated per system, sequential nitride -> carbide (broadcasts).

    Engineering approximation: the far stabler nitride takes its share of the metal first,
    then the remainder is offered to the carbide. K <= 0 / empty chemistry degrade to zero
    precipitation through _precipitated's guards.
    """
    M0 = np.asarray(M0, dtype=float)
    C0 = np.asarray(C0, dtype=float)
    N0 = np.asarray(N0, dtype=float)
    T = np.asarray(T_C, dtype=float) + 273.15
    out = {}
    remaining = M0
    for label in METAL_SYSTEMS.get(metal, ()):
        sysd = SOLUBILITY[label]
        X0 = N0 if sysd["inter"].endswith("_N") else C0
        K = 10.0 ** (sysd["A"] - sysd["B"] / T)
        s = _precipitated(remaining, X0, K, _stoich_ratio(sysd))
        out[label] = s
        remaining = remaining - s
    return out


def undissolved_fraction(metal, T_C, M0, C0, N0):
    """Undissolved metal fraction [0, 1] under the sequential coupled model."""
    s_map = sequential_precipitated(metal, T_C, M0, C0, N0)
    total = sum(s_map.values()) if s_map else 0.0
    M0a = np.asarray(M0, dtype=float)
    frac = np.where(M0a > 0.0, np.asarray(total, dtype=float) / np.where(M0a > 0.0, M0a, 1.0), 0.0)
    frac = np.clip(frac, 0.0, 1.0)
    if np.ndim(frac) == 0:
        return float(frac)
    return frac


def precipitate_mass_fractions(metal, T_C, M0, C0, N0):
    """Steel mass fraction tied up in undissolved precipitate, per system (scalar input)."""
    s_map = sequential_precipitated(metal, T_C, M0, C0, N0)
    out = {}
    for label, s in s_map.items():
        sysd = SOLUBILITY[label]
        out[label] = float(np.asarray(s, dtype=float)) * (1.0 + _stoich_ratio(sysd)) / 100.0
    return out


def volume_fraction(f_mass, rho_ppt):
    """Precipitate volume fraction from its mass fraction and density."""
    return float(f_mass) * RHO_MATRIX / float(rho_ppt)


def zener_pinning_mpa(f_v, r_nm, gamma=GAMMA_GB):
    """Zener pinning pressure P_z = 3 f_v gamma / (2 r) in MPa.

    Clamps f_v into [0, 1] and the radius to at least 1 nm so degenerate slider input can
    never divide by zero or go negative.
    """
    f_v = min(max(float(f_v), 0.0), 1.0)
    r_m = max(float(r_nm), 1.0) * 1e-9
    return 3.0 * f_v * float(gamma) / (2.0 * r_m) / 1.0e6


# ----------------------------------------------------------------------------- helpers
def _num(raw, key, default):
    """Defensive float read: missing, non-numeric or NaN input falls back, never raises."""
    try:
        v = float(raw.get(key, default))
    except (TypeError, ValueError):
        return float(default)
    return v if np.isfinite(v) else float(default)


def _chem(raw):
    return {el: _num(raw, f"Chem_{el}", 0.0) for el in ("C", "N", "Nb", "Ti", "V", "Al")}


def _route_temps(mill, raw):
    """Entry / FRT / exit temperatures of the current schedule, per route."""
    if mill == "Plate":
        entry = _num(raw, "FM_Temp_Before_Rolling", 985.0)
        frt = _num(raw, "FM_Finish_Temp", 832.0)
        exit_t = _num(raw, "ACC_Temp_Post_Laminar", 744.0)
        exit_lbl = "ACC exit"
    else:
        entry = _num(raw, "Temp_Transfer_Bar_C", 937.0)
        frt = _num(raw, "Temp_FT_Avg_C", 835.0)
        exit_t = (_num(raw, "Temp_CT_Top_C", 595.0) + _num(raw, "Temp_CT_Bot_C", 558.0)) / 2.0
        exit_lbl = "coiling"
    return {"entry": entry, "frt": frt, "exit": exit_t, "exit_label": exit_lbl}


def _section(label):
    st.markdown(f'<div class="section"><span>{label}</span><i></i></div>', unsafe_allow_html=True)


def _base_fig(title, x_title, y_title, height=420):
    fig = go.Figure()
    fig.update_layout(
        title=dict(text=title, font=dict(size=12.5), x=0, xanchor="left"),
        plot_bgcolor="#ffffff", paper_bgcolor="#ffffff",
        margin=dict(l=52, r=18, t=44, b=44),
        font=dict(family="Inter, -apple-system, 'Segoe UI', sans-serif", size=11, color=INK),
        height=height,
    )
    fig.update_xaxes(title_text=x_title, linecolor=HAIR, gridcolor=GRID, zeroline=False)
    fig.update_yaxes(title_text=y_title, linecolor=HAIR, gridcolor=GRID, zeroline=False)
    return fig


# ----------------------------------------------------------------------------- charts
def _solubility_fig(chem, frt):
    """Chart 1: log K vs reciprocal temperature, chemistry levels and the FRT marker."""
    T = np.linspace(1100.0, 1550.0, 220)
    x = 1000.0 / T
    fig = _base_fig("Solubility products: log₁₀[M][X] vs 1000/T",
                    "1000/T (1/K)", "log₁₀ ([Me]·[X], wt%²)")
    for i, (label, sysd) in enumerate(SOLUBILITY.items()):
        fig.add_scatter(x=x, y=sysd["A"] - sysd["B"] / T, mode="lines", name=label,
                        line=dict(color=SERIES[i], width=2.0))
    x_frt = 1000.0 / (frt + 273.15)
    fig.add_vline(x=x_frt, line=dict(color=INK, width=1.3, dash="dot"))
    fig.add_annotation(x=x_frt, y=1.0, yref="paper", showarrow=False,
                       text=f"FRT {frt:.0f} °C", font=dict(size=10, color=INK),
                       xanchor="right", yanchor="top")
    for i, (label, sysd) in enumerate(SOLUBILITY.items()):
        m_el, x_el = sysd["metal"][5:], sysd["inter"][5:]
        M0, X0 = chem[m_el], chem[x_el]
        if M0 > 0.0 and X0 > 0.0:
            y0 = float(np.log10(M0 * X0))
            fig.add_scatter(x=[x[0], x[-1]], y=[y0, y0], mode="lines",
                            name=f"{label} level log₁₀({m_el}·{x_el})",
                            line=dict(color=SERIES[i], width=1.2, dash="dash"))
    fig.update_xaxes(range=[min(float(x.min()), x_frt) - 0.02,
                            max(float(x.max()), x_frt) + 0.02])
    fig.update_layout(legend=dict(orientation="h", y=-0.32, x=0, font=dict(size=9)))
    return fig


def _undissolved_fig(chem, temps):
    """Chart 2: sequential-model undissolved fraction per metal over the rolling range."""
    t_lo = max(400.0, min(800.0, temps["entry"], temps["frt"], temps["exit"]) - 15.0)
    Tc = np.linspace(t_lo, 1400.0, 301)
    fig = _base_fig("Undissolved Nb / Ti / V / Al vs temperature",
                    "Temperature (°C)", "Undissolved fraction")
    for metal in ("Nb", "Ti", "V", "Al"):
        frac = undissolved_fraction(metal, Tc, chem[metal], chem["C"], chem["N"])
        fig.add_scatter(x=Tc, y=frac, mode="lines", name=metal,
                        line=dict(color=METAL_COLOR[metal], width=2.0))
        f_frt = undissolved_fraction(metal, temps["frt"], chem[metal], chem["C"], chem["N"])
        fig.add_scatter(x=[temps["frt"]], y=[f_frt], mode="markers", showlegend=False,
                        marker=dict(color=METAL_COLOR[metal], size=8, line=dict(width=1, color="#ffffff")))
    for t_val, lbl, dash in ((temps["entry"], "entry", "dot"),
                             (temps["frt"], "FRT", "dash"),
                             (temps["exit"], temps["exit_label"], "dot")):
        if Tc[0] <= t_val <= Tc[-1]:
            fig.add_vline(x=t_val, line=dict(color="#9ca3af", width=1.1, dash=dash),
                          annotation_text=lbl, annotation_position="top",
                          annotation_font_size=9)
    fig.update_yaxes(range=[0.0, 1.02])
    return fig


def _ternary_fig(metal, T_C, chem):
    """Chart 4: C-N-Me ternary stability map, colour = undissolved Me fraction."""
    m_max = TERNARY_MAX[metal]
    a = np.linspace(0.0, 1.0, 41)
    A, B = np.meshgrid(a, a)
    keep = A + B <= 1.0 + 1e-9
    a_n, b_n = A[keep], B[keep]
    c_n = 1.0 - a_n - b_n
    me_wt = a_n * m_max
    c_wt = b_n * TERNARY_MAX["C"]
    n_wt = c_n * TERNARY_MAX["N"]
    frac = undissolved_fraction(metal, float(T_C), me_wt, c_wt, n_wt)

    fig = go.Figure(go.Scatterternary(
        a=a_n, b=b_n, c=c_n, mode="markers",
        customdata=np.stack([me_wt, c_wt, n_wt, frac], axis=-1),
        marker=dict(color=frac, colorscale="Viridis", cmin=0.0, cmax=1.0, size=5,
                    line=dict(width=0),
                    colorbar=dict(title=dict(text="undissolved", font=dict(size=10)),
                                  tickformat=".0%", tickfont=dict(size=9))),
        hovertemplate=("Me %{customdata[0]:.4f} wt% · C %{customdata[1]:.4f} wt% · "
                       "N %{customdata[2]:.4f} wt%<br>undissolved %{customdata[3]:.1%}"
                       "<extra></extra>"),
    ))
    fig.update_layout(
        title=dict(text=f"{metal}–C–N stability map at {T_C:.0f} °C "
                        "(colour: undissolved metal fraction)",
                   font=dict(size=12.5), x=0, xanchor="left"),
        paper_bgcolor="#ffffff",
        margin=dict(l=60, r=20, t=48, b=52),
        font=dict(family="Inter, -apple-system, 'Segoe UI', sans-serif", size=11, color=INK),
        height=520,
    )
    def _axis(title, w_max, color):
        vals = [0.0, 0.25, 0.5, 0.75, 1.0]
        return dict(title=title, min=0.0, color=color, showgrid=True, gridcolor=GRID,
                    tickvals=vals, ticktext=[f"{v * w_max:.3f}" for v in vals])
    fig.update_ternaries(
        bgcolor="#ffffff",
        aaxis=_axis(f"{metal} (wt%)", m_max, SERIES[0]),
        baxis=_axis("C (wt%)", TERNARY_MAX["C"], "#6b7280"),
        caxis=_axis("N (wt%)", TERNARY_MAX["N"], "#6b7280"),
    )
    return fig


def _zener_fig(f_v, drive_mpa):
    """Chart 5b: Zener pinning pressure vs mean precipitate radius, threshold line."""
    r = np.linspace(5.0, 100.0, 200)
    pz = 3.0 * f_v * GAMMA_GB / (2.0 * r * 1e-9) / 1.0e6
    fig = _base_fig("Zener pinning pressure vs precipitate radius",
                    "Mean precipitate radius (nm)", "P_z (MPa)", height=360)
    fig.add_scatter(x=r, y=pz, mode="lines", name="P_z",
                    line=dict(color=SERIES[0], width=2.0))
    fig.add_hline(y=float(drive_mpa), line=dict(color=SERIES[1], width=1.3, dash="dash"),
                  annotation_text=f"driving pressure {drive_mpa:.1f} MPa",
                  annotation_position="top right", annotation_font_size=9)
    fig.update_yaxes(type="log")
    return fig


# ----------------------------------------------------------------------------- view
def view_precipitation(mill, raw):
    """Precipitation & Solubility tab: module-owned view, called from main()."""
    chem = _chem(raw)
    temps = _route_temps(mill, raw)

    _section("precipitation & solubility")
    st.markdown(
        '<div class="note">Paraequilibrium screening of Nb, Ti, V and Al precipitates at this '
        'schedule\'s temperatures: not Thermo-Calc. Chemistry comes from the sidebar; edit it '
        'there.</div>', unsafe_allow_html=True)

    # --- facts card: per-system solution temperature and status at FRT -------------------
    cells = []
    for label, sysd in SOLUBILITY.items():
        M0 = chem[sysd["metal"][5:]]
        X0 = chem[sysd["inter"][5:]]
        T_sol = solution_temperature(M0, X0, sysd["A"], sysd["B"])
        if T_sol is None:
            cells.append(
                f'<div><span>{label} · solution temp</span><b>—</b>'
                f'<em>no chemistry</em><br><span class="pill neutral">no chemistry</span></div>')
            continue
        und = float(system_undissolved(label, M0, X0, temps["frt"]))
        if und <= 1e-6:
            pill = '<span class="pill ok">fully dissolved at FRT</span>'
            sub = f"dissolved at {temps['frt']:.0f} °C"
        else:
            cls = "warn" if und < 0.5 else "bad"
            pill = f'<span class="pill {cls}">undissolved {und * 100:.1f}% at FRT</span>'
            sub = f"T_sol {T_sol - 273.15:.0f} °C"
        cells.append(
            f'<div><span>{label} · solution temp</span><b>{T_sol - 273.15:.0f} °C</b>'
            f'<em>{sub}</em><br>{pill}</div>')
    st.markdown('<div class="facts">' + "".join(cells) + '</div>', unsafe_allow_html=True)
    st.markdown('<div style="height:.8rem"></div>', unsafe_allow_html=True)

    # --- charts 1 & 2 --------------------------------------------------------------------
    col_a, col_b = st.columns(2)
    with col_a:
        st.plotly_chart(_solubility_fig(chem, temps["frt"]), use_container_width=True)
        st.markdown(
            '<div class="note">Each solid curve is log₁₀[M][X] = A − B/T for one precipitate '
            'system; the dashed horizontal line is this steel\'s own log₁₀(Me·X) level. Where the '
            'level crosses its curve the precipitate is just fully dissolved: that intersection '
            'is the solution temperature. Right of the dotted FRT marker (colder) the level sits '
            'above the curve, so the excess precipitates out.</div>', unsafe_allow_html=True)
    with col_b:
        st.plotly_chart(_undissolved_fig(chem, temps), use_container_width=True)
        st.markdown(
            '<div class="note">Sequential coupled model, one metal at a time: the far stabler '
            'nitride (especially TiN) precipitates first, then the remaining metal is offered to '
            'the carbide. That nitride-then-carbide order is the standard engineering '
            'approximation of the coupled equilibrium. Markers sit on each curve at the '
            'schedule\'s finish temperature.</div>', unsafe_allow_html=True)

    st.markdown('<div style="height:1rem"></div>', unsafe_allow_html=True)

    # --- ternary stability map ------------------------------------------------------------
    _section("c–n–me stability map")
    ctl_a, ctl_b = st.columns([1, 2])
    metal = ctl_a.selectbox("Metal", ["Nb", "Ti", "V"], key=f"ppt_metal_{mill}",
                            help="Axis maxima scale the simplex: Nb 0.09, Ti 0.06, V 0.12 wt%; "
                                 "C 0.35 wt%; N 0.02 wt%.")
    t_def = min(max(temps["frt"], 800.0), 1300.0)
    t_tern = ctl_b.slider("Map temperature (°C)", 800.0, 1300.0, float(t_def), 5.0,
                          key=f"ppt_ttemp_{mill}")
    st.plotly_chart(_ternary_fig(metal, t_tern, chem), use_container_width=True)
    st.markdown(
        '<div class="note">Every grid point is a chemistry normalised onto the ternary simplex '
        '(axes labelled in true wt%); the colour is the sequential-model undissolved metal '
        'fraction at the chosen temperature. Deep purple corners are compositions that would '
        'leave microalloy in solution; yellow interiors pin it out as carbonitride.</div>',
        unsafe_allow_html=True)

    st.markdown('<div style="height:1rem"></div>', unsafe_allow_html=True)

    # --- pancaking indicator + Zener pinning ----------------------------------------------
    _section("pancaking & zener pinning")
    f_entry = float(undissolved_fraction(metal, temps["entry"], chem[metal], chem["C"], chem["N"]))
    f_frt = float(undissolved_fraction(metal, temps["frt"], chem[metal], chem["C"], chem["N"]))
    f_exit = float(undissolved_fraction(metal, temps["exit"], chem[metal], chem["C"], chem["N"]))
    st.markdown(
        '<div class="facts">'
        f'<div><span>{metal} undissolved at entry ({temps["entry"]:.0f} °C)</span>'
        f'<b>{f_entry * 100:.1f}</b><em>%</em></div>'
        f'<div><span>{metal} undissolved at FRT ({temps["frt"]:.0f} °C)</span>'
        f'<b>{f_frt * 100:.1f}</b><em>%</em></div>'
        f'<div><span>{metal} undissolved at {temps["exit_label"]} ({temps["exit"]:.0f} °C)</span>'
        f'<b>{f_exit * 100:.1f}</b><em>%</em></div>'
        '</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="note">Pancaked-austenite indicator: undissolved {metal} surviving into the '
        f'finishing passes pins the recrystallisation front. A large entry-to-FRT fraction means '
        f'the strain-induced precipitates can hold the elongated, unrecrystallised grain shape '
        f'through to the transformation.</div>', unsafe_allow_html=True)

    st.markdown('<div style="height:.8rem"></div>', unsafe_allow_html=True)
    zc1, zc2 = st.columns(2)
    r_nm = zc1.slider("Mean precipitate radius (nm)", 5.0, 100.0, 20.0, 1.0,
                      key=f"ppt_rnm_{mill}")
    drive_mpa = zc2.slider("Recrystallization driving pressure (MPa)", 0.1, 5.0, 1.0, 0.1,
                           key=f"ppt_drv_{mill}",
                           help="At high reduction the stored-energy driving pressure is of "
                                "order 0.1–10 MPa; this slider sets the screening threshold.")

    f_mass_map = precipitate_mass_fractions(metal, temps["frt"], chem[metal],
                                            chem["C"], chem["N"])
    f_v = sum(volume_fraction(fm, RHO_PPT[label]) for label, fm in f_mass_map.items())
    p_z = zener_pinning_mpa(f_v, r_nm)
    if p_z >= drive_mpa:
        verdict, v_cls = "PINNING EFFECTIVE", "ok"
    elif p_z >= 0.5 * drive_mpa:
        verdict, v_cls = "MARGINAL", "warn"
    else:
        verdict, v_cls = "PINNING WEAK", "bad"

    st.markdown(
        '<div class="facts">'
        f'<div><span>precipitate mass fraction at FRT</span><b>{sum(f_mass_map.values()):.2e}</b>'
        f'<em>wt fraction</em></div>'
        f'<div><span>volume fraction f_v</span><b>{f_v:.2e}</b><em>ρ<sub>matrix</sub> 7.87</em></div>'
        f'<div><span>Zener pinning pressure</span><b>{p_z:.3f}</b><em> MPa · 3·f_v·γ / 2r</em></div>'
        f'<div><span>driving pressure</span><b>{drive_mpa:.1f}</b><em> MPa · screening threshold</em></div>'
        f'<div><span>verdict</span><span class="pill {v_cls}">{verdict}</span></div>'
        '</div>', unsafe_allow_html=True)
    st.plotly_chart(_zener_fig(f_v, drive_mpa), use_container_width=True)
    st.markdown(
        '<div class="note">Pinning pressure scales with volume fraction and inversely with '
        'radius, so finer, more abundant precipitates pin harder. The driving-pressure threshold '
        'is a screening criterion, not a calculation: the stored-energy driving pressure at high '
        'reduction is order-of-magnitude O(0.1–10) MPa.</div>', unsafe_allow_html=True)

    # --- method note -----------------------------------------------------------------------
    with st.expander("Method, sources & caveats"):
        st.markdown(
            '<div class="note"><b>Solubility products</b> (austenite, log₁₀ of the wt%-product): '
            'NbC and NbN, VC, VN: Irvine et al., 1967; TiN: Wada &amp; Pehlke, 1977; TiC: '
            'austenite fit, cf. the Wada &amp; Pehlke family; AlN: austenite fit, cf. the '
            'Davern &amp; Edmonds family. The wt%-product fits carry a literature spread of '
            'roughly ±200 K on solution temperatures.<br><br>'
            'These are <b>paraequilibrium screening estimates</b>, not Thermo-Calc calls: '
            'substitutional diffusion is assumed frozen and the microalloy partitions 1:1-molar '
            'between matrix and precipitate at each temperature. The sequential '
            'nitride-then-carbide order per metal is the standard engineering approximation of '
            'the coupled equilibrium: nitrides (especially TiN) are far stabler and precipitate '
            'first. Zener pinning uses γ = 0.5 J/m² and matrix density 7.87 g/cm³.</div>',
            unsafe_allow_html=True)
