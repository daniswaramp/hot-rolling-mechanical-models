"""Pass schedule tab: recrystallization regime & deformation-penetration analysis.

The operator drafts a pass sequence for the current sidebar schedule and the tab classifies
every pass's softening regime — DRX (metadynamic recrystallization folded into it), SRX, PRX
(partial) or NRX (no recrystallization) — plus whether the draft penetrates the full
thickness of the bar. The metallurgical model is a Sellars-type engineering screening chain:
Zener-Hollomon parameter, peak/critical strain, dynamic recrystallization during the pass,
solute-and-precipitation-retarded static recrystallization in the interpass, recrystallized
grain size, and a contact-length penetration criterion.

Why this tab never dispatches to Modal (unlike the optimizer, atlas and shadow): every step
is a closed-form numpy scalar expression — no model bundle is loaded, no grid is searched,
no artifact is read — so a 15-pass schedule costs well under a millisecond of arithmetic and
the charts dwarf it. A cloud round-trip would add seconds to a computation that is free, so
everything runs locally behind the single "Analyze schedule" submit button. The result is
kept in ``st.session_state[f"ps_result_{mill}"]`` and replayed on later reruns; nothing
recomputes without the button.

Calibration honesty: the prefactors (``a_p``, ``C_srx``, ``A_g``, ``k_nb``) are calibrated to
typical hot-rolling interpass behaviour at textbook anchor points — they are literature-shaped
screening constants, NOT fitted to plant data — and each anchor point is asserted in
``tests/test_pass_schedule.py``. All constants live in the one ``RX_CONSTANTS`` dict below,
each entry carrying its source string.

Simplification carried through to the UI: the austenite grain size entering the next pass is
the recrystallized grain size only when the interpass fully recrystallized (X_srx >= 0.95);
otherwise D0 carries over unchanged (no partial-refinement interpolation).

``precipitation`` is imported read-only for the Nb(C,N) solution temperature
(``precipitation.SOLUBILITY`` + ``precipitation.solution_temperature``).
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import physics       # noqa: E402 - read-only: Pickering sigma_ss
import precipitation  # noqa: E402 - read-only: SOLUBILITY dict + solution_temperature

INK, HAIR, GRID, MUTED = "#0b0d10", "#e5e7eb", "#f1f2f4", "#6b7280"
ACCENT = "#1746c8"          # GRP brand blue, same token as the app stylesheet

# Regime pill classes against the app stylesheet (.pill ok/warn/bad/neutral exist there;
# DRX gets an inline "info blue" pill because the file scope here forbids editing inject_css).
PILL_CLASS = {"SRX": "ok", "PRX": "warn", "NRX": "bad", "DRX": None}
PILL_DRX_INLINE = ("style='background:#eff6ff;color:#1746c8;border:1px solid #bfdbfe;'")
REGIME_COLOR = {"DRX": ACCENT, "SRX": "#065f46", "PRX": "#92400e", "NRX": "#991b1b"}
REGIMES = ("DRX", "SRX", "PRX", "NRX")

# ----------------------------------------------------------------------------- constants
# ONE module-level dict; every entry is (value, source string). The four calibrated
# prefactors state their anchor point and the fact that they are hot-rolling screening
# calibrations, not plant fits. Anchors are asserted in tests/test_pass_schedule.py.
RX_CONSTANTS = {
    "R_gas": (8.314, "universal gas constant, J/(mol K)"),
    "Q_def": (300.0e3, "activation energy for hot deformation of austenite, J/mol (Sellars)"),
    # eps_p = a_p * D0_um**b_p * Z**c_p; calibrated: eps_p = 0.20 at D0=100 um,
    # epsdot = 1/s, T = 1000 C (Z = 2.036e12). Prefactor calibrated to typical
    # hot-rolling interpass behaviour, not fitted to plant data.
    "a_p": (0.014234, "peak-strain prefactor - calibrated so eps_p = 0.20 at "
                      "D0 = 100 um, epsdot = 1/s, T = 1000 C (hot-rolling screening "
                      "calibration, not a plant fit)"),
    "b_p": (0.5, "peak-strain grain-size exponent, eps_p ~ D0^0.5 (Sellars)"),
    "c_p": (0.012, "peak-strain Zener-Hollomon exponent, eps_p ~ Z^0.012 (Sellars)"),
    "eps_c_frac": (0.83, "critical strain eps_c = 0.83 * eps_p (Sellars)"),
    "x_drx_k": (0.693, "Avrami ln 2, so X = 0.5 at one characteristic strain (dynamic)"),
    "x_drx_n": (1.5, "dynamic-recrystallization Avrami exponent (typical austenite)"),
    # t_50 = C_srx * eps^-4 * D0^2 * exp(Q_srx/(R T)) * retardation; calibrated:
    # t_50 = 4.0 s at T = 1000 C, eps = 0.15, D0 = 100 um, no solute retardation.
    # The anchor is a roughing-representative strain (0.15, not a finishing 0.30):
    # with the old 0.30 anchor the eps^-4 law over-penalized the 0.09-0.20 drafts
    # real plate schedules use, calling every roughing pass NRX. With this anchor
    # C-Mn roughing recrystallizes in ~10-30 s and Nb microalloyed grades below
    # the Nb(C,N) solution temperature stay unrecrystallized - the classic TMCP
    # picture. Screening calibration, not a plant fit.
    "C_srx": (9.835e-20, "static-recrystallization prefactor - calibrated so t_50 = 4.0 s "
                         "at T = 1000 C, eps = 0.15, D0 = 100 um, no retardation "
                         "(roughing-representative anchor; hot-rolling screening "
                         "calibration, not a plant fit)"),
    "srx_eps_exp": (-4.0, "t_50 ~ eps^-4 (Sellars)"),
    "srx_D0_exp": (2.0, "t_50 ~ D0^2 (Sellars)"),
    "Q_srx": (300.0e3, "activation energy for static recrystallization, J/mol (Sellars)"),
    "k_avr": (1.0, "static Avrami exponent: X = 1 - exp(-ln2 * t/t_50), exactly 0.5 at "
                   "t_50 by construction"),
    # retardation = (1 + k_nb [Nb]^1.3) * (1 + k_ti [Ti]^1.0); calibrated: an 8x t_50
    # ratio between 0.030 and 0.002 wt% Nb (Dutta-Sellars-type screening form).
    "k_nb": (875.26, "Nb solute-drag / strain-induced-precipitation retardation "
                     "coefficient - calibrated so 0.030 wt% Nb retards t_50 8x vs "
                     "0.002 wt% (Dutta & Sellars-type screening form)"),
    "nb_exp": (1.3, "Nb concentration exponent of the retardation factor (Dutta & Sellars)"),
    "k_ti": (300.0, "Ti solute-drag retardation coefficient (engineering value, same "
                    "screening form, [Ti]^1.0)"),
    "ti_exp": (1.0, "Ti concentration exponent of the retardation factor"),
    "ppt_pinning_factor": (3.0, "extra t_50 multiplier while rolling below the Nb(C,N) "
                                "solution temperature (precipitation pinning)"),
    # d_rx = A_g * eps^-0.5 * D0^0.5; calibrated: d_rx = 45 um at eps = 0.30, D0 = 100 um.
    "A_g": (2.4648, "recrystallized grain-size prefactor - calibrated so d_rx = 45 um at "
                    "eps = 0.30, D0 = 100 um (Sellars-type d_rx ~ eps^-0.5 D0^0.5)"),
    "drx_threshold": (0.05, "X_drx at or above which the pass itself is classified DRX"),
    "srx_complete": (0.95, "X_srx at or above which the interpass counts fully recrystallized"),
    "penetration_threshold": (0.5, "full-penetration rule of thumb l_c/h_m >= 0.5 "
                                   "(TMCP practice)"),
    "centre_floor": (0.25, "centre strain floor at P = 0: the centre sees 25% of the "
                           "surface strain"),
    "centre_gain": (0.75, "centre strain gain with penetration P"),
}


def _c(name):
    """The numeric value of one RX_CONSTANTS entry (its tuple's first slot)."""
    return RX_CONSTANTS[name][0]


# ----------------------------------------------------------------------------- pure model
def chem_from_raw(raw):
    """Sidebar chemistry (wt%) the tab needs, defensive against missing/NaN cells."""
    def num(key, default=0.0):
        try:
            v = float(raw.get(key, default))
        except (TypeError, ValueError):
            return float(default)
        return v if np.isfinite(v) else float(default)
    return {"Nb": num("Chem_Nb"), "Ti": num("Chem_Ti"),
            "N": num("Chem_N", 0.004), "C": num("Chem_C", 0.18)}


def nb_solution_temperature_k(chem):
    """Nb(C,N) solution temperature in K: the max of NbC and NbN for this chemistry.

    Reuses precipitation.SOLUBILITY and precipitation.solution_temperature; a missing
    side (empty Nb, N or C) simply drops out of the max, and no side at all returns None.
    """
    temps = []
    for label in ("NbC", "NbN"):
        sysd = precipitation.SOLUBILITY[label]
        m, x = chem["Nb"], chem["N"] if label == "NbN" else chem["C"]
        t = precipitation.solution_temperature(m, x, sysd["A"], sysd["B"])
        if t is not None:
            temps.append(t)
    return max(temps) if temps else None


def solute_retardation(nb_wt, ti_wt):
    """Solute-drag retardation factor on t_50: (1 + k_nb [Nb]^1.3)(1 + k_ti [Ti])."""
    return (1.0 + _c("k_nb") * max(float(nb_wt), 0.0) ** _c("nb_exp")) * \
           (1.0 + _c("k_ti") * max(float(ti_wt), 0.0) ** _c("ti_exp"))


def t50_seconds(t_k, eps, d0_um, nb_wt, ti_wt, below_solution):
    """Half-recrystallization time of the interpass (s), Sellars form + retardation.

    t_50 = C_srx * eps^-4 * D0^2 * exp(Q_srx/(R T)) * solute_retardation, multiplied by a
    further 3x precipitation-pinning factor when rolling below the Nb(C,N) solution
    temperature.
    """
    base = _c("C_srx") * (eps ** _c("srx_eps_exp")) * (d0_um ** _c("srx_D0_exp")) * \
        np.exp(_c("Q_srx") / (_c("R_gas") * float(t_k)))
    retard = solute_retardation(nb_wt, ti_wt)
    if below_solution:
        retard *= _c("ppt_pinning_factor")
    return float(base * retard)


def peak_strain(d0_um, zener):
    """Sellars-type peak strain eps_p = a_p * D0^0.5 * Z^0.012."""
    return _c("a_p") * (float(d0_um) ** _c("b_p")) * (float(zener) ** _c("c_p"))


def recrystallized_grain_size(eps, d0_um):
    """d_rx = A_g * eps^-0.5 * D0^0.5 (um)."""
    return _c("A_g") * (float(eps) ** -0.5) * (float(d0_um) ** 0.5)


def penetration_fraction(roll_radius_mm, dh_mm, h_prev_mm, h_i_mm):
    """Penetration fraction P in [0, 1] from the contact-length rule of thumb.

    l_c = sqrt(R * dh); P = clip(l_c / (0.5 * h_m), 0, 1) with h_m the mean thickness:
    full penetration at l_c/h_m >= 0.5 (TMCP practice).
    """
    h_m = 0.5 * (float(h_prev_mm) + float(h_i_mm))
    l_c = np.sqrt(max(float(roll_radius_mm), 1e-9) * max(float(dh_mm), 0.0))
    return float(np.clip(l_c / max(0.5 * h_m, 1e-9), 0.0, 1.0))


def classify_regime(x_drx, x_srx):
    """Pass regime: DRX (MDRX folded in) when the pass dynamically recrystallized,
    otherwise the interpass class SRX / PRX / NRX on X_srx."""
    if x_drx >= _c("drx_threshold"):
        return "DRX"
    if x_srx >= _c("srx_complete"):
        return "SRX"
    if x_srx >= _c("drx_threshold"):
        return "PRX"
    return "NRX"


def estimate_ferrite_microstructure(final_d_gamma_um, retained_strain,
                                    cooling_rate_k_s=5.0, sigma_ss_mpa=80.0):
    """Sellars-Beynon transformed ferrite grain size (um) and Hall-Petch yield screening (MPa).

    Links the final austenite grain size and accumulated retained strain (pancaking) at
    the end of the rolling schedule to transformed ferrite grain size d_alpha and a
    Hall-Petch microstructural yield screening estimate.
    """
    cr = max(float(cooling_rate_k_s), 0.2)
    d_gamma = max(float(final_d_gamma_um), 5.0)
    eps_ret = max(float(retained_strain), 0.0)
    d_alpha = float(np.clip(
        (1.8 + 3.2 * (cr ** -0.35)) * (d_gamma ** 0.33) * np.exp(-0.35 * eps_ret),
        2.5, 45.0,
    ))
    ys_micro_mpa = float(96.5 + float(sigma_ss_mpa) + 17.4 * ((d_alpha / 1000.0) ** -0.5))
    return {
        "d_alpha_um": d_alpha,
        "ys_micro_mpa": ys_micro_mpa,
    }


def _schedule_micro_params(mill, raw):
    """Derive screening cooling rate (K/s) and Pickering sigma_ss (MPa) from sidebar inputs."""
    def _num(key, default):
        try:
            v = float(raw.get(key, default))
        except (TypeError, ValueError):
            return float(default)
        return v if np.isfinite(v) else float(default)

    if any(k in raw for k in ("Chem_Mn", "Chem_Si", "Chem_Ni", "Chem_Cr", "Chem_Cu", "Chem_Mo")):
        sigma_ss_mpa = float(physics.sigma_ss(
            _num("Chem_Mn", 0.90), _num("Chem_Si", 0.20), _num("Chem_Ni", 0.02),
            _num("Chem_Cr", 0.02), _num("Chem_Cu", 0.02), _num("Chem_Ti", 0.002),
            _num("Chem_Mo", 0.005), _num("Chem_N", 0.004),
        ))
    else:
        sigma_ss_mpa = 80.0

    if mill == "Plate" and any(k in raw for k in ("ACC_Laminar_Header", "ACC_Headers_n", "Plate_T_mm")):
        headers = max(0.0, _num("ACC_Laminar_Header", _num("ACC_Headers_n", 0.0)))
        plate_t = max(_num("Plate_T_mm", 16.0), 4.0)
        cooling_rate_k_s = float(np.clip((1.0 + 1.5 * headers) * (16.0 / plate_t), 0.2, 60.0))
    elif mill != "Plate" and any(k in raw for k in ("Coil_T_mm", "Temp_FT_Avg_C", "Temp_CT_Top_C")):
        coil_t = max(_num("Coil_T_mm", 6.0), 1.5)
        ft = _num("Temp_FT_Avg_C", 835.0)
        ct = 0.5 * (_num("Temp_CT_Top_C", 595.0) + _num("Temp_CT_Bot_C", 558.0))
        cooling_rate_k_s = float(np.clip(max(ft - ct, 20.0) / max(coil_t * 2.5, 4.0), 0.2, 60.0))
    else:
        cooling_rate_k_s = 5.0

    return cooling_rate_k_s, sigma_ss_mpa


def compute_schedule(tb_mm, drafts, t1_c, dt_c, t_int_s, pass_duration_s,
                     roll_radius_mm, d0_um, chem,
                     cooling_rate_k_s=5.0, sigma_ss_mpa=80.0):
    """Run the Sellars chain over a validated draft sequence; pure, no Streamlit.

    Returns {"passes": [row dicts], "summary": {...}, "d_alpha_um": ..., "ys_micro_mpa": ...}.
    Pass i deforms at T_i = T1 - (i-1) * dT; the interpass after pass i sits at the same T_i.
    """
    t_sol_k = nb_solution_temperature_k(chem)
    nb, ti = chem["Nb"], chem["Ti"]
    rows = []
    h_prev = float(tb_mm)
    d0 = float(d0_um)
    retained_strain = 0.0
    for i, draft in enumerate(drafts, start=1):
        h_i = h_prev - float(draft)
        eps = float(np.log(h_prev / h_i))
        epsdot = eps / float(pass_duration_s)
        t_c = float(t1_c) - (i - 1) * float(dt_c)
        t_k = t_c + 273.15
        zener = epsdot * np.exp(_c("Q_def") / (_c("R_gas") * t_k))
        eps_p = peak_strain(d0, zener)
        eps_c = _c("eps_c_frac") * eps_p
        if eps >= eps_c:
            x_drx = float(1.0 - np.exp(-_c("x_drx_k") *
                                       ((eps - eps_c) / eps_p) ** _c("x_drx_n")))
        else:
            x_drx = 0.0
        below = t_sol_k is not None and t_k < t_sol_k
        t50 = t50_seconds(t_k, eps, d0, nb, ti, below)
        x_srx = float(1.0 - np.exp(-np.log(2.0) * (float(t_int_s) / t50) ** _c("k_avr")))
        regime = classify_regime(x_drx, x_srx)
        d_rx = recrystallized_grain_size(eps, d0)
        d0_out = d_rx if x_srx >= _c("srx_complete") else d0
        if x_srx >= _c("srx_complete"):
            retained_strain = 0.0
        else:
            retained_strain = (retained_strain + eps) * max(0.0, 1.0 - x_drx) * max(0.0, 1.0 - x_srx)
        p = penetration_fraction(roll_radius_mm, float(draft), h_prev, h_i)
        centre_eps = eps * (_c("centre_floor") + _c("centre_gain") * p)
        rows.append({
            "pass": i, "h_in": h_prev, "h_out": h_i, "draft": float(draft),
            "eps": eps, "t_c": t_c, "t_k": t_k, "epsdot": epsdot, "Z": float(zener),
            "eps_p": eps_p, "eps_c": eps_c, "x_drx": x_drx, "t50": t50,
            "retardation": float(solute_retardation(nb, ti) *
                                 (_c("ppt_pinning_factor") if below else 1.0)),
            "pinned": bool(below), "x_srx": x_srx, "regime": regime,
            "d_rx": d_rx, "d0_in": d0, "d0_out": d0_out,
            "retained_strain": retained_strain,
            "P": p, "centre_eps": centre_eps,
        })
        h_prev = h_i
        d0 = d0_out
    final = rows[-1]["h_out"] if rows else float(tb_mm)
    first_unpen = next((r["pass"] for r in rows if r["P"] < _c("penetration_threshold")), None)
    micro = estimate_ferrite_microstructure(
        d0, retained_strain,
        cooling_rate_k_s=cooling_rate_k_s,
        sigma_ss_mpa=sigma_ss_mpa,
    )
    summary = {
        "tb_mm": float(tb_mm), "final_gauge": final,
        "total_reduction_pct": 100.0 * (float(tb_mm) - final) / float(tb_mm) if tb_mm else 0.0,
        "n_full_pen": sum(1 for r in rows if r["P"] >= 0.99),
        "n_nrx": sum(1 for r in rows if r["regime"] == "NRX"),
        "first_unpenetrated": first_unpen,
        "t_sol_c": (t_sol_k - 273.15) if t_sol_k is not None else None,
        "n_passes": len(rows),
        "final_d_gamma_um": float(d0),
        "retained_strain": float(retained_strain),
        "d_alpha_um": micro["d_alpha_um"],
        "ys_micro_mpa": micro["ys_micro_mpa"],
    }
    return {"passes": rows, "summary": summary,
            "d_alpha_um": micro["d_alpha_um"],
            "ys_micro_mpa": micro["ys_micro_mpa"],
            "params": {"t1_c": float(t1_c), "dt_c": float(dt_c), "t_int_s": float(t_int_s),
                       "pass_duration_s": float(pass_duration_s),
                       "roll_radius_mm": float(roll_radius_mm), "d0_um": float(d0_um),
                       "chem": dict(chem),
                       "cooling_rate_k_s": float(cooling_rate_k_s),
                       "sigma_ss_mpa": float(sigma_ss_mpa)}}


def draft_sequence(tb_mm, n_passes, mode, equal_draft, custom_drafts, gauge_mm):
    """Validate a drafting mode into a per-pass draft list; returns (drafts, error).

    Equal drafting repeats one draft n times. Custom per-pass takes the n-1 entered drafts
    and lets the last pass absorb the remainder down to the product gauge. The total
    reduction must leave a final gauge of at least 4 mm.
    """
    n = int(n_passes)
    if n < 2:
        return None, "At least 2 passes are required."
    if mode == "Equal drafting (mm)":
        drafts = [float(equal_draft)] * n
    else:
        entered = [float(d) for d in custom_drafts]
        if any(d <= 0.0 for d in entered):
            return None, "Every entered draft must be positive."
        remainder = float(tb_mm) - float(gauge_mm) - sum(entered)
        if remainder <= 0.0:
            return None, (f"The entered drafts already exceed the available reduction "
                          f"({float(tb_mm):.1f} -> {float(gauge_mm):.1f} mm).")
        drafts = entered + [remainder]
    if any(d <= 0.0 for d in drafts):
        return None, "Every pass needs a positive draft."
    final = float(tb_mm) - sum(drafts)
    if final < 4.0:
        return None, (f"Total reduction leaves a final gauge of {final:.1f} mm; "
                      f"the analysis needs at least 4 mm.")
    return drafts, None


# ----------------------------------------------------------------------------- helpers
def _section(label):
    st.markdown(f'<div class="section"><span>{label}</span><i></i></div>', unsafe_allow_html=True)


def _base_fig(title, x_title, y_title, height=380):
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


def _pill(regime):
    cls = PILL_CLASS[regime]
    if cls is None:  # DRX: inline info-blue pill, the stylesheet carries no .pill.info
        return f"<span class='pill' {PILL_DRX_INLINE}>{regime}</span>"
    return f"<span class='pill {cls}'>{regime}</span>"


def _regime_table_html(rows):
    head = ("Pass", "Gauge in (mm)", "Gauge out (mm)", "ε", "T (°C)", "Z", "εp", "εc",
            "X<sub>srx</sub>", "Regime", "Penetration", "d<sub>rx</sub> (µm)", "Note")
    body = []
    for r in rows:
        note = "Nb(C,N) pins: ×3 t₅₀" if r["pinned"] else ""
        body.append(
            f"<tr><td>{r['pass']}</td>"
            f"<td>{r['h_in']:.1f}</td><td>{r['h_out']:.1f}</td>"
            f"<td>{r['eps']:.3f}</td><td>{r['t_c']:.0f}</td>"
            f"<td>{r['Z']:.2e}</td><td>{r['eps_p']:.3f}</td><td>{r['eps_c']:.3f}</td>"
            f"<td>{r['x_srx'] * 100:.1f}%</td><td>{_pill(r['regime'])}</td>"
            f"<td>{r['P'] * 100:.0f}%</td><td>{r['d_rx']:.1f}</td>"
            f"<td class='sub'>{note}</td></tr>")
    th = "".join(f"<th>{h}</th>" for h in head)
    return ('<div class="card" style="padding:0 1rem 0.4rem; margin: .6rem 0;">'
            '<table class="ut"><thead><tr>' + th + "</tr></thead><tbody>"
            + "".join(body) + "</tbody></table></div>")


def _passes_frame(rows):
    """The per-pass table as a tidy frame for the CSV download."""
    return pd.DataFrame([{
        "Pass": r["pass"], "Gauge in (mm)": round(r["h_in"], 2),
        "Gauge out (mm)": round(r["h_out"], 2), "Draft (mm)": round(r["draft"], 2),
        "True strain": round(r["eps"], 4), "Strain rate (1/s)": round(r["epsdot"], 3),
        "T (C)": round(r["t_c"], 1), "Zener-Hollomon Z": f"{r['Z']:.3e}",
        "eps_p": round(r["eps_p"], 4), "eps_c": round(r["eps_c"], 4),
        "X_drx": round(r["x_drx"], 4),
        "t_50 (s)": round(r["t50"], 3), "Retardation factor": round(r["retardation"], 2),
        "Precipitation pinned": "yes" if r["pinned"] else "no",
        "X_srx": round(r["x_srx"], 4), "Regime": r["regime"],
        "d_rx (um)": round(r["d_rx"], 1), "D0 in (um)": round(r["d0_in"], 1),
        "Retained strain": round(r.get("retained_strain", 0.0), 4),
        "Penetration P": round(r["P"], 3), "Centre strain": round(r["centre_eps"], 4),
    } for r in rows])


def _strain_contour_fig(rows):
    """2D strain map: x = pass, y = normalised through-thickness position, z = strain.

    eps(y) = centre + (surface - centre) * y^2 with the surface strain at |y| = 1 and the
    penetration-reduced centre strain at y = 0, so the unpenetrated zone reads as the
    low-strain band in the middle.
    """
    y = np.linspace(-1.0, 1.0, 41)
    passes = [r["pass"] for r in rows]
    z = np.array([[r["centre_eps"] + (r["eps"] - r["centre_eps"]) * (yy ** 2)
                   for r in rows] for yy in y])
    fig = _base_fig("Equivalent strain through the thickness (surface to surface)",
                    "Pass", "Normalised position (-1 surface … +1 surface)", height=400)
    fig.add_trace(go.Contour(
        x=passes, y=y, z=z, colorscale="Blues", line_smoothing=0.85,
        colorbar=dict(title=dict(text="strain", font=dict(size=10)), tickfont=dict(size=9)),
        hovertemplate=("pass %{x}<br>position %{y:.2f}<br>strain %{z:.3f}<extra></extra>"),
    ))
    return fig


def _penetration_fig(rows):
    fig = _base_fig("Penetration per pass (l_c / half mean thickness, clipped to 1)",
                    "Pass", "Penetration P", height=320)
    fig.add_scatter(x=[r["pass"] for r in rows], y=[r["P"] for r in rows],
                    mode="lines+markers", name="P", line=dict(color=ACCENT, width=2),
                    marker=dict(size=8, color=ACCENT,
                                line=dict(width=1, color="#ffffff")),
                    hovertemplate="pass %{x}<br>P %{y:.2f}<extra></extra>")
    thr = _c("penetration_threshold")
    fig.add_hline(y=thr, line=dict(color="#991b1b", width=1.3, dash="dash"),
                  annotation_text="penetration threshold 0.5",
                  annotation_position="top right", annotation_font_size=9)
    fig.update_yaxes(range=[-0.02, 1.05])
    return fig


def _timeline_fig(rows):
    fig = _base_fig("Regime timeline: pass temperature coloured by softening regime",
                    "Pass", "Pass temperature (°C)", height=320)
    for regime in REGIMES:
        sel = [r for r in rows if r["regime"] == regime]
        if not sel:
            continue
        fig.add_scatter(x=[r["pass"] for r in sel], y=[r["t_c"] for r in sel],
                        mode="markers", name=regime,
                        marker=dict(size=11, color=REGIME_COLOR[regime],
                                    line=dict(width=1, color="#ffffff")),
                        hovertemplate=(f"{regime} · pass %{{x}} · %{{y:.0f}} °C"
                                       "<extra></extra>"))
    fig.update_layout(legend=dict(orientation="h", y=-0.3, x=0, font=dict(size=9)))
    return fig


def _render_result(mill, stored):
    """Render one stored analysis: regime table, facts, contour, penetration, timeline, CSV."""
    rows = stored["passes"]
    s = stored["summary"]
    chem = stored["params"]["chem"]

    _section("regime per pass")
    st.markdown(_regime_table_html(rows), unsafe_allow_html=True)
    st.markdown(
        '<div class="note">DRX (metadynamic recrystallization folded in) means the pass itself '
        'recrystallized (ε ≥ ε<sub>c</sub> and X<sub>drx</sub> ≥ 5%). Otherwise the interpass '
        'decides: SRX ≥ 95% softening, PRX in between, NRX below 5% — the dangerous passes that '
        'carry pancaked austenite into the next draft. "Nb(C,N) pins" marks passes rolled below '
        'the steel\'s Nb(C,N) solution temperature'
        + (f" ({s['t_sol_c']:.0f} °C)" if s["t_sol_c"] is not None else "")
        + ', where precipitation adds a further ×3 to t₅₀.</div>', unsafe_allow_html=True)

    verdict_pill = ('<span class="pill ok">all passes penetrate</span>'
                    if s["first_unpenetrated"] is None
                    else '<span class="pill bad">partial penetration</span>')
    d_alpha = s.get("d_alpha_um", stored.get("d_alpha_um"))
    ys_micro = s.get("ys_micro_mpa", stored.get("ys_micro_mpa"))
    if d_alpha is None or ys_micro is None:
        final_d_gamma = rows[-1]["d0_out"] if rows else stored["params"].get("d0_um", 100.0)
        ret_eps = rows[-1].get("retained_strain", 0.0) if rows else 0.0
        micro = estimate_ferrite_microstructure(final_d_gamma, ret_eps)
        d_alpha = micro["d_alpha_um"]
        ys_micro = micro["ys_micro_mpa"]
    d_gamma_final = s.get(
        "final_d_gamma_um",
        rows[-1]["d0_out"] if rows else stored["params"].get("d0_um", 100.0),
    )
    st.markdown('<div class="facts">'
                f'<div><span>total reduction</span><b>{s["total_reduction_pct"]:.1f}</b>'
                f'<em>%</em></div>'
                f'<div><span>final gauge</span><b>{s["final_gauge"]:.1f}</b><em>mm</em></div>'
                f'<div><span>fully penetrated (P ≥ 0.99)</span>'
                f'<b>{s["n_full_pen"]}</b><em>of {s["n_passes"]} passes</em></div>'
                f'<div><span>NRX passes (unrecrystallized)</span><b>{s["n_nrx"]}</b>'
                f'<em>dangerous</em></div>'
                f'<div><span>ferrite grain d<sub>α</sub></span><b>{d_alpha:.1f}</b>'
                f'<em>µm (D<sub>γ</sub> {d_gamma_final:.0f} µm)</em></div>'
                f'<div><span>Hall-Petch YS<sub>micro</sub></span><b>{ys_micro:.0f}</b>'
                f'<em>MPa (screening)</em></div>'
                f'<div><span>verdict</span>{verdict_pill}</div>'
                '</div>', unsafe_allow_html=True)
    if s["first_unpenetrated"] is not None:
        st.markdown(
            f'<div class="note"><b>strain does not penetrate below pass '
            f'{s["first_unpenetrated"]}</b> — from that pass on the contact length is under '
            'half the mean thickness, so the centre deforms less than the surface and the '
            'through-thickness structure diverges. Smaller drafts per pass, a bigger roll '
            'radius or a thinner bar fixes it.</div>', unsafe_allow_html=True)
    if s["n_nrx"] == s["n_passes"] and s["n_passes"] > 0:
        st.markdown(
            '<div class="note"><b>every pass came out NRX.</b> With light drafts the '
            'ε⁻⁴ law stretches t₅₀ far past the interpass time — that is the model, not a '
            'display fault. Heavier early drafts (roughing wants 0.15-0.35 strain per pass), '
            'longer interpass times, a higher entry temperature or a leaner chemistry move '
            'passes toward PRX and SRX; a Nb-bearing grade rolled below its Nb(C,N) '
            'solution temperature will hold NRX by design — that is what pancaking is.'
            '</div>', unsafe_allow_html=True)
    elif s["n_nrx"] > 0:
        st.markdown(
            f'<div class="note"><b>{s["n_nrx"]} of {s["n_passes"]} passes stay '
            'unrecrystallized.</b> Passes below the Nb(C,N) solution temperature carry '
            'pancaked austenite into the next draft — the intended TMCP state for ferrite '
            'grain refinement, provided the earlier passes fully recrystallized.</div>',
            unsafe_allow_html=True)

    _section("through-thickness strain")
    st.plotly_chart(_strain_contour_fig(rows), use_container_width=True,
                    config={"displayModeBar": False})
    st.markdown('<div class="note">Parabolic strain profile per pass: surface strain ε at the '
                'two surfaces, the penetration-reduced centre strain in the middle. The pale '
                'vertical band is the unpenetrated zone — strain that never reached the '
                'centre.</div>', unsafe_allow_html=True)

    _section("penetration & timeline")
    left, right = st.columns(2)
    with left:
        st.plotly_chart(_penetration_fig(rows), use_container_width=True,
                        config={"displayModeBar": False})
    with right:
        st.plotly_chart(_timeline_fig(rows), use_container_width=True,
                        config={"displayModeBar": False})
    st.markdown('<div class="note">Left: penetration per pass against the 0.5 rule-of-thumb '
                'threshold (full penetration at l<sub>c</sub>/h<sub>m</sub> ≥ 0.5, TMCP '
                'practice). Right: each pass plotted at its deformation temperature, coloured '
                'by regime — a run of NRX at falling temperature is the classic sign the last '
                'passes are rolling unrecrystallized austenite.</div>', unsafe_allow_html=True)

    _section("per-pass data")
    frame = _passes_frame(rows)
    st.download_button("Download per-pass table (CSV)",
                       data=frame.to_csv(index=False).encode("utf-8"),
                       file_name=f"pass_schedule_{mill}.csv", mime="text/csv",
                       key=f"ps_csv_{mill}")
    params = stored["params"]
    st.caption("Chemistry of this analysis (sidebar schedule): "
               f"Nb {chem['Nb']:.3f} · Ti {chem['Ti']:.3f} · N {chem['N']:.4f} · "
               f"C {chem['C']:.3f} wt% · cooling {params.get('cooling_rate_k_s', 5.0):.1f} K/s · "
               f"σ_ss {params.get('sigma_ss_mpa', 80.0):.0f} MPa.")


# ----------------------------------------------------------------------------- view
def view_pass_schedule(mill, raw):
    """The Pass schedule tab: draft a sequence, press Analyze schedule, read the regimes.

    Every input lives inside one st.form, so touching a widget never recomputes anything;
    the submitted analysis is kept in st.session_state[f"ps_result_{mill}"] and replayed on
    later reruns, exactly like the Inverse Optimizer's result. The math itself is closed-form
    numpy (see the module docstring) — local by design, no Modal dispatch.
    """
    raw = dict(raw)
    chem = chem_from_raw(raw)

    # Defaults from the sidebar schedule: transfer-bar thickness and the product gauge the
    # custom drafting mode drafts down to (assumption: the sidebar's product gauge —
    # Plate_T_mm / Coil_T_mm — is the target the last pass absorbs the remainder to).
    gauge_default = float(raw.get("Plate_T_mm", raw.get("Coil_T_mm", 10.0)))
    if mill == "Plate":
        if raw.get("TB_Thickness_mm") is not None:
            tb_default = float(raw["TB_Thickness_mm"])
        elif raw.get("TB_Ratio") is not None:
            tb_default = float(raw["TB_Ratio"]) * float(raw.get("Plate_T_mm", gauge_default))
        else:
            tb_default = 41.0
        t1_default = float(raw.get("FM_Temp_Before_Rolling",
                                   raw.get("Temp_Transfer_Bar_C", 985.0)))
    else:
        if raw.get("TB_Thick_mm") is not None:
            tb_default = float(raw["TB_Thick_mm"])
        elif raw.get("TB_Thickness_mm") is not None:
            tb_default = float(raw["TB_Thickness_mm"])
        elif raw.get("TB_Ratio") is not None:
            tb_default = float(raw["TB_Ratio"]) * float(raw.get("Coil_T_mm", gauge_default))
        else:
            tb_default = 24.8
        t1_default = float(raw.get("Temp_Transfer_Bar_C",
                                   raw.get("FM_Temp_Before_Rolling", 937.0)))
    tb_default = min(max(tb_default, 15.0), 120.0)

    state_key = f"ps_result_{mill}"
    n_pending = int(st.session_state.get(f"ps_n_{mill}", 7))
    # A shrunk pass count would leave out-of-range draft widgets behind.
    for i in range(n_pending - 1, 14):
        st.session_state.pop(f"ps_draft_{i}_{mill}", None)

    _section("pass schedule recrystallization")
    st.markdown(
        '<div class="note">Sellars-type screening of your drafting sequence: which passes '
        'dynamically recrystallize, which interpasses fully recover (SRX), only partly '
        '(PRX) or not at all (NRX), and whether each draft penetrates the full thickness. '
        'Chemistry comes from the sidebar; everything computes only when you press '
        '<b>Analyze schedule</b>, then replays from session state.</div>',
        unsafe_allow_html=True)

    with st.form(key=f"ps_form_{mill}"):
        c1, c2, c3 = st.columns(3)
        tb = c1.number_input("Transfer bar thickness (mm)", 15.0, 120.0,
                             value=tb_default, step=0.5, key=f"ps_tb_{mill}")
        n_passes = c2.number_input("Number of passes", 2, 15, 7, 1, key=f"ps_n_{mill}")
        gauge = c3.number_input("Product gauge (mm)", 1.5, 120.0, value=gauge_default,
                                step=0.5, key=f"ps_gauge_{mill}",
                                help="The finish gauge the sequence drafts down to; the "
                                     "custom mode's last pass absorbs whatever remains.")
        mode = st.radio("Reduction per pass", ["Equal drafting (mm)", "Custom per-pass (mm)"],
                        horizontal=True, key=f"ps_mode_{mill}")
        if mode == "Equal drafting (mm)":
            equal_draft = st.number_input(
                "Draft per pass (mm)", 0.0, 120.0,
                value=max((tb_default - gauge_default) / 7.0, 0.1), step=0.1,
                key=f"ps_equal_{mill}",
                help="Applied identically to every pass; the final gauge is whatever "
                     "remains, and must stay at or above 4 mm.")
        else:
            st.caption(f"{int(n_passes) - 1} drafts to enter — the last pass absorbs the "
                       f"remainder down to the {gauge_default:.1f} mm product gauge.")
            custom = []
            cols = st.columns(4)
            for i in range(int(n_passes) - 1):
                share = max((tb_default - gauge_default) / int(n_passes), 0.1)
                custom.append(cols[i % 4].number_input(
                    f"Pass {i + 1} draft (mm)", 0.0, 120.0, value=share, step=0.1,
                    key=f"ps_draft_{i}_{mill}"))
        c4, c5, c6 = st.columns(3)
        t1 = c4.number_input("Entry temperature T1 (°C)", 700.0, 1250.0,
                             value=t1_default, step=1.0, key=f"ps_t1_{mill}")
        dt = c5.number_input("Interpass temperature drop ΔT (°C/pass)", 0.0, 100.0,
                             value=20.0, step=1.0, key=f"ps_dt_{mill}")
        t_int = c6.number_input("Interpass time (s)", 1.0, 300.0, value=15.0, step=0.5,
                                key=f"ps_tint_{mill}")
        c7, c8, c9 = st.columns(3)
        dur = c7.number_input("Pass duration (s)", 0.1, 5.0, value=0.5, step=0.1,
                              key=f"ps_dur_{mill}",
                              help="Sets the mean strain rate: epsdot = ε / duration.")
        roll_r = c8.number_input("Roll radius (mm)", 200.0, 600.0, value=460.0, step=5.0,
                                 key=f"ps_roll_{mill}")
        d0 = c9.number_input("Initial austenite grain size D0 (µm)", 20.0, 300.0,
                             value=100.0, step=5.0, key=f"ps_d0_{mill}")
        st.caption(f"Sidebar chemistry (wt%): Nb {chem['Nb']:.3f} · Ti {chem['Ti']:.3f} · "
                   f"N {chem['N']:.4f} · C {chem['C']:.3f} · "
                   f"TB_Ratio {tb / max(gauge, 1e-6):.2f}× ({tb:.1f} mm / {gauge:.1f} mm).")
        submitted = st.form_submit_button("Analyze schedule", type="primary",
                                          help="Runs the recrystallization & penetration "
                                               "analysis for this sequence.")

    if submitted:
        drafts, err = draft_sequence(tb, n_passes, mode, equal_draft if mode ==
                                     "Equal drafting (mm)" else 0.0,
                                     custom if mode != "Equal drafting (mm)" else [],
                                     gauge)
        if err:
            st.error(err)
        else:
            cooling_rate_k_s, sigma_ss_mpa = _schedule_micro_params(mill, raw)
            st.session_state[state_key] = compute_schedule(
                tb, drafts, t1, dt, t_int, dur, roll_r, d0, chem,
                cooling_rate_k_s=cooling_rate_k_s, sigma_ss_mpa=sigma_ss_mpa)

    stored = st.session_state.get(state_key)
    if not stored:
        st.caption("Enter the drafting sequence above and press **Analyze schedule** to "
                   "classify its recrystallization regimes and penetration.")
        return
    if not submitted:
        st.caption("Replayed from your last analysis; change the sequence and press "
                   "Analyze schedule to recompute.")
    _render_result(mill, stored)
