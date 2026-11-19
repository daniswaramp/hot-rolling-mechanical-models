"""
Pure-function tests for the Precipitation & Solubility tab (app/precipitation.py).

No Streamlit rendering: only the thermodynamic helpers — solubility products, solution
temperatures, the 1:1-molar precipitation balance, the sequential coupled model and the Zener
pinning formula.
"""

import os
import sys

import numpy as np
import pytest

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(TEST_DIR)
APP_DIR = os.path.join(REPO_ROOT, "app")

for p in [REPO_ROOT, APP_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

import precipitation as ppt  # noqa: E402

# typical microalloyed HSLA chemistry (wt%)
CHEM = {"C": 0.18, "N": 0.005, "Nb": 0.03, "Ti": 0.002, "V": 0.05, "Al": 0.03}

SYSTEM_CHEM = {
    "NbC": ("Nb", "C"), "NbN": ("Nb", "N"),
    "TiC": ("Ti", "C"), "TiN": ("Ti", "N"),
    "VC": ("V", "C"), "VN": ("V", "N"),
    "AlN": ("Al", "N"),
}


def _t_sol(label, M0, X0):
    sysd = ppt.SOLUBILITY[label]
    return ppt.solution_temperature(M0, X0, sysd["A"], sysd["B"])


def test_solution_temperature_ordering():
    """Stability ladder at typical chemistry: TiN highest, VC dissolves easiest.

    Asserted chain (K): TiN 1462 > NbN 1435 > AlN 1395 > NbC 1389 > VN 1141 > VC 1084,
    i.e. the textbook ordering with VC fully soluble at these levels.
    """
    temps = {label: _t_sol(label, CHEM[m], CHEM[x]) for label, (m, x) in SYSTEM_CHEM.items()}
    assert all(t is not None and np.isfinite(t) for t in temps.values())
    ladder = ["TiN", "NbN", "AlN", "NbC", "VN", "VC"]
    seq = [temps[label] for label in ladder]
    assert all(a > b for a, b in zip(seq, seq[1:])), f"ladder broken: {dict(zip(ladder, seq))}"
    assert temps["TiN"] == max(temps.values())
    assert temps["VC"] == min(temps.values())


def test_solution_temperature_guards():
    assert ppt.solution_temperature(0.0, 0.18, 3.42, 7900.0) is None
    assert ppt.solution_temperature(0.03, 0.0, 3.42, 7900.0) is None
    assert ppt.solution_temperature(None, 0.18, 3.42, 7900.0) is None
    # product so large that log10(P) >= A has no finite positive solution (10^3.42 ~ 2630)
    assert ppt.solution_temperature(100.0, 100.0, 3.42, 7900.0) is None


def test_undissolved_bounds_and_monotonic():
    """Fraction stays in [0, 1] and never increases with temperature (sequential model)."""
    T = np.linspace(800.0, 1400.0, 301)
    for metal in ("Nb", "Ti", "V", "Al"):
        f = ppt.undissolved_fraction(metal, T, CHEM[metal], CHEM["C"], CHEM["N"])
        assert f.shape == T.shape
        assert np.all(f >= 0.0) and np.all(f <= 1.0)
        assert np.all(np.diff(f) <= 1e-10), f"{metal} fraction rises with temperature"
        # single-system balance shares the same guarantees
        for label in ppt.METAL_SYSTEMS[metal]:
            sysd = ppt.SOLUBILITY[label]
            X0 = CHEM["N"] if sysd["inter"].endswith("_N") else CHEM["C"]
            g = ppt.system_undissolved(label, CHEM[metal], X0, T)
            assert np.all(g >= 0.0) and np.all(g <= 1.0)
            assert np.all(np.diff(g) <= 1e-10)


def test_n_starvation_leaves_titanium_dissolved():
    """N-starved steel cannot precipitate TiN, so its Ti stays dissolved.

    (Direction note: the physically correct sense is that *more* nitrogen ties up *more*
    titanium as undissolved TiN — the task brief's wording inverted the two chemistries, the
    assertion below encodes the thermodynamics.)
    """
    f_starved_nitride = ppt.system_undissolved("TiN", 0.002, 0.001, 1000.0)
    f_rich_nitride = ppt.system_undissolved("TiN", 0.002, 0.02, 1000.0)
    assert f_rich_nitride > 0.9, "N-rich must pin nearly all Ti as TiN at 1000 C"
    assert f_starved_nitride < 0.25, "N-starved TiN is mostly dissolved at 1000 C"
    f_starved = ppt.undissolved_fraction("Ti", 1000.0, 0.002, 0.18, 0.001)
    f_rich = ppt.undissolved_fraction("Ti", 1000.0, 0.002, 0.18, 0.02)
    assert f_rich > f_starved
    assert f_rich > 0.9


def test_round_trip_at_solution_temperature():
    """At T = T_sol the system is just fully dissolved: undissolved fraction ~ 0."""
    for label, (m, x) in SYSTEM_CHEM.items():
        M0, X0 = CHEM[m], CHEM[x]
        T_sol = _t_sol(label, M0, X0)
        assert T_sol is not None
        und = ppt.system_undissolved(label, M0, X0, T_sol - 273.15)
        assert und == pytest.approx(0.0, abs=1e-3), f"{label} not fully dissolved at T_sol"
        assert (1.0 - und) == pytest.approx(1.0, abs=1e-3)


def test_zero_chemistry_degrades_cleanly():
    T = np.linspace(800.0, 1400.0, 5)
    for metal in ("Nb", "Ti", "V", "Al"):
        f = ppt.undissolved_fraction(metal, T, 0.0, 0.0, 0.0)
        assert np.all(f == 0.0)
    assert ppt.system_undissolved("NbC", 0.0, 0.18, 900.0) == 0.0
    fmap = ppt.precipitate_mass_fractions("Nb", 900.0, 0.0, 0.0, 0.0)
    assert sum(fmap.values()) == pytest.approx(0.0)


def test_precipitate_mass_fractions_consistent():
    fmap = ppt.precipitate_mass_fractions("Ti", 1000.0, 0.002, 0.18, 0.005)
    assert set(fmap) == {"TiN", "TiC"}
    assert all(v >= 0.0 for v in fmap.values())
    frac = ppt.undissolved_fraction("Ti", 1000.0, 0.002, 0.18, 0.005)
    # each system's steel mass fraction is s*(1 + r)/100, so it exceeds the bare metal fraction
    assert sum(fmap.values()) >= frac * 0.002 / 100.0


def test_zener_pinning_scaling_and_clamps():
    base = ppt.zener_pinning_mpa(1e-4, 20.0)
    assert base > 0.0
    # linear in volume fraction, inverse in radius
    assert ppt.zener_pinning_mpa(2e-4, 20.0) == pytest.approx(2.0 * base)
    assert ppt.zener_pinning_mpa(1e-4, 10.0) == pytest.approx(2.0 * base)
    # clamps: negative volume fraction -> 0, f_v > 1 saturates, radius floors at 1 nm
    assert ppt.zener_pinning_mpa(0.0, 20.0) == 0.0
    assert ppt.zener_pinning_mpa(-1.0, 20.0) == 0.0
    assert ppt.zener_pinning_mpa(5.0, 20.0) == pytest.approx(ppt.zener_pinning_mpa(1.0, 20.0))
    assert ppt.zener_pinning_mpa(1e-4, 0.0) == pytest.approx(ppt.zener_pinning_mpa(1e-4, 1.0))
    assert ppt.zener_pinning_mpa(1e-4, -5.0) == pytest.approx(ppt.zener_pinning_mpa(1e-4, 1.0))


def test_volume_fraction_density_ratio():
    assert ppt.volume_fraction(0.001, 7.83) == pytest.approx(0.001 * 7.87 / 7.83)
    assert ppt.volume_fraction(0.0, 3.26) == 0.0


# --------------------------------------------------------------------------- kinetics
def _T_for_drive(label, M0, X0, D):
    """Temperature (K) at which the system's supersaturation ratio is exactly 10**D.

    From K(T) = 10^(A - B/T) and S = P/K = 10**D:  A - B/T = log10(P) - D.
    """
    sysd = ppt.SOLUBILITY[label]
    return sysd["B"] / (sysd["A"] - np.log10(M0 * X0) + D)


def test_kinetics_constants_calibration_bounds():
    """Calibration targets: D = 1.0 -> t_50 ~ 100-600 s; D = 0.2 -> t_50 >= 1 h.

    The driving force is set exactly by choosing T from the solubility law of the metal's
    primary (nitride) system, so the assertion exercises the full kinetics_t50 path.
    """
    M0, C0, N0 = 0.03, 0.18, 0.005
    t50_fast = ppt.kinetics_t50("Nb", _T_for_drive("NbN", M0, N0, 1.0) - 273.15, M0, C0, N0)
    assert t50_fast is not None
    assert 100.0 <= t50_fast <= 600.0, f"t_50 at D=1.0 is {t50_fast:.0f} s, want 100-600 s"
    t50_slow = ppt.kinetics_t50("Nb", _T_for_drive("NbN", M0, N0, 0.2) - 273.15, M0, C0, N0)
    assert t50_slow is not None
    assert t50_slow >= 3600.0, f"t_50 at D=0.2 is {t50_slow:.0f} s, want >= 1 h"
    # normalization: t_50 equals T50_REF_S exactly at the reference driving force
    t50_ref = ppt.kinetics_t50("Nb", _T_for_drive("NbN", M0, N0, 0.5) - 273.15, M0, C0, N0)
    assert t50_ref == pytest.approx(ppt.KINETICS_CONSTANTS["T50_REF_S"], rel=1e-9)
    # capped at 30 days no matter how weak the driving force
    t50_tiny = ppt.kinetics_t50("Nb", _T_for_drive("NbN", M0, N0, 1e-4) - 273.15, M0, C0, N0)
    assert t50_tiny <= ppt.KINETICS_CONSTANTS["T50_CAP_S"] + 1e-6


def test_kinetics_t50_none_and_system_resolution():
    """No chemistry or no supersaturation -> None; system labels resolve like metals."""
    assert ppt.kinetics_t50("Nb", 900.0, 0.0, 0.18, 0.005) is None
    assert ppt.kinetics_t50("Nb", 900.0, 0.03, 0.18, 0.0) is None
    assert ppt.kinetics_t50("Nb", 900.0, None, 0.18, 0.005) is None
    assert ppt.kinetics_t50("Unknown", 900.0, 0.03, 0.18, 0.005) is None
    # above the solution temperature the system is undersaturated: no driving force
    t_sol = _t_sol("NbN", 0.03, 0.005)
    assert ppt.kinetics_t50("Nb", t_sol - 273.15 + 50.0, 0.03, 0.18, 0.005) is None
    # a direct system label resolves to the same clock as its metal (nitride-first)
    assert ppt.kinetics_t50("NbN", 900.0, 0.03, 0.18, 0.005) == \
        ppt.kinetics_t50("Nb", 900.0, 0.03, 0.18, 0.005)
    # carbide label uses the carbon pool, not nitrogen
    assert ppt.kinetics_t50("NbC", 900.0, 0.03, 0.0, 0.005) is None
    assert ppt.kinetics_t50("NbC", 900.0, 0.03, 0.18, 0.0) is not None


def test_jmak_fraction_half_at_t50():
    """X(t_50) = 1 - exp(-ln2) = 0.5 exactly; boundary behaviour at t = 0 and t -> inf."""
    assert ppt.jmak_fraction(1000.0, 1000.0) == pytest.approx(0.5, abs=1e-9)
    assert ppt.jmak_fraction(0.0, 1000.0) == 0.0
    assert ppt.jmak_fraction(-5.0, 1000.0) == 0.0
    assert ppt.jmak_fraction(1.0e7, 1000.0) <= 1.0
    assert ppt.jmak_fraction(1.0e7, 1000.0) > 0.999999
    assert ppt.jmak_fraction(100.0, None) == 0.0


def test_precipitated_fraction_monotone_and_bounded():
    """Kinetic fraction is non-decreasing in time and never exceeds the equilibrium amount."""
    M0, C0, N0 = 0.03, 0.18, 0.005
    # the equilibrium-precipitated fraction at T IS the sequential model's "undissolved"
    # (tied-up-in-precipitate) fraction
    eq = float(ppt.undissolved_fraction("Nb", 900.0, M0, C0, N0))
    assert eq > 0.5, "test premise: Nb must be mostly precipitated at 900 C"
    t = np.linspace(0.0, 1.0e6, 200)
    f = ppt.precipitated_fraction_at("Nb", 900.0, t, M0, C0, N0)
    assert f.shape == t.shape
    assert np.all(np.diff(f) >= -1e-12), "kinetic fraction must not decrease with time"
    assert np.all(f >= 0.0) and np.all(f <= eq + 1e-12), "bounded in [0, eq_fraction]"
    # long holds converge to the equilibrium amount (kinetics modulate speed, not extent)
    assert f[-1] == pytest.approx(eq, rel=1e-3)
    # no driving force -> nothing precipitated on any practical timescale
    assert ppt.precipitated_fraction_at("Nb", t_sol_C_high(), t, M0, C0, N0).max() <= 1e-6


def t_sol_C_high():
    """A temperature 150 K above NbN's solution temperature (fully dissolved region)."""
    return _t_sol("NbN", 0.03, 0.005) - 273.15 + 150.0


def test_stage_path_dissolves_then_reprecipitates():
    """Rolling path: precipitate cold -> dissolve on reheating above T_sol -> re-precipitate.

    Asserts the direction of the state actually changes sign twice.
    """
    stages = [(600.0, 3600.0), (1250.0, 600.0), (900.0, 600.0)]
    out = ppt.stage_path("Nb", stages, 0.03, 0.18, 0.005)
    assert len(out) == 3
    fr = [s["fraction"] for s in out]
    modes = [s["mode"] for s in out]
    assert modes == ["precipitate", "dissolve", "precipitate"], modes
    assert fr[0] > 0.5, "cold hold must precipitate most of the Nb"
    assert fr[1] < fr[0] and fr[1] < 0.1, "reheating above T_sol must dissolve the precipitate"
    assert fr[2] > fr[1] + 0.5, "cooling back must re-precipitate"
    assert all(0.0 <= f <= 1.0 for f in fr)
    # on precipitation stages the state never overshoots the stage's equilibrium fraction
    for s in out:
        if s["mode"] == "precipitate":
            assert s["fraction"] <= s["equilibrium"] + 1e-6
    # pure numpy: per-stage dicts carry the temperature and time back out
    assert out[0]["T_C"] == 600.0 and out[0]["time_s"] == 3600.0


def test_stage_path_hold_and_bounds():
    """A zero-time stage is a hold; fractions always stay in [0, 1]."""
    stages = [(900.0, 0.0), (900.0, 3600.0)]
    out = ppt.stage_path("Nb", stages, 0.03, 0.18, 0.005)
    assert out[0]["mode"] == "hold" and out[0]["fraction"] == 0.0
    assert 0.0 <= out[1]["fraction"] <= 1.0
    assert out[1]["fraction"] > out[0]["fraction"]
    # empty path is legal
    assert ppt.stage_path("Nb", [], 0.03, 0.18, 0.005) == []


def test_rolling_stages_from_schedule():
    """Stage lists derive from the sidebar schedule, per route, with the specified shapes."""
    plate_raw = {"FM_Temp_Before_Rolling": 985.0, "FM_Finish_Temp": 832.0,
                 "ACC_Temp_Post_Laminar": 744.0}
    st_plate = ppt.rolling_stages("Plate", plate_raw)
    assert [s[0] for s in st_plate] == ["Roughing exit", "Finishing", "ACC", "Coil/stack hold"]
    assert [s[1] for s in st_plate] == [985.0, 832.0, 744.0, 644.0]
    assert all(dt > 0.0 for _l, _t, dt in st_plate)
    hrc_raw = {"Temp_Transfer_Bar_C": 937.0, "Temp_FT_Avg_C": 835.0,
               "Temp_CT_Top_C": 595.0, "Temp_CT_Bot_C": 558.0}
    st_hrc = ppt.rolling_stages("HRC", hrc_raw)
    assert [s[0] for s in st_hrc] == ["Entry", "Finishing", "Runout", "Coil hold"]
    assert st_hrc[2][1] == pytest.approx((595.0 + 558.0) / 2.0)
    assert st_hrc[3][1] == st_hrc[2][1]
