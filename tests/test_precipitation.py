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
