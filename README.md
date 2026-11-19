# Specification Limit & Mechanical Models: Dual-Mill Industrial Steel Platform

I roll plate and coil for a living, and the question I keep getting asked is what a heat will
certify before it is rolled. This repository is my attempt at answering it with the models I can
defend, for both routes at the plant: heavy plate off the roughing and finishing mills, and Steckel
mill coil off the reversing mill.

Plate: 23,845 plates, 84 logged columns, ten certified properties including Charpy at 0, -20 and
-40 °C and through-thickness reduction of area. Coil: 20,298 coils logged, 19,116 usable after
the sanitising pass removed 1,182 duplicate records, three properties. Both tables carry a
written audit trail (`reports/sanitise_log.csv` stays out of the public mirror with the heats).

---

## What the numbers are

Split by melt, not by row. `GroupShuffleSplit` on the ladle heat identifier keeps sister plates from one
cast on the same side of the boundary, which is stricter than a random split and the reason these
scores sit lower than a naive fit would show.

| route | target | R2 (stacked ensemble) | R2 (served Champion GBDT) | 95% Conformal Width |
|---|---|---|---|---|
| Plate | tensile, as rolled | 0.9435 | 0.9419 | ±27.26 MPa (95.0% coverage) |
| Plate | yield, as rolled | 0.9204 | 0.9204 | ±33.96 MPa (95.0% coverage) |
| Plate | elongation, as rolled | 0.6628 | 0.4986 | ±4.46 % (95.0% coverage) |
| Plate | tensile, normalised | 0.6331 | 0.5695 | ±39.31 MPa (94.6% coverage) |
| Plate | yield, normalised | 0.4926 | 0.4020 | ±32.68 MPa (94.6% coverage) |
| Plate | elongation, normalised | 0.6533 | 0.5709 | ±5.04 % (94.6% coverage) |
| Plate | Charpy 0 °C | 0.3799 | 0.3533 | ±99.4 J (94.7% coverage) |
| Plate | Charpy −20 °C | 0.3033 | 0.2983 | ±121.36 J (94.8% coverage) |
| Plate | Charpy −40 °C | 0.4266 | 0.3716 | ±118.77 J (92.9% coverage) |
| Plate | Z-test | 0.6169 | 0.5667 | ±9.95 % (94.2% coverage) |
| Coil | tensile | 0.7954 | 0.7922 | ±36.91 MPa (95.0% coverage) |
| Coil | elongation | 0.6356 | 0.6306 | ±5.43 % (95.0% coverage) |
| Coil | yield | 0.6146 | 0.6097 | ±53.25 MPa (95.0% coverage) |

The app serves **Champion GBDT** by default (the best of CatBoost/XGBoost/LightGBM per target,
from `models/boosted_out/`), so the "served" column is what a fresh simulation reports. The stacked
`.joblib` ensembles score a point or two higher on several targets and stay selectable in the
sidebar's engine dropdown; their numbers are the second column. Both are scored on the same
held-out melts, and the conformal widths beside every prediction were calibrated on the champion
residuals.

Plate covers 353 held-out heats and 7,934 measured plates; coil covers 919 heats and 2,221 coils.
Every score carries a 95 % bootstrap interval and a finite-sample calibrated split-conformal prediction interval.

Coil scores lower for a reason I can name: a reversing mill with heated coiler boxes leaves less
thermal history to learn from than roughing plus finishing, and there is no normalising step.

---

## Key Platform Features

### 1. Split-Conformal Prediction Intervals
Unlike naive Gaussian standard error formulas (`1.96 * RMSE / sqrt(n)`) which achieve only ~25% coverage on individual rolled products, our uncertainty quantification uses rigorous **inductive conformal prediction calibrated on held-out ladle melts**. Every prediction guarantees $\ge 95\%$ empirical coverage on future heats.

### 2. Thermodynamic Softening Coupling & Domain Invariants
Normalizing furnace heat treatment relieves work hardening and recrystallizes austenite grains into equiaxed ferrite-pearlite. The engine enforces thermodynamic coupling:
$$TS_{AN} < TS_{BN}, \quad YS_{AN} < YS_{BN}, \quad EL_{AN} > EL_{BN}$$
The softening is physically coupled to the incoming strain and TMCP cooling rate: faster-cooled plates with greater work hardening experience proportionally larger softening upon normalizing. In addition, the engine flags normalizing requests on mild structural grades ($CE < 0.38$), advising that industrial normalizing is standard for pressure vessel boiler grades ($CE \ge 0.38$, e.g. SA 516 Gr 70).

### 3. TreeSHAP Attribution & Metallurgical Factor Decomposition
Prediction explainability is powered by native C-level TreeSHAP (<20 ms). Individual property predictions are decomposed into 4 interpretable metallurgical drivers:
- **Base Steel & Solid Solution Strengthening**: Carbon, Manganese, Pickering lattice distortion.
- **Grain Refinement & Microalloy Precipitation**: Hall-Petch effect ($1/\sqrt{t}$), Niobium, Vanadium, Titanium.
- **True Deformation Strain**: Logarithmic drafting across roughing and finishing.
- **Thermomechanical Cooling**: Finish rolling temperature, laminar spray cooling rate, and transformation supercooling ($FRT - Ar_3$).

### 4. Inverse Setpoint Optimizer (Reverse Design)
Given target specification limits (e.g. ASTM A 572 Gr 50, JIS G 3101 SS400, SM490B, ASME SA 516 Gr 70) and plate gauge, the optimizer evaluates discrete candidate schedules in vectorized batches (<150 ms). It solves for the optimal chemistry, finish rolling temperature, and laminar cooling headers that achieve statistical compliance ($C_{pk} \ge 1.33$ conformal margin) while minimizing Carbon Equivalent ($CE_{IIW}$) and microalloy cost.

### 5. Applicability Domain & Out-Of-Distribution (OOD) Monitor
A continuous envelope monitor checks input parameters against historical P01–P99 mill percentiles across 20 Plate and 21 Steckel features, preventing hazardous extrapolation.

### 6. Pass Schedule Recrystallization & Penetration Screening
The **Pass schedule** tab classifies every pass of a drafted sequence into its softening regime — dynamic (DRX, metadynamic folded in), full static (SRX), partial (PRX) or none (NRX) — with a Sellars-type chain (Zener–Hollomon, peak/critical strain, solute- and Nb(C,N)-pinning-retarded interpass recrystallization) and flags whether each draft penetrates the full thickness via the contact-length rule of thumb. It runs locally in milliseconds behind its own "Analyze schedule" button and replays from session state.

---

## Artifacts & Verification

- **Model Card**: [`docs/MODEL_CARD.md`](docs/MODEL_CARD.md) (Mitchell et al. compliant specification)
- **Data Card**: [`docs/DATA_CARD.md`](docs/DATA_CARD.md) (Gebru et al. compliant dataset governance)
- **Model Registry**: [`models/registry.json`](models/registry.json) (Cryptographic SHA-256 hashes & conformal widths)
- **Automated Test Suite**: `tests/` verifying schema contracts, physical invariants, conformal coverage, zero heat leakage, and the HRC route regression harness. The two zero-leakage tests skip on a clone without the private cleaned tables; everything else runs in CI.
- **Hostile QA Battery**: 34 adversarial test cases (boundary fuzzing, string types, cryogenic/extreme temperatures, and model engine parity) run locally against the deployed build; they are not shipped in this public mirror.

---

## Notebooks

- `Plate/notebooks/Plate_Rolling_ML_Analysis.ipynb`
- `HRC/notebooks/HRC_Steckel_ML_Analysis.ipynb`
- `notebooks/Dual_Mill_Mechanical_Property_Storytelling.ipynb`

The per-route notebooks run from the cleaning log through the physics features, the nine-family
model ladder, parity and residual plots on held-out melts, bootstrap intervals and permutation
importance. They read the shipped reports and models, so they cannot drift from what the app serves.

---

## Features

Pickering solid solution strengthening, Hall-Petch `1/sqrt(t)`, true strain, carbon equivalents,
Ar3, and the margin between finishing temperature and Ar3. The last passes are intercritical or they
are not, and that is what decides whether properties scatter.

---

## The App

`app/streamlit_app.py` has nine modular sections:
1. **Simulator & TreeSHAP Attribution**: Single-plate forward prediction with metallurgical explainability.
2. **Inverse Setpoint Optimizer**: Reverse engineering optimal rolling schedules to target specs.
3. **Specification Check**: Compliance verification against international standards (ASTM, JIS, ASME, EN).
4. **2D & 3D Response Surfaces**: Interactive physical contour landscapes.
5. **Capability Study ($C_p / C_{pk}$)**: Monte Carlo process capability under rolling drift.
6. **Batch Scoring**: Fast multi-plate CSV inference with template download.
7. **Refit & AutoML**: In-browser training on newly uploaded campaign data.
8. **Diagnostics & Conformal Calibration**: Parity, residuals, and prediction interval coverage audit.
9. **About & Domain Knowledge**: Mill layouts, physics formulas, and system architecture.

```cmd
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

Deployed entry point is `app/streamlit_app.py`. The repository is public; the campaign sheets
(`Plate/csv/`, `HRC/csv/`) are not. The specification workbook **is** published in
`specification-list/` because the Specification check tab reads it at runtime.

### Serverless compute (Modal)

The three heavy calculations (recipe search, batch scoring, capability study) can run on Modal
instead of the Streamlit container. They only ever run when the operator presses the matching
button (`Calculate recipe`, `Calculate scores`, `Calculate capability`); nothing is computed on
rerun, at boot, or while editing inputs. If Modal is not configured or a call fails, the app falls
back to local execution and says so under the result.

One-time setup:

```cmd
modal token new
modal deploy modal_app.py
```

Then add the credentials to the deployment (Streamlit Cloud: Manage app > Settings > Secrets):

```toml
MODAL_TOKEN_ID = "ak-..."
MODAL_TOKEN_SECRET = "as-..."
```

Locally, put the same two lines in `.streamlit/secrets.toml` (gitignored; never commit it). The
app is viewer-gated, but treat those tokens as the spending key: any viewer's button press uses
credits, so share the app only with people who should be able to run campaigns.

---

## Limits

These models describe two mills and the grade families already rolled on them. Outside those ranges
they extrapolate. Toughness and normalised yield carry the widest intervals, and the
after-normalising targets are a different physical problem: the heat treatment does work the rolling
never did.

The input widgets now accept transfer-bar thickness well beyond each mill's historical envelope
(Plate up to 120 mm, coil up to 105 mm), and the feature modules no longer saturate silently above
the old training ceilings. The applicability-domain monitor flags such rows as extrapolation;
predictions there carry the model's own conformal interval but no training support.

MIT licence.
