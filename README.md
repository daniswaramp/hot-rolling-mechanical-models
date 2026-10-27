# Dual-mill mechanical property modelling

Two production routes, one question: before a heat is rolled, what will it certify?

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/daniswaramp/hot-rolling-mechanical-models/blob/main/Plate/notebooks/Plate_Rolling_ML_Analysis.ipynb)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)

- **Plate/** — heavy plate, 6–120 mm, from the rougher through the finisher and laminar ACC, delivered
  as-rolled/TMCP or normalised. 23,845 plates, 84 logged columns.
- **HRC/** — Steckel mill coil, 2.5–16 mm, rougher into a reversing Steckel mill with heated coiler
  boxes, runout table cooling, downcoiler. 20,298 coils, 35 columns.

Both routes predict yield strength, tensile strength and elongation. Plate additionally predicts the
normalised condition, Charpy impact at 0/−20/−40 °C, and through-thickness reduction of area.

## What is published, and what is not

Published: the trained model artifacts, the aggregated reports (leaderboards, bootstrap intervals,
validation predictions, cleaning logs) and both notebooks with every figure and score rendered.

Not published: the campaign sheets themselves. They carry live grade chemistry and rolling schedules
for the mill, so they stay out of this repository and out of its history. The notebooks therefore
carry **pre-generated outputs** — read them as a record of the full-data run, not as a live
recomputation. Running a notebook against your own schedule works if you drop a CSV at the path named
in its first cell.

## Results

Held-out scores use `GroupShuffleSplit` on `Heat_ID`, so sister plates from one melt never straddle
the split. That is the number worth quoting; a random row split leaks melt identity and flatters
everything.

| route | target | R² | RMSE | within tolerance |
|---|---|---|---|---|
| Plate | Tensile, as rolled | 0.9435 | 13.50 MPa | 89.6 % (±20 MPa) |
| Plate | Yield, as rolled | 0.9204 | 16.97 MPa | 81.5 % (±20 MPa) |
| Plate | Elongation, as rolled | 0.6628 | 2.10 % | 100 % |
| Plate | Tensile, normalised | 0.6331 | 16.36 MPa | 79.6 % |
| Plate | Z-test reduction of area | 0.6169 | 4.41 % | 98.8 % |
| HRC | Tensile | 0.7954 | 18.73 MPa | 76.6 % (±20 MPa) |
| HRC | Elongation | 0.6356 | 2.73 % | 65.4 % (±2.5 %) |
| HRC | Yield | 0.6146 | 26.86 MPa | 60.7 % (±20 MPa) |

Plate: 353 held-out heats, 7,934 measured plates. HRC: 919 held-out heats, 2,221 coils. Every score
carries a 95 % bootstrap interval in the per-route reports.

The gap between routes is not incidental. The Steckel route has less thermal history to learn from —
two reversing stands and a coiler box instead of a roughing-plus-finishing sequence — and no
normalising step, which is why its yield model sits at 0.61 while plate tensile reaches 0.94.

## Notebooks

- [`Plate/notebooks/Plate_Rolling_ML_Analysis.ipynb`](Plate/notebooks/Plate_Rolling_ML_Analysis.ipynb)
- [`HRC/notebooks/HRC_Steckel_ML_Analysis.ipynb`](HRC/notebooks/HRC_Steckel_ML_Analysis.ipynb)
- [`notebooks/Dual_Mill_Mechanical_Property_Storytelling.ipynb`](notebooks/Dual_Mill_Mechanical_Property_Storytelling.ipynb) — both routes side by side

Each per-route notebook runs top to bottom: cleaning log, physics features with their equations, where
the properties come from, the nine-family model ladder, parity and residual plots on held-out melts,
bootstrap intervals, and permutation importance. They read the shipped reports and models, so they
cannot drift from what the app serves.

## Features

The feature set is metallurgical rather than a dump of the log: Pickering solid-solution
strengthening, Hall–Petch `1/sqrt(t)`, true strain `ln(t_slab/t_plate)`, carbon equivalents, `Ar3`,
and the margin between finishing temperature and `Ar3` — the last passes are intercritical or they are
not, and that decides whether properties scatter.

## The app

`app/streamlit_app.py` serves both routes in five sections:

- **Simulator** — one schedule in, every certified property out, each with the interval its bootstrap
  report implies, plus the physical indicators (Pickering σ_ss, CE, Pcm, Ar3, true strain) and where
  the finishing temperature sits relative to Ar3.
- **Response surface** — pick any two schedule variables and a property; 2D contour or rotated 3D
  surface. Every node is a full pass through the mill's own feature module, and the current schedule
  is marked on the field.
- **Batch** — download a template, fill one row per plate or coil, upload, download the scored sheet.
- **Retrain** — refit a fresh model for one measured property from your own sheet, on a melt-grouped
  holdout. The shipped stacking ensembles are baked artefacts, not incrementally trainable, so this
  trains new models rather than pretending to fine-tune the old ones.
- **Diagnostics** — leaderboards, 95 % bootstrap intervals and the physical formulas as stored.

```cmd
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

Streamlit Community Cloud entry point is `app/streamlit_app.py`. The repository is private; the app
is public.

## Layout

```text
Plate/            cleaned CSV, models, reports, notebooks, script/ml
HRC/              same, Steckel route
app/              streamlit_app.py, model_loader.py, views/
notebooks/        dual-route storytelling notebook
requirements.txt
```

`script/ml/` is the library the notebooks and the app both import: `data_prep`, `features`, `models`,
`evaluate`, `config`. Training now runs from the notebooks; the old CLI wrappers and their `.bat`
launchers were removed.

`Plate/csv/` and `HRC/csv/` exist locally but are git-ignored — that is where the campaign sheets go
if you want to re-run the notebooks.

## Limits

Scores describe these two mills and the grade families already rolled on them. Outside those ranges
the model is extrapolating. Toughness and normalised yield carry the widest intervals, and the
after-normalising targets are a different physical problem — the heat treatment does work the rolling
never did — so their lower scores are structural rather than a tuning failure.

MIT licence. Built from Gunung Raja Paksi production logs.
