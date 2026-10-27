# Specification Limit & Mechanical Models

I roll plate and coil for a living, and the question I keep getting asked is what a heat will
certify before it is rolled. This repository is my attempt at answering it with the models I can
defend, for both routes at the plant: heavy plate off the roughing and finishing mills, and Steckel
mill coil off the reversing mill.

Plate: 23,845 plates, 84 logged columns, ten certified properties including Charpy at 0, -20 and
-40 C and through-thickness reduction of area. Coil: 20,298 coils, three properties.

## What the numbers are

Split by melt, not by row. `GroupShuffleSplit` on the heat identifier keeps sister plates from one
cast on the same side of the boundary, which is stricter than a random split and the reason these
scores sit lower than a naive fit would show.

| route | target | R2 | RMSE | inside tolerance |
|---|---|---|---|---|
| Plate | tensile, as rolled | 0.9435 | 13.50 MPa | 89.6 % (±20 MPa) |
| Plate | yield, as rolled | 0.9204 | 16.97 MPa | 81.5 % (±20 MPa) |
| Plate | elongation, as rolled | 0.6628 | 2.10 % | 100 % |
| Plate | tensile, normalised | 0.6331 | 16.36 MPa | 79.6 % |
| Plate | Z-test | 0.6169 | 4.41 % | 98.8 % |
| Coil | tensile | 0.7954 | 18.73 MPa | 76.6 % (±20 MPa) |
| Coil | elongation | 0.6356 | 2.73 % | 65.4 % (±2.5 %) |
| Coil | yield | 0.6146 | 26.86 MPa | 60.7 % (±20 MPa) |

Plate covers 353 held-out heats and 7,934 measured plates; coil covers 919 heats and 2,221 coils.
Every score carries a 95 % bootstrap interval in the per-route reports.

Coil scores lower for a reason I can name: a reversing mill with heated coiler boxes leaves less
thermal history to learn from than roughing plus finishing, and there is no normalising step.

## Notebooks

- `Plate/notebooks/Plate_Rolling_ML_Analysis.ipynb`
- `HRC/notebooks/HRC_Steckel_ML_Analysis.ipynb`
- `notebooks/Dual_Mill_Mechanical_Property_Storytelling.ipynb`

The per-route notebooks run from the cleaning log through the physics features, the nine-family
model ladder, parity and residual plots on held-out melts, bootstrap intervals and permutation
importance. They read the shipped reports and models, so they cannot drift from what the app serves.

## Features

Pickering solid solution strengthening, Hall-Petch `1/sqrt(t)`, true strain, carbon equivalents,
Ar3, and the margin between finishing temperature and Ar3. The last passes are intercritical or they
are not, and that is what decides whether properties scatter.

## The app

`app/streamlit_app.py` has eight sections: simulator, specification check against the GRP standard
matrix, 2D and 3D response surfaces, a capability study with Cp and Cpk over a drift band you set,
batch scoring from a template, a refit tab that trains on an uploaded sheet and logs every run,
diagnostics, and an about page.

```cmd
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

Deployed entry point is `app/streamlit_app.py`. The repository is public; the campaign sheets and the
specification workbook are not.

## Limits

These models describe two mills and the grade families already rolled on them. Outside those ranges
they extrapolate. Toughness and normalised yield carry the widest intervals, and the
after-normalising targets are a different physical problem: the heat treatment does work the rolling
never did.

MIT licence.
