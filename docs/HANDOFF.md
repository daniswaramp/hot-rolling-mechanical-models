# Handoff — state, decisions, and how to work here

Written for whoever picks this up next, human or agent. `AGENTS.md` holds the invariants; this file
holds the story, the numbers and the traps.

## Run it

```cmd
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

Local first load takes 20–30 s: ten `.joblib` bundles for plate, three for coil. Nothing reads the
campaign sheets at runtime — the app needs models and `reports/` only, which is why it runs on a
fresh clone with no data.

The campaign sheets live at `Plate/csv/plate_rolling_2026_cleaned.csv` and
`HRC/csv/steckel_mill_cleaned.csv`, both git-ignored. The notebooks and any refit need them; the app
does not.

## Deployment

Streamlit Community Cloud, repo `daniswaramp/hot-rolling-mechanical-models`, branch `main`, entry
`app/streamlit_app.py`. Repo public, sheets out of history. The sidebar prints a build stamp and a
boot self-test (`Plate: 10/10`, `HRC: 3/3`); check it before believing a bug — a stale container has
caused every false alarm so far. Reboot from Manage app when in doubt.

## What was decided, and why

- **Melt-grouped split.** `GroupShuffleSplit` on `Heat_ID`. Sister plates from one cast never
  straddle the boundary. Costs about 0.02–0.03 R² against a row split and is the only number worth
  quoting.
- **Nine model families per target**, chosen by leaderboard: OLS, metallurgical NLR, Ridge, XGBoost,
  LightGBM, CatBoost, stacking, bagging. Stacking won on plate, bagging on coil.
- **Bootstrap intervals** by melt, stored beside every score, shown in the app rather than a bare R².
- **CLI training removed, notebooks kept.** Ten `.bat` launchers and four entry-point scripts were
  deleted. Training is the notebooks now; `script/ml/` stayed as a library because both the notebooks
  and the app import it.
- **Campaign data withheld.** Repo made public for the portfolio, so the sheets were dropped and the
  history squashed to one commit — a visibility flip alone would still have served the old blobs.
  Notebooks ship pre-generated outputs with a note saying so and a guard that explains the missing
  file instead of throwing `FileNotFoundError`.
- **Capability tab** reports Cp/Cpk over a user-defined drift box, plus a per-input linear band with a
  ±3σ envelope. Stated assumption: inputs vary independently and uniformly, which real schedules do
  not.

## Numbers to quote

Plate, 353 held-out heats, 7,934 measured plates: tensile 0.9435 / 13.50 MPa, yield 0.9204 /
16.97 MPa, elongation 0.6628 / 2.10 points, normalised tensile 0.6331, Z-test 0.6169.
Coil, 919 held-out heats, 2,221 coils: tensile 0.7954 / 18.73 MPa, elongation 0.6356, yield 0.6146 /
26.86 MPa.

The coil numbers are lower for a physical reason — a reversing mill with coiler boxes carries less
thermal history than roughing plus finishing, and there is no normalising step to learn from.

## Traps, in the order they cost time

1. Streamlit's header is absolutely positioned and 60 px tall. Less than ~4.4 rem container padding
   and the first line looks cropped. Three separate "logo is cropped" reports came from this.
2. `st.logo()` is the only reliable place for the mark. Image tags inside markdown got clipped.
3. The shipped pickles reference `script.ml.tune_bootstrap.StackingEnsemble`, and both mills use that
   name with different internals. Delete the shim and every model fails; assume one layout and the
   coil route dies with `no attribute 'model_names'`.
4. Normalised targets were fitted on feature subsets. Try `feature_names_in_` first or they return
   NaN with no error. The loader's public function swallows exceptions into NaN — that is how a
   broken route shipped twice.
5. The cleaned plate CSV still holds a yield strength of 41,205 MPa. It wrecks any refit that includes
   it; the retrain tab now filters physical bounds, the ETL does not.
6. `scikit-learn==1.9.1` is pinned because that build wrote the artifacts.

## Not done

- CV entry for this project.
- Old Streamlit app still exists and its subdomain should be reused or retired; the archived repo
  `plate-rolling-property-model` is private and read-only, so any old public links are dead.
- ETL fix for the 41,205 MPa row.
- Capability sampling from the historical joint distribution instead of a box.
