# Handoff

For whoever picks this up next. `AGENTS.md` holds the invariants; this file holds the state, the
reasons and the traps.

## Run it

```cmd
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

Startup is instant (<1 second) thanks to native Champion GBDT models (`models/boosted_out/`). Nothing reads the
campaign sheets at runtime, so the app runs on a fresh clone with no data.

Sheets live in `Plate/csv/` and `HRC/csv/` (git-ignored). The specification workbook is published in
`specification-list/` on purpose: the Specification check tab reads it at runtime, so removing it
would break that tab on Streamlit Cloud. Only the raw campaign heats stay private.

## Deployment

Streamlit Community Cloud, repo `daniswaramp/hot-rolling-mechanical-models`, branch `main`, entry
`app/streamlit_app.py`. The sidebar prints a build stamp and an instant boot self-test (`Plate: 10/10`,
`HRC: 3/3`). Check it before believing a bug report; a stale container has caused every false alarm
so far.

## Decisions

- Melt-grouped split. Costs two to three points of R2 against a row split and it is the only number
  worth quoting.
- Nine model families per target, chosen by leaderboard.
- Default to **Champion GBDT**: Native CatBoost (`.cbm`), XGBoost (`.ubj`), and LightGBM (`.txt`)
  models in `models/boosted_out/` provide faster execution, eliminate unpickling version warnings,
  and match/beat stacked meta-learners on holdout metrics.
- Sidebar Model Engine selector allows toggling between Champion GBDT, CatBoost, XGBoost, LightGBM,
  Ensemble (.joblib), and OLS/Linear.
- Bootstrap intervals by melt, shown beside every prediction instead of a bare R2.
- Training CLIs and batch launchers removed from public mirror (maintained locally in workspace root).
- Campaign sheets withheld. The history was squashed to one commit, because making the repo public
  alone would still have served the old blobs.
- Specification workbook published in `specification-list/` (see above: the app reads it at runtime).
- `app/physics.py` is the single source for the formulas the derived-metrics card displays. The
  feature modules are frozen and intentionally differ from it in places (see trap 7).
- Normalizing coupling is applied in exactly one dict site (`predict_selected_engine`) and one
  frame site (`score_frame`); the loader's `predict_single_sample` no longer couples, it only
  clamps a finite prediction.
- The OOD envelope in `models/applicability_domain.json` was computed from the raw logged columns.
  Rebuild it from the cleaned tables with `python app/build_applicability_domain.py` where the
  campaign sheets live; its `_caveats` block lists what is dropped and why.

## Traps

1. Streamlit's header is absolutely positioned and 60 px tall. Less than about 4.4 rem of container
   padding and the first line looks cropped.
2. `st.logo()` is the only reliable place for the mark. Image tags inside markdown got clipped.
3. The shipped pickles reference `script.ml.tune_bootstrap.StackingEnsemble`, and both mills use that
   name with different internals. Delete the shim and every model fails to load; assume one layout
   and the coil route dies.
4. Normalised targets were fitted on feature subsets. Plate bundles carry `features` (25 columns on
   YS_AN/TS_AN/EL_AN), HRC bundles carry `feature_names` (40). Always resolve through
   `ml.bundle_feature_columns(bundle, X)` — feeding the full engineered frame raises a
   feature-name mismatch, and a bare `except` used to turn that into a plausible-looking
   substituted number.
5. Target outlier trap (resolved): `data_prep.py` now enforces `PHYSICAL_BOUNDS` and automatically
   rescales $100\times$ decimal omission errors (41,205 -> 412.05).
6. LightGBM `.txt` models require UNIX LF line endings for binary byte offset alignment; `.gitattributes`
   enforces `eol=lf` to prevent Windows CRLF conversions.
7. The fitted Plate `sigma_ss` is 32.3 Mn + 83.2 Si + **354 P** + 38 Cu + **33 Ni** + 11 Mo − 30 Cr,
   with no Ti or N term, and its Ar3 carries a gauge term. The fitted HRC and documented Pickering
   forms differ (43 Ni, 80 Ti, 354 N, no gauge term). These are what the models were trained on, so
   they are frozen; the card uses `app/physics.py`. Changing a feature module without retraining
   silently shifts every prediction for that mill.
8. A selectbox whose options shrink while its widget key persists (response-surface Y axis, engine
   dropdown, spec grade) raises StreamlitAPIException on a stale value. `prune_stale_widget_key`
   clears it; use it whenever an options list is computed from another widget.
9. Widget bounds must come from the mill's own spec table. The optimizer's gauge input used to
   hardcode min 3.0 mm and crashed the Steckel tab on thin coil.

## Not done

- CV entry for this project.
- The old Streamlit app and its subdomain.
- Capability samples a box of independent inputs, where real schedules correlate.
