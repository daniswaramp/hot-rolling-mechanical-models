# Handoff

For whoever picks this up next. `AGENTS.md` holds the invariants; this file holds the state, the
reasons and the traps.

## Run it

```cmd
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

First load takes 20 to 30 seconds: ten model bundles for plate, three for coil. Nothing reads the
campaign sheets at runtime, so the app runs on a fresh clone with no data.

Sheets live in `Plate/csv/` and `HRC/csv/` (git-ignored). The specification workbook lives in
`specification-list/` and is also ignored, because it is a GRP standard and this repository is
public.

## Deployment

Streamlit Community Cloud, repo `daniswaramp/hot-rolling-mechanical-models`, branch `main`, entry
`app/streamlit_app.py`. The sidebar prints a build stamp and a boot self-test (`Plate: 10/10`,
`HRC: 3/3`). Check it before believing a bug report; a stale container has caused every false alarm
so far.

## Decisions

- Melt-grouped split. Costs two to three points of R2 against a row split and it is the only number
  worth quoting.
- Nine model families per target, chosen by leaderboard. Stacking won on plate, bagging on coil.
- Bootstrap intervals by melt, shown beside every prediction instead of a bare R2.
- Training CLIs and all ten `.bat` launchers removed. Training is the notebooks; `script/ml/` stays
  because both the notebooks and the app import it.
- Campaign sheets withheld. The history was squashed to one commit, because making the repo public
  alone would still have served the old blobs.
- Specification workbook withheld for the same reason.
- Linear families (OLS, Ridge) committed as coefficients in `models/linear_families.json` so a
  reader can compare a readable equation against the ensemble. Anything whose holdout R2 falls below
  0.15 is not served at all, marked with a dash instead.

## Traps

1. Streamlit's header is absolutely positioned and 60 px tall. Less than about 4.4 rem of container
   padding and the first line looks cropped.
2. `st.logo()` is the only reliable place for the mark. Image tags inside markdown got clipped.
3. The shipped pickles reference `script.ml.tune_bootstrap.StackingEnsemble`, and both mills use that
   name with different internals. Delete the shim and every model fails to load; assume one layout
   and the coil route dies.
4. Normalised targets were fitted on feature subsets. Try `feature_names_in_` first or they return
   NaN with no error. The loader swallows exceptions into NaN, which is how a broken route shipped
   twice.
5. The cleaned plate CSV still holds a yield strength of 41,205 MPa, and two more rows like it. Any
   least-squares fit that includes them is useless: one of them turned a 0.90 fit into -1.36. The
   refit tab filters physical bounds; the ETL does not.
6. `scikit-learn==1.9.1` is pinned because that build wrote the artifacts.

## Not done

- CV entry for this project.
- The old Streamlit app and its subdomain.
- ETL fix for the implausible measured values.
- Capability samples a box of independent inputs, where real schedules correlate.
