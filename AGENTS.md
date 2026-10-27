# AGENTS.md — repository guide for the next agent

Dual-mill mechanical property modelling: heavy plate and Steckel mill coil. This file describes the
repository **as it is now**. Anything here that contradicts the code is a bug in this file.

## What the repo is

Two routes, one model layer:

- `Plate/` — heavy plate, 6–120 mm. 23,845 plates, 84 logged columns, ten targets (yield, tensile,
  elongation as-rolled and normalised, Charpy 0/-20/-40 C, through-thickness Z).
- `HRC/` — Steckel coil, 2.5–16 mm. 20,298 coils, 35 columns, three targets.

## Layout

```text
Plate/{csv,models,reports,notebooks,script/ml}   HRC/{same}
app/streamlit_app.py      the deployed app, single file, eight tabs
app/model_loader.py       artifact loading, feature wiring, physical indicators
app/spec_checker.py       standard lookup, tier matching, chemical & mechanical decency
specification-list/       SPECIFICATION LIST.xlsx (GRP.ST.QA.09 standard reference matrix)
assets/grp.png            header mark, padded so nothing clips
notebooks/                dual-route storytelling notebook
```

`script/ml/` is a library, not a CLI: `config`, `data_prep`, `features`, `models`, `evaluate`. The
notebooks and the app both import it. Training runs live in the notebooks.

## Invariants

1. **No training CLIs and no `.bat` launchers.** Removed deliberately; do not reintroduce wrappers
   around train or tune.
2. **`tune_bootstrap.py` must stay** even though it looks like dead code. The shipped `.joblib`
   files record their estimator classes under that module path; delete it and every model fails to
   unpickle. It now holds only the class definitions.
3. **Both mills pickle a class named `StackingEnsemble`** under the same import path, with different
   internals (plate: dict of base models plus `meta_learner`; coil: list plus `meta_model`). The shim
   handles both layouts. Do not "clean up" one of them.
4. **Trust the estimator's own feature list first.** Normalised targets were fitted on per-target
   subsets. `candidate_columns()` tries `feature_names_in_`, then `feature_names`, then `features`,
   then the full frame. A single candidate silently returns NaN for those targets.
5. **Campaign sheets never enter git.** `Plate/csv/` and `HRC/csv/` are ignored, and history was
   squashed to one commit so they cannot be recovered from the public repo. Keep it that way.
6. **No `graphify-out/` here.** It was deleted: the graph was stale and its hub names duplicated, so
   it described a structure that no longer existed. If you regenerate one, check it first — this repo
   is public and a graph must not carry absolute local paths.
7. **Streamlit's app header is absolutely positioned, 60 px tall.** Content needs about 4.4 rem of
   container padding or the first line looks clipped. Three separate "logo cropped" reports came from
   this one fact.
8. **Specification reference matrix (`specification-list/SPECIFICATION LIST.xlsx`) is public reference**,
   distinct from the git-ignored campaign sheets. `app/spec_checker.py` loads and caches it via
   `@st.cache_data`. It evaluates thickness against both standard gauge tier and GRP capability
   (`PROD THICK`), validates active chemistry, and evaluates mechanical property decency against 95%
   bootstrap confidence intervals.
9. **Unified white-theme table architecture (`table.ut`).** Simulator, Specification check, and
   Diagnostics share a single crisp HTML table style (`table.ut`), avoiding redundant columns
   (such as the old static tolerance/tautological dot), broken colspans, or unstyled mixed-unit
   dataframes.

## Environment

`requirements.txt` pins `scikit-learn==1.9.1` because that build wrote the artifacts. Streamlit
Community Cloud installs it; a local 1.8.x loads them with an `InconsistentVersionWarning`.

Entry point for deployment: `app/streamlit_app.py`, branch `main`. The sidebar carries a build stamp
and a boot self-test (`Plate: 10/10 targets scoring`, `HRC: 3/3`). Check both before believing a bug
report — a stale container is the most common cause of a broken app here.

## Validation contract

`GroupShuffleSplit` on `Heat_ID`. Sister plates from one melt never straddle the split. Melt-grouped
bootstrap intervals sit beside every score. Never report a row-split score as if it were comparable.

## Known defects

- The cleaned plate CSV still contains a yield strength of **41,205 MPa**. One such row turns a 0.86
  refit into R2 -93. The retrain tab rejects values outside physical bounds; the ETL still needs the
  fix at source.
- Capability (Cp/Cpk) samples a box of independent uniform inputs. Real schedules correlate, so it is
  an envelope under a stated assumption, not SPC capability.
