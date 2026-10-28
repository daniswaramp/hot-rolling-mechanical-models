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
.devcontainer/            dev container environment configuration
```

`script/ml/` is a library, not a CLI: `config`, `data_prep`, `features`, `models`, `evaluate`. The
notebooks and the app both import it. Training runs live in the notebooks.

## Invariants

1. **No training CLIs and no `.bat` launchers in the public repo.** Removed deliberately; do not reintroduce wrappers
   around train or tune here (training pipelines and batch scripts remain exclusively in the local workspace).
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
6. **No `graphify-out/` here.** It was deleted from the public mirror to prevent leaking local paths.
   Project-level GraphRAG is maintained exclusively in the local root.
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
10. **Do not override `span` typography in CSS.** Streamlit expander toggles and icon buttons use
   Material Symbols ligatures (`_arrow_right`, `_arrow_down`) rendered inside `<span>` tags. Overriding
   `span { font-family: ... !important; }` prevents the ligature from rendering as a glyph, resulting
   in raw text overlapping headers.
11. **Sidebar schedule inputs must strictly match active model features.** Plate schedule inputs in
   `PLATE_SPEC` feed `Plate/script/ml/features.py`. Non-model variables (`Slab_W_mm`, `FM_Number_of_Passes`,
   `FM_Last_Pass_Thick`, `Chem_B`) were removed to avoid confusing operators. Finishing drafting is wired
   directly via `FM_Last_Pass_Drafting`.
12. **App title and legal status.** The deployed platform title is `"Specification Limit & Mechanical Models"`.
   The About tab disclaimer states independent exploratory study status without commercial endorsement
   from PT Gunung Raja Paksi Tbk.
13. **Model Engine Hierarchy & Champion GBDT Default.** The application defaults to the Champion
   GBDT architecture (`models/boosted_out/`), selecting the highest-scoring native CatBoost (`.cbm`),
   XGBoost (`.ubj`), or LightGBM (`.txt`) booster per target. Standalone boosters match or exceed
   the stacked meta-learner on melt holdouts (e.g., HRC TS $R^2 = 0.792$ vs $0.750$; Plate YS_AN
   $R^2 = 0.402$ vs $0.309$) and load instantly (<1 second) without scikit-learn unpickling overhead. The UI sidebar
   provides a Model Engine selector (`[Champion GBDT (Recommended), CatBoost, XGBoost, LightGBM, Ensemble (.joblib), OLS / Linear]`),
   allowing operators to toggle architectures dynamically across Simulator, Spec Check, Response Surface,
   Capability, and Batch tabs.
14. **Dual-Directory Repository Architecture.**
   - `push-to-GH/`: Clean public GitHub mirror connected to Streamlit Community Cloud (`origin/main`).
     Strictly zero private credentials and zero raw Excel campaign sheets.
   - Main local directory: Dedicated development environment containing raw Excel sources, local ETL,
     and Modal serverless cloud compute runners.
15. **Target Outlier Invariant & Decimal Rescaling.** In `Plate/script/ml/data_prep.py`, `prepare_target_dataset`
   strictly validates targets against `PHYSICAL_BOUNDS` and automatically rescales $100\times$ decimal omission
   typos ($41,205 \rightarrow 412.05\text{ MPa}$; $28,205 \rightarrow 282.05\text{ MPa}$), ensuring that
   least-squares or tree models are never poisoned by data entry errors.

## Environment

`requirements.txt` specifies `streamlit>=1.38.0,<=1.43.0`, `scikit-learn>=1.5.0`, `xgboost`, `lightgbm`,
and `catboost`. Native boosted models in `models/boosted_out/` bypass scikit-learn version constraints
and load across Python 3.10 through 3.13 without warnings. LightGBM `.txt` models are pinned to LF line
endings via `.gitattributes` to preserve binary tree byte offsets.

Entry point for deployment: `app/streamlit_app.py`, branch `main`. The sidebar carries a build stamp
and an instant boot self-test (`Plate: 10/10 targets scoring`, `HRC: 3/3`). Check both before believing a bug
report — a stale container is the most common cause of a broken app here.

## Validation contract

`GroupShuffleSplit` on `Heat_ID`. Sister plates from one melt never straddle the split. Melt-grouped
bootstrap intervals sit beside every score. Never report a row-split score as if it were comparable.

## Resolved & Known Items

- **Target Decimal Typo Trap (Resolved)**: The $41,205\text{ MPa}$ and $28,205\text{ MPa}$ decimal shift
  outliers have been cleaned in the source CSV and permanently guarded in `data_prep.py` with automatic
  $100\times$ rescaling against `PHYSICAL_BOUNDS`.
- **Capability (Cp/Cpk)**: Samples a box of independent uniform inputs. Real rolling schedules correlate,
  so it represents a conservative envelope under stated operational drift assumptions.
