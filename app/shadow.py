"""Digital shadow: how far the model's prediction sits from measured results, per property.

This tab does not train anything and it does not touch a model file. It scores a sheet of
schedules whose mechanical results have already been measured, takes the residual
``measured - predicted`` per property, and reports the median residual (the model's bias on
that route) plus its p10/p90 spread. When the sheet carries a ``Grade`` column the same
summary is produced per grade, which is where a systematic offset usually hides: one grade
family is read 15 MPa low, the rest are fine.

The calibration is computed by ``remote_jobs.update_shadow`` (Modal app ``hrcm-jobs`` when it
is configured, this process otherwise) and only when "Update shadow" is pressed. The result
is kept in ``st.session_state[f"shadow_{mill}"]`` so a rerun replays it instead of rescoring
the sheet, and "Load last calibration" reads the stored JSON back from the Modal volume.

Applying the bias is deliberately a session-only display: the "Preview corrected predictions"
checkbox adds the stored median residual to a table computed locally for the current sidebar
schedule and labels it display-only. No model, no other tab and no stored artifact changes,
because a bias measured on one grade family must not silently refit the whole route.

``streamlit_app`` is imported lazily inside the helpers: it imports this module at boot, so a
module-level import here would close the cycle.
"""

import importlib
import os
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

APP_DIR = os.path.dirname(os.path.abspath(__file__))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

import remote_jobs  # noqa: E402
try:
    importlib.reload(remote_jobs)
except Exception:  # noqa: BLE001
    pass

DEFAULT_ENGINE = remote_jobs.DEFAULT_ENGINE

TEMPLATE_ID = "H-000001"
TEMPLATE_GRADE = "SAMPLE-GRADE"

SOURCE_CAPTIONS = {"modal": "calibrated on Modal", "local": "calibrated locally"}


def _app():
    """The Streamlit app module, imported on first use (it imports this module at boot)."""
    import streamlit_app  # noqa: PLC0415 - avoids the import cycle
    return streamlit_app


def template_frame(mill):
    """One blank row in the shape the calibrator reads.

    ``Heat_ID`` identifies the plate or coil, ``Grade`` is optional but unlocks the per-grade
    table, the SPECS columns are the sidebar inputs the scorer needs, and one ``Measured_<key>``
    column per property the route models is where the lab result goes. The measured cells are
    left empty on purpose: a pre-filled number would calibrate the model against itself.
    """
    app = _app()
    row = {"Heat_ID": TEMPLATE_ID, "Grade": TEMPLATE_GRADE}
    for spec_row in app.SPECS[mill]:
        row[spec_row[1]] = float(spec_row[6])
    for target in app.TARGET_LABEL[mill]:
        row[f"Measured_{target}"] = np.nan
    return pd.DataFrame([row])


def template_bytes(mill):
    return template_frame(mill).to_csv(index=False).encode("utf-8")


def _source_caption(source, result):
    text = SOURCE_CAPTIONS.get(str(source or ""), "calibrated locally")
    stamp = str(result.get("timestamp") or "").replace("T", " ").replace("+00:00", " UTC")
    if stamp:
        text += f" · measured {stamp}"
    text += f" · {int(result.get('n_rows') or 0):,} rows scored"
    if result.get("engine"):
        text += f" · {result['engine']}"
    return text


def _calibration_table(mill, calibration):
    """One tidy row per property: the median residual and the spread around it."""
    labels = _app().TARGET_LABEL[mill]
    rows = []
    for target, summary in (calibration or {}).items():
        label, unit = labels.get(target, (str(target), ""))
        rows.append({"Property": label, "Key": str(target), "Unit": unit,
                     "Bias (median residual)": round(float(summary["bias"]), 2),
                     "p10": round(float(summary["p10"]), 2),
                     "p90": round(float(summary["p90"]), 2),
                     "n": int(summary["n"])})
    if not rows:
        return pd.DataFrame()
    table = pd.DataFrame(rows)
    return table.reindex(table["Bias (median residual)"].abs().sort_values(ascending=False).index)


def _grade_table(mill, by_grade):
    """The same summary split by grade, when the uploaded sheet carried a Grade column."""
    labels = _app().TARGET_LABEL[mill]
    rows = []
    for grade, block in (by_grade or {}).items():
        for target, summary in block.items():
            label, unit = labels.get(target, (str(target), ""))
            rows.append({"Grade": str(grade), "Property": label, "Unit": unit,
                         "Bias (median residual)": round(float(summary["bias"]), 2),
                         "p10": round(float(summary["p10"]), 2),
                         "p90": round(float(summary["p90"]), 2),
                         "n": int(summary["n"])})
    return pd.DataFrame(rows)


def view_shadow(mill, raw, engine=DEFAULT_ENGINE):
    """The Digital shadow tab: calibrate model bias against measured results.

    Every expensive step sits behind a button ("Update shadow", "Load last calibration") and
    its outcome is kept in ``st.session_state[f"shadow_{mill}"]``, so a rerun replays the
    stored calibration and never rescoring the sheet on its own.
    """
    _app().section("digital shadow")
    st.markdown('<div class="note">Upload schedules whose mechanical results have already been '
                'measured. The app scores them, takes measured minus predicted for every '
                'property, and reports the median bias with its p10/p90 spread, overall and per '
                'grade. This calibrates your reading of the model against the works: no model '
                'file is retrained, replaced or altered.</div>', unsafe_allow_html=True)

    state_key = f"shadow_{mill}"
    left, mid, right = st.columns([1.4, 1, 1], vertical_alignment="bottom")
    left.download_button("Download the measured-results template (CSV)",
                         data=template_bytes(mill),
                         file_name=f"shadow_template_{mill}.csv", mime="text/csv",
                         key=f"shadow_tpl_{mill}")
    uploaded = left.file_uploader("Measured results (CSV)", type=["csv"],
                                  key=f"shadow_up_{mill}",
                                  help="Heat_ID, the schedule columns in the sidebar and one "
                                       "Measured_* column per property. A Grade column "
                                       "unlocks the per-grade table.")
    update = mid.button("Update shadow", type="primary", key=f"shadow_go_{mill}",
                        help="Scores the uploaded sheet and summarises the residuals. Uses "
                             "Modal when it is configured.")
    load = right.button("Load last calibration", key=f"shadow_load_{mill}",
                        help="Reads the calibration stored on the Modal volume for this route.")

    if update:
        if uploaded is None:
            st.warning("Attach the measured-results CSV first, then press Update shadow.")
        else:
            with st.spinner("Calibrating against measured results..."):
                result, source = remote_jobs.update_shadow(mill, uploaded.getvalue(),
                                                           engine=engine)
            if not result:
                st.error(f"Calibration failed: {remote_jobs.last_error() or 'no calibration produced'}"
                         ". Check that the sheet carries the schedule columns and at least one "
                         "Measured_* column filling a property this route scores.")
            else:
                st.session_state[state_key] = {"result": result, "source": source,
                                               "engine": engine}
    elif load:
        with st.spinner("Loading the last calibration..."):
            result, source = remote_jobs.load_shadow(mill)
        if not result:
            st.warning(f"No stored calibration for {mill} yet"
                       + (f" ({remote_jobs.last_error()})" if remote_jobs.last_error() else "")
                       + ". Press Update shadow to create one.")
        else:
            st.session_state[state_key] = {"result": result, "source": source,
                                           "engine": engine}

    stored = st.session_state.get(state_key)
    if not stored:
        st.caption("Upload a sheet of measured results and press Update shadow, or load the "
                   "last calibration, to see this route's bias.")
        return
    if not update and not load:
        st.caption("Shown from your last calibration. Press Update shadow to refit it against "
                   "a new sheet.")
    _render_calibration(mill, raw, stored, engine)


def _render_calibration(mill, raw, stored, engine):
    """Render one stored calibration: source, per-target bias, per-grade bias, optional preview."""
    result = stored.get("result") or {}
    calibration = result.get("calibration") or {}
    by_grade = result.get("by_grade") or {}
    if not calibration:
        st.error("This calibration carries no property, so there is nothing to show. The "
                 "uploaded sheet's Measured_* columns must name properties this route scores: "
                 + ", ".join(_app().TARGET_LABEL[mill]) + ".")
        return

    st.caption(_source_caption(stored.get("source"), result))

    section = _app().section
    section("bias per property")
    table = _calibration_table(mill, calibration)
    st.dataframe(table.drop(columns=["Key"]), hide_index=True, use_container_width=True)
    worst = table.iloc[0]
    st.caption(f"Bias is measured minus predicted, so a positive figure means the model reads "
               f"low and the measured population sits above it. Largest offset: {worst['Property']}"
               f" {worst['Bias (median residual)']:+.2f} {worst['Unit']} (p10 {worst['p10']:+.2f}, "
               f"p90 {worst['p90']:+.2f}, n={int(worst['n'])}).")

    fig = go.Figure(go.Bar(
        x=[f"{row['Property']}" for _, row in table.iterrows()],
        y=table["Bias (median residual)"],
        error_y=dict(type="data", symmetric=False,
                     array=(table["p90"] - table["Bias (median residual)"]).clip(lower=0),
                     arrayminus=(table["Bias (median residual)"] - table["p10"]).clip(lower=0),
                     color="#6b7280", thickness=1, width=4),
        marker_color="#1746c8",
        hovertemplate="%{x}<br>median residual %{y:+.2f}<extra></extra>",
    ))
    fig.add_hline(y=0, line_color="#0b0d10", line_width=1)
    fig.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor="white",
                      paper_bgcolor="white", showlegend=False,
                      yaxis=dict(title="measured - predicted", showline=False,
                                 gridcolor="#f1f2f4", zeroline=False),
                      xaxis=dict(showline=True, linecolor="#e5e7eb", tickfont=dict(size=10)))
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    st.caption("Bars are the median residual; whiskers span p10 to p90, so the bar is the "
               "offset and the whisker is how consistently that offset holds.")

    if by_grade:
        section("bias per grade")
        st.dataframe(_grade_table(mill, by_grade), hide_index=True, use_container_width=True)
        st.caption("One row per grade and property. Compare a grade's median against the global "
                   "figure above: a grade that sits well off it is where a separate grade "
                   "correction would earn its keep.")
    else:
        st.caption("No Grade column in the uploaded sheet, so only the global calibration is "
                   "available. Add a Grade column to the template for the per-grade table.")

    section("preview the correction")
    preview = st.checkbox("Preview corrected predictions for this session", value=False,
                          key=f"shadow_preview_{mill}",
                          help="Display-only: adds the stored median residual to the current "
                               "sidebar schedule. No model and no other tab changes.")
    if not preview:
        st.caption("Leave this off to keep reading the models as they are. The correction is a "
                   "display aid for this session only.")
        return

    app = _app()
    try:
        scored = app.score_frame(mill, pd.DataFrame([dict(raw)]), engine=engine)
    except Exception as exc:  # noqa: BLE001 - a preview must not kill the tab
        st.error(f"Could not score the current schedule: {type(exc).__name__}: {exc}")
        return
    labels = app.TARGET_LABEL[mill]
    rows = []
    for target, (label, unit) in labels.items():
        if target not in scored.columns:
            continue
        predicted = float(pd.to_numeric(scored[target], errors="coerce").iloc[0])
        summary = calibration.get(target)
        bias = float(summary["bias"]) if summary else 0.0
        rows.append({"Property": label, "Unit": unit,
                     "Model": round(predicted, 2) if np.isfinite(predicted) else None,
                     "Bias": round(bias, 2) if summary else None,
                     "Corrected (display only)": (round(predicted + bias, 2)
                                                  if np.isfinite(predicted) else None),
                     "n": int(summary["n"]) if summary else 0})
    if not rows:
        st.caption("This engine scored none of the route's properties, so there is nothing to "
                   "correct.")
        return
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    st.caption("Display-only correction: the model column is what the app reports everywhere "
               "else; the last column adds the median residual measured on your uploaded sheet "
               "to this one schedule. No model file and no other tab is touched.")
