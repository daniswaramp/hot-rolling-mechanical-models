"""
Thermal Process Window Optimizer & Operating Contour Explorer.
Generates 2D response surfaces across Finish Rolling and Coiling/Post-Cooling temperatures
to identify safe manufacturing setpoints meeting customer mechanical constraints.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from model_loader import load_model_bundle, TARGET_CONFIG
from views.simulator import PRESETS

try:
    from script.ml.features import compute_metallurgical_features as engineer_plate_features
except ImportError:
    engineer_plate_features = None

try:
    from ml.features import engineer_hrc_features
except ImportError:
    engineer_hrc_features = None


def render_optimizer_view():
    st.markdown("### Safe Operating Process Window Explorer")
    st.markdown(
        "Map the 2D thermomechanical response surface across **Finish Rolling Temperature (FRT)** and "
        "**Cooling/Coiling Temperature (CT / T_post)**. Identify the safe manufacturing envelope "
        "that satisfies customer mechanical property guarantees."
    )

    col1, col2 = st.columns([1, 2])
    with col1:
        mill_type = st.radio("Rolling Mill", ["Plate", "HRC"], horizontal=True, key="opt_mill")
    with col2:
        preset_options = list(PRESETS[mill_type].keys())
        selected_grade = st.selectbox("Base Steel Grade Chemistry", preset_options, key="opt_grade")

    base_inputs = PRESETS[mill_type][selected_grade].copy()

    # Allow gauge tweak
    st.markdown("---")
    c_gauge, c_ys_min, c_ts_min = st.columns(3)
    with c_gauge:
        if mill_type == "Plate":
            target_t = st.number_input("Target Gauge (mm)", 6.0, 100.0, float(base_inputs["Plate_T_mm"]), 1.0)
            base_inputs["Plate_T_mm"] = target_t
        else:
            target_t = st.number_input("Target Gauge (mm)", 2.5, 16.0, float(base_inputs["Coil_T_mm"]), 0.5)
            base_inputs["Coil_T_mm"] = target_t

    with c_ys_min:
        ys_min = st.number_input("Customer Minimum Yield Strength (MPa)", 200.0, 700.0, 355.0, 10.0)
    with c_ts_min:
        ts_min = st.number_input("Customer Minimum Tensile Strength (MPa)", 300.0, 900.0, 490.0, 10.0)

    # Grid sweep configuration
    frt_range = np.linspace(740.0, 940.0, 15)
    ct_range = np.linspace(460.0, 680.0, 15)

    if st.button("Compute Process Window Contour Surface", type="primary"):
        with st.spinner("Generating 225-point thermomechanical response grid..."):
            grid_records = []
            for frt in frt_range:
                for ct in ct_range:
                    rec = base_inputs.copy()
                    if mill_type == "Plate":
                        rec["FM_Finish_Temp"] = frt
                        rec["ACC_Temp_Post_Laminar"] = min(ct, frt - 50.0)
                    else:
                        rec["Temp_FT_Avg_C"] = frt
                        rec["Temp_CT_Top_C"] = ct
                        rec["Temp_CT_Bot_C"] = ct
                    grid_records.append(rec)

            df_grid = pd.DataFrame(grid_records)
            if mill_type == "Plate":
                X_grid = engineer_plate_features(df_grid)
            else:
                X_grid, _ = engineer_hrc_features(df_grid, is_train=False)

            # Load YS and TS models
            ys_bundle = load_model_bundle(mill_type, "YS_BN", use_tuned=True)
            ts_bundle = load_model_bundle(mill_type, "TS_BN", use_tuned=True)

            if ys_bundle is None or ts_bundle is None:
                st.error("Required YS or TS model bundles could not be loaded.")
                return

            # Align features
            ys_cols = ys_bundle.get("feature_names", X_grid.columns.tolist())
            ts_cols = ts_bundle.get("feature_names", X_grid.columns.tolist())

            X_ys = pd.DataFrame({c: X_grid[c] if c in X_grid.columns else 0.0 for c in ys_cols}, index=X_grid.index)
            X_ts = pd.DataFrame({c: X_grid[c] if c in X_grid.columns else 0.0 for c in ts_cols}, index=X_grid.index)

            ys_preds = ys_bundle["model"].predict(X_ys)
            ts_preds = ts_bundle["model"].predict(X_ts)

            # Reshape into 2D grids (rows: CT, cols: FRT)
            Z_ys = ys_preds.reshape(len(frt_range), len(ct_range)).T
            Z_ts = ts_preds.reshape(len(frt_range), len(ct_range)).T

            # Plotly Contour Subplots
            col_plot1, col_plot2 = st.columns(2)

            with col_plot1:
                st.markdown("#### Yield Strength Contour (YS_BN, MPa)")
                fig_ys = go.Figure(data=go.Contour(
                    z=Z_ys, x=frt_range, y=ct_range,
                    colorscale="Blues", contours=dict(showlabels=True, labelfont=dict(size=11, color="white")),
                    colorbar=dict(title="YS (MPa)")
                ))
                # Add threshold line
                fig_ys.update_layout(
                    xaxis_title="Finish Rolling Temp FRT (°C)",
                    yaxis_title="Coiling / Water Stop Temp (°C)",
                    template="plotly_dark", height=450
                )
                st.plotly_chart(fig_ys, use_container_width=True)

            with col_plot2:
                st.markdown("#### Tensile Strength Contour (TS_BN, MPa)")
                fig_ts = go.Figure(data=go.Contour(
                    z=Z_ts, x=frt_range, y=ct_range,
                    colorscale="Inferno", contours=dict(showlabels=True, labelfont=dict(size=11, color="white")),
                    colorbar=dict(title="TS (MPa)")
                ))
                fig_ts.update_layout(
                    xaxis_title="Finish Rolling Temp FRT (°C)",
                    yaxis_title="Coiling / Water Stop Temp (°C)",
                    template="plotly_dark", height=450
                )
                st.plotly_chart(fig_ts, use_container_width=True)

            # Feasibility Mask
            feasible = (Z_ys >= ys_min) & (Z_ts >= ts_min)
            feasibility_pct = (np.sum(feasible) / feasible.size) * 100.0

            st.markdown("---")
            if feasibility_pct > 0:
                st.success(f"✓ **Process Window Viable**: {feasibility_pct:.1f}% of the investigated thermal parameter space safely exceeds customer requirements ($YS \ge {ys_min}$ MPa and $TS \ge {ts_min}$ MPa).")
            else:
                st.error(f"⚠️ **Infeasible Parameter Space**: None of the thermal setpoints hit the target combination ($YS \ge {ys_min}$ MPa, $TS \ge {ts_min}$ MPa). Consider alloy strengthening (increasing Mn, microalloying Nb/V) or decreasing final gauge.")
