"""
Batch Schedule Scorer View.
Allows operations engineers to upload rolling schedule CSV/Excel files,
automatically executes domain feature extraction, and returns scored property reports.
"""

import streamlit as st
import pandas as pd
import numpy as np
import io
from model_loader import load_model_bundle, TARGET_CONFIG

try:
    from script.ml.features import compute_metallurgical_features as engineer_plate_features
except ImportError:
    engineer_plate_features = None

try:
    from ml.features import engineer_hrc_features
except ImportError:
    engineer_hrc_features = None


def render_batch_view():
    st.markdown("### Batch Rolling Schedule Prediction")
    st.markdown(
        "Upload a planned production schedule (Excel `.xlsx` or `.csv`) containing chemistry, dimensions, "
        "and rolling temperatures. The system automatically computes all metallurgical features, evaluates the "
        "trained ensembles, and returns an enriched dataset ready for production release."
    )

    col1, col2 = st.columns([1, 2])
    with col1:
        mill_type = st.radio("Mill Route", ["Plate", "HRC"], horizontal=True, key="batch_mill")
    with col2:
        uploaded_file = st.file_uploader("Upload Rolling Schedule File", type=["csv", "xlsx"])

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv"):
                df_raw = pd.read_csv(uploaded_file)
            else:
                df_raw = pd.read_excel(uploaded_file)
            st.success(f"Loaded schedule with {len(df_raw):,} records and {len(df_raw.columns)} columns.")
        except Exception as e:
            st.error(f"Error reading file: {e}")
            return

        with st.expander("Preview Uploaded Raw Data (First 5 Rows)"):
            st.dataframe(df_raw.head())

        if st.button("Run Batch Mechanical Prediction Pipeline", type="primary"):
            with st.spinner("Extracting physical features and evaluating ensemble models..."):
                try:
                    if mill_type == "Plate":
                        if engineer_plate_features is None:
                            st.error("Plate feature pipeline unavailable.")
                            return
                        X_features = engineer_plate_features(df_raw)
                    else:
                        if engineer_hrc_features is None:
                            st.error("HRC feature pipeline unavailable.")
                            return
                        X_features, _ = engineer_hrc_features(df_raw, is_train=False)
                except Exception as e:
                    st.error(f"Feature engineering failed: {e}")
                    return

                df_scored = df_raw.copy()
                targets = TARGET_CONFIG[mill_type]

                for tgt_key, meta in targets.items():
                    bundle = load_model_bundle(mill_type, tgt_key, use_tuned=True)
                    if bundle is None:
                        continue
                    model = bundle["model"]
                    cols = bundle.get("feature_names", X_features.columns.tolist())
                    
                    # Safe column alignment
                    X_aligned = pd.DataFrame(index=X_features.index)
                    for c in cols:
                        X_aligned[c] = X_features[c] if c in X_features.columns else 0.0

                    try:
                        preds = model.predict(X_aligned)
                        df_scored[f"PRED_{tgt_key}_{meta['unit']}"] = np.round(preds, 2)
                    except Exception as e:
                        df_scored[f"PRED_{tgt_key}_{meta['unit']}"] = np.nan

                st.markdown("---")
                st.subheader("Scored Production Schedule")
                st.dataframe(df_scored.head(20))

                # Export buttons
                csv_buf = io.StringIO()
                df_scored.to_csv(csv_buf, index=False)
                st.download_button(
                    label="📥 Download Scored Schedule (CSV)",
                    data=csv_buf.getvalue(),
                    file_name=f"{mill_type}_scored_schedule.csv",
                    mime="text/csv"
                )
