"""
Dual-Mill Steel Mechanical Property Modeling Platform
Main Streamlit Application Entrypoint.
Connects Heavy Plate Sequential Rolling & Steckel Mill HRC pipelines.
"""

import os
import sys
import streamlit as st

# Setup paths
APP_DIR = os.path.dirname(os.path.abspath(__file__))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

from views.simulator import render_simulator_view
from views.process_optimizer import render_optimizer_view
from views.batch_predictor import render_batch_view
from views.diagnostics import render_diagnostics_view

st.set_page_config(
    page_title="Dual-Mill Mechanical Property Modeling | Heavy Plate & Steckel HRC",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #38bdf8;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #94a3b8;
        margin-bottom: 1.5rem;
    }
    .stMetric {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 12px 16px;
    }
</style>
""", unsafe_allow_html=True)


def main():
    st.sidebar.image("https://img.icons8.com/fluency/96/steel-i-beam.png", width=64)
    st.sidebar.title("Dual-Mill ML Platform")
    st.sidebar.caption("Heavy Plate & Steckel Mill HRC")

    st.sidebar.markdown("---")
    menu = st.sidebar.radio(
        "Navigation",
        [
            "⚙️ Property Simulator",
            "🗺️ Process Window Explorer",
            "📁 Batch Schedule Scorer",
            "🔬 Diagnostics & Physics"
        ],
        index=0
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        """
        **System Status**:
        - **Heavy Plate Route**: Active (10 Targets)
        - **Steckel HRC Route**: Active (3 Targets)
        - **Compute Backend**: Modal Serverless Cloud
        - **Validation Protocol**: GroupSplit on Heat ID (Zero Leakage)
        """
    )
    st.sidebar.caption("© 2026 PT Gunung Raja Paksi / Industrial Analytics")

    # Header Banner
    st.markdown('<div class="main-header">Industrial Dual-Mill Mechanical Property Platform</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Physics-Grounded Machine Learning for Room-Temperature Tensile, Impact Toughness & Z-Test Properties</div>',
        unsafe_allow_html=True
    )

    if menu == "⚙️ Property Simulator":
        render_simulator_view()
    elif menu == "🗺️ Process Window Explorer":
        render_optimizer_view()
    elif menu == "📁 Batch Schedule Scorer":
        render_batch_view()
    elif menu == "🔬 Diagnostics & Physics":
        render_diagnostics_view()


if __name__ == "__main__":
    main()
