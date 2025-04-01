# app.py
import streamlit as st
import pandas as pd
# Use the CSV/Parquet loader
from utils.data_loader import load_prediction_data
# Import other necessary utilities if needed later
# Import SHAP loading and calculation
from utils.shap_helpers import (load_shap_data
                                , calculate_global_shap_importance
                                , aggregate_shap_importance)
# from utils.data_loader import extract_evaluation_data, reconstruct_feature_dataframe
# Import your color definitions
from utils.theme import (
    PRIMARY_COLOR, BACKGROUND_COLOR, SECONDARY_BACKGROUND_COLOR,
    TEXT_COLOR, PLOT_COLOR_SEQUENCE, GRID_COLOR
)
import plotly.io as pio
import plotly.graph_objects as go

# --- Create Custom Plotly Theme/Template ---
custom_template = go.layout.Template()

# Set color sequence for traces (lines, bars, markers)
custom_template.layout.colorway = PLOT_COLOR_SEQUENCE

# Set layout colors
custom_template.layout.font = dict(color=TEXT_COLOR, family="Segoe UI") # Match font
custom_template.layout.paper_bgcolor = BACKGROUND_COLOR # Overall chart background
custom_template.layout.plot_bgcolor = BACKGROUND_COLOR # Plotting area background (can also use SECONDARY_BACKGROUND_COLOR for contrast)

# Axis colors
custom_template.layout.xaxis = dict(gridcolor=GRID_COLOR, linecolor=GRID_COLOR, zerolinecolor=GRID_COLOR)
custom_template.layout.yaxis = dict(gridcolor=GRID_COLOR, linecolor=GRID_COLOR, zerolinecolor=GRID_COLOR)
# Apply to colorbars, ternary, geo, mapbox, polar, scene axes too if you use them

# Legend colors (can be subtle)
custom_template.layout.legend = dict(bgcolor='rgba(255,255,255,0.7)', bordercolor=GRID_COLOR) # Semi-transparent white

# Title color
custom_template.layout.title = dict(font=dict(color=TEXT_COLOR))

# Register the template
template_name = "streamlit_theme"
pio.templates[template_name] = custom_template

# Set it as the default (combine with a base theme like 'plotly_white' for other defaults)
# pio.templates.default = "plotly_white+" + template_name
# OR just use the custom one if it defines enough
pio.templates.default = template_name

print(f"Plotly default template set to: {pio.templates.default}") # For confirmation


# set page title
st.set_page_config(page_title="Model Monitor", layout="wide")
st.title("Binary Classification Model Monitoring Dashboard")

st.markdown("""
This dashboard helps with monitoring the performance, fairness, and stability
of the binary classification models.

**Upload your prediction data (holdout/test) using the sidebar, or use the sample data.**
""")

# --- Initialize session state keys ---
if 'pred_df' not in st.session_state: st.session_state['pred_df'] = None
if 'ref_df' not in st.session_state: st.session_state['ref_df'] = None
if 'drift_df' not in st.session_state: st.session_state['drift_df'] = None
if 'drift_importance_df' not in st.session_state: st.session_state['drift_importance_df'] = None
if 'active_dataset_key' not in st.session_state: st.session_state['active_dataset_key'] = 'pred'
# --- NEW SHAP Session State Keys ---
if 'shap_data_dict' not in st.session_state: st.session_state['shap_data_dict'] = None # To store dict from load_shap_data
if 'shap_importance_df_raw' not in st.session_state: st.session_state['shap_importance_df_raw'] = None # Raw importance (post-processor)
if 'shap_importance_df_agg' not in st.session_state: st.session_state['shap_importance_df_agg'] = None # Aggregated importance (original features)
# Store file names to prevent reloading same file
if 'pred_file_name' not in st.session_state: st.session_state['pred_file_name'] = None
if 'ref_file_name' not in st.session_state: st.session_state['ref_file_name'] = None
if 'shap_values_file_name' not in st.session_state: st.session_state['shap_values_file_name'] = None
if 'shap_features_file_name' not in st.session_state: st.session_state['shap_features_file_name'] = None
if 'shap_base_value_file_name' not in st.session_state: st.session_state['shap_base_value_file_name'] = None
if 'shap_processed_data_file_name' not in st.session_state: st.session_state['shap_processed_data_file_name'] = None

# --- Sidebar for Data Upload ---
with st.sidebar:
    st.image("allyy_2025_transparent_svg.svg", width=200)
st.sidebar.header("Load Datasets")

# --- Prediction Data Uploader ---
uploaded_pred_file = st.sidebar.file_uploader(
    "1. Upload Prediction/Holdout Data*",
    type=["csv", "parquet"],
    help="Required. Features, true labels ('y_true'), prediction scores ('y_pred_prob').",
    key="pred_uploader"
)
if uploaded_pred_file is not None:
    if st.session_state.get('pred_file_name') != uploaded_pred_file.name:
        pred_df_loaded = load_prediction_data(uploaded_file=uploaded_pred_file)
        if pred_df_loaded is not None:
            st.session_state['pred_df'] = pred_df_loaded
            st.session_state['pred_file_name'] = uploaded_pred_file.name
            st.session_state['drift_df'] = None # Reset drift on data change
            st.session_state['drift_importance_df'] = None
            st.session_state['shap_importance_df_agg'] = None # Reset aggregated SHAP on data change
            if st.session_state['ref_df'] is None: st.session_state['active_dataset_key'] = 'pred'
            st.sidebar.success("Prediction data loaded!")
        else:
            st.sidebar.error("Failed to load prediction data.")
            st.session_state['pred_df'] = None

# --- Reference Data Uploader ---
uploaded_ref_file = st.sidebar.file_uploader(
    "2. Upload Reference/Training Data (Optional)",
    type=["csv", "parquet"],
    help="Optional. For drift analysis and comparison.",
    key="ref_uploader"
)
if uploaded_ref_file is not None:
    if st.session_state.get('ref_file_name') != uploaded_ref_file.name:
        ref_df_loaded = load_prediction_data(uploaded_file=uploaded_ref_file)
        if ref_df_loaded is not None:
            st.session_state['ref_df'] = ref_df_loaded
            st.session_state['ref_file_name'] = uploaded_ref_file.name
            st.session_state['drift_df'] = None # Reset drift on data change
            st.session_state['drift_importance_df'] = None
            st.sidebar.success("Reference data loaded!")
        else:
            st.sidebar.error("Failed to load reference data.")
            st.session_state['ref_df'] = None

# --- >>> NEW: SHAP Data Uploaders <<< ---
st.sidebar.markdown("---")
st.sidebar.header("Load SHAP Data (Optional)")
st.sidebar.caption("For Explainability page and enhanced Drift analysis.")

uploaded_shap_values_file = st.sidebar.file_uploader(
    "3. SHAP Values (Positive Class)",
    type=["npy", "csv"], help=".npy array or .csv (samples x features)", key="shap_values_uploader"
)
uploaded_shap_features_file = st.sidebar.file_uploader(
    "4. SHAP Feature Names",
    type=["json", "txt"], help=".json list or .txt (one per line)", key="shap_features_uploader"
)
uploaded_shap_base_value_file = st.sidebar.file_uploader(
    "5. SHAP Base Value",
    type=["json", "txt"], help=".json {'base_value': x} or .txt (single number)", key="shap_base_uploader"
)
uploaded_shap_processed_data_file = st.sidebar.file_uploader(
    "6. Data for SHAP Display*", # Changed label
    type=["parquet", "csv"],
    help="*Recommended: Data corresponding to SHAP values, ideally inverse-transformed where possible (samples x features). Required for beeswarm/dependence plots.", # Updated help text
    key="shap_data_uploader"
)

# Load SHAP data if all required files are uploaded and haven't been loaded before
required_shap_files = [uploaded_shap_values_file, uploaded_shap_features_file, uploaded_shap_base_value_file]
if all(required_shap_files):
    # Check if any of the SHAP files are new
    new_shap_file_uploaded = False
    if st.session_state.get('shap_values_file_name') != uploaded_shap_values_file.name:
        new_shap_file_uploaded = True
        st.session_state['shap_values_file_name'] = uploaded_shap_values_file.name
    if st.session_state.get('shap_features_file_name') != uploaded_shap_features_file.name:
        new_shap_file_uploaded = True
        st.session_state['shap_features_file_name'] = uploaded_shap_features_file.name
    if st.session_state.get('shap_base_value_file_name') != uploaded_shap_base_value_file.name:
        new_shap_file_uploaded = True
        st.session_state['shap_base_value_file_name'] = uploaded_shap_base_value_file.name
    # Check optional file change
    if uploaded_shap_processed_data_file and \
       st.session_state.get('shap_processed_data_file_name') != uploaded_shap_processed_data_file.name:
        new_shap_file_uploaded = True
        st.session_state['shap_processed_data_file_name'] = uploaded_shap_processed_data_file.name
    elif not uploaded_shap_processed_data_file and st.session_state.get('shap_processed_data_file_name') is not None:
         # Handle case where optional file is removed
         new_shap_file_uploaded = True
         st.session_state['shap_processed_data_file_name'] = None


    # If any file changed or SHAP data isn't loaded yet, attempt loading
    if new_shap_file_uploaded or st.session_state.get('shap_data_dict') is None:
        st.sidebar.info("Attempting to load SHAP data...")
        shap_data_dict_loaded = load_shap_data(
            uploaded_shap_values_file,
            uploaded_shap_features_file,
            uploaded_shap_base_value_file,
            uploaded_shap_processed_data_file # Pass optional file
        )
        if shap_data_dict_loaded:
            st.session_state['shap_data_dict'] = shap_data_dict_loaded
            st.sidebar.success("SHAP data loaded!")
            # --- Calculate Importance on Load ---
            with st.spinner("Calculating SHAP importance..."):
                 st.session_state['shap_importance_df_raw'] = calculate_global_shap_importance(
                     st.session_state['shap_data_dict']['shap_values'],
                     st.session_state['shap_data_dict']['feature_names']
                 )
                 # Try to calculate aggregated importance if prediction data is loaded
                 if st.session_state['pred_df'] is not None:
                     try:
                         # Exclude non-feature columns from original list
                         cols_to_exclude_agg = ['y_true', 'y_pred_prob', 'ContactId', 'EntityId', 'ResponseTimestamp', 'rank', 'bin', 'group', '_merge']
                         original_cols = [c for c in st.session_state['pred_df'].columns if c not in cols_to_exclude_agg]
                         st.session_state['shap_importance_df_agg'] = aggregate_shap_importance(
                             st.session_state['shap_importance_df_raw'],
                             original_cols
                         )
                     except Exception as agg_e:
                          st.warning(f"Could not calculate aggregated SHAP importance: {agg_e}")
                          st.session_state['shap_importance_df_agg'] = None
                 else:
                     st.session_state['shap_importance_df_agg'] = None

            st.session_state['drift_importance_df'] = None # Reset merged drift df
        else:
            st.sidebar.error("Failed to load SHAP data.")
            st.session_state['shap_data_dict'] = None
            st.session_state['shap_importance_df_raw'] = None
            st.session_state['shap_importance_df_agg'] = None

elif any(required_shap_files):
    st.sidebar.warning("Please upload all required SHAP files (Values, Features, Base Value).")


# --- >>> NEW: Dataset Selector for Analysis <<< ---
st.sidebar.markdown("---")
st.sidebar.header("Select Active Dataset")

# Options depend on which datasets are loaded
options = {}
if st.session_state['pred_df'] is not None:
    options['Prediction/Holdout'] = 'pred'
if st.session_state['ref_df'] is not None:
    options['Reference/Training'] = 'ref'

# Determine current selection based on state, default to 'pred' if available
current_selection_key = st.session_state.get('active_dataset_key', 'pred')
# Ensure the current selection is actually available, otherwise default
if current_selection_key == 'ref' and 'Reference/Training' not in options:
    current_selection_key = 'pred'
# Find the index for the radio button
current_index = list(options.values()).index(current_selection_key) if current_selection_key in options.values() else 0

if len(options) > 0:
    selected_label = st.sidebar.radio(
        "Analyze Performance On:",
        options=list(options.keys()), # Show user-friendly names
        index=current_index,
        key='dataset_selector_radio'
    )
    # Update session state if selection changed
    new_active_key = options[selected_label]
    if new_active_key != st.session_state['active_dataset_key']:
        st.session_state['active_dataset_key'] = new_active_key
        # No rerun needed here, pages will read the updated state on their next run
else:
    st.sidebar.info("Upload prediction data to enable analysis.")


# --- Main Panel Status Display ---
st.header("Current Status")
data_summary = []
active_key_display = st.session_state.get('active_dataset_key', 'N/A')
active_label = "Prediction/Holdout" if active_key_display == 'pred' else "Reference/Training"

# Prediction Data Status
if st.session_state['pred_df'] is not None:
    status = f"✅ Prediction Data ('pred_df')"
    if active_key_display == 'pred': status += " **(Active for Analysis)**"
    data_summary.append(f"{status}: **{st.session_state['pred_df'].shape}** (Rows, Cols)")
else:
    data_summary.append("⚠️ Prediction Data: **Not Loaded** (Required)")

# Reference Data Status
if st.session_state['ref_df'] is not None:
    status = f"✅ Reference Data ('ref_df')"
    if active_key_display == 'ref': status += " **(Active for Analysis)**"
    data_summary.append(f"{status}: **{st.session_state['ref_df'].shape}** (Rows, Cols)")
else:
    data_summary.append("ℹ️ Reference Data: **Not Loaded** (Optional)")

# SHAP Data Status
if st.session_state['shap_data_dict'] is not None:
    shap_status = "✅ SHAP Data: **Loaded**"
    if st.session_state['shap_importance_df_raw'] is not None:
         shap_status += f" ({len(st.session_state['shap_importance_df_raw'])} features)"
    if st.session_state['shap_data_dict'].get('processed_data') is None:
        shap_status += " *(Processed Data missing)*"
    data_summary.append(shap_status)
else:
    data_summary.append("ℹ️ SHAP Data: **Not Loaded** (Optional)")


st.markdown("\n".join([f"- {s}" for s in data_summary]))

if st.session_state['pred_df'] is None:
    st.warning("Please upload Prediction/Holdout data using the sidebar.")
else:
    st.info(f"Navigate using the sidebar. Performance pages will analyze the **{active_label.upper()}** dataset.")

st.markdown("---")