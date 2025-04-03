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
st.sidebar.header("Load SHAP Importance (Pre-calculated)")
uploaded_shap_importance_raw_file = st.sidebar.file_uploader(
    "3. Upload Raw SHAP Importance*",
    type=["csv", "parquet"], help="Pre-calculated mean abs SHAP per processed feature.",
    key="shap_importance_raw_uploader"
)
uploaded_shap_importance_agg_file = st.sidebar.file_uploader(
    "4. Upload Aggregated SHAP Importance (Optional)",
    type=["csv", "parquet"], help="Pre-calculated SHAP importance aggregated to original features.",
    key="shap_importance_agg_uploader"
)

# --- Inside the loading logic ---
# Reset dependent states if files change
new_shap_importance_file = False
if uploaded_shap_importance_raw_file and \
   st.session_state.get('shap_importance_raw_file_name') != uploaded_shap_importance_raw_file.name:
    new_shap_importance_file = True
    st.session_state['shap_importance_raw_file_name'] = uploaded_shap_importance_raw_file.name

if uploaded_shap_importance_agg_file and \
   st.session_state.get('shap_importance_agg_file_name') != uploaded_shap_importance_agg_file.name:
    new_shap_importance_file = True
    st.session_state['shap_importance_agg_file_name'] = uploaded_shap_importance_agg_file.name
elif not uploaded_shap_importance_agg_file and st.session_state.get('shap_importance_agg_file_name') is not None:
    new_shap_importance_file = True
    st.session_state['shap_importance_agg_file_name'] = None


if new_shap_importance_file or st.session_state.get('shap_importance_df_raw') is None:
    st.session_state['shap_data_dict'] = None # Clear old dict if it existed
    st.session_state['shap_importance_df_raw'] = None
    st.session_state['shap_importance_df_agg'] = None
    st.session_state['drift_importance_df'] = None # Reset drift merge

    if uploaded_shap_importance_raw_file:
        st.sidebar.info("Loading pre-calculated SHAP importance...")
        try:
            if uploaded_shap_importance_raw_file.name.endswith('.csv'):
                raw_imp_df = pd.read_csv(uploaded_shap_importance_raw_file)
            else: # parquet
                raw_imp_df = pd.read_parquet(uploaded_shap_importance_raw_file)

            # Validate essential columns
            if 'feature' in raw_imp_df.columns and 'feature_importance' in raw_imp_df.columns:
                st.session_state['shap_importance_df_raw'] = raw_imp_df.sort_values('feature_importance', ascending=False).reset_index(drop=True)
                st.sidebar.success("Raw SHAP importance loaded!")

                # Try loading aggregated importance if provided
                if uploaded_shap_importance_agg_file:
                    try:
                        if uploaded_shap_importance_agg_file.name.endswith('.csv'):
                             agg_imp_df = pd.read_csv(uploaded_shap_importance_agg_file)
                        else: # parquet
                             agg_imp_df = pd.read_parquet(uploaded_shap_importance_agg_file)

                        if 'original_feature' in agg_imp_df.columns and 'aggregated_importance' in agg_imp_df.columns:
                             st.session_state['shap_importance_df_agg'] = agg_imp_df.sort_values('aggregated_importance', ascending=False).reset_index(drop=True)
                             st.sidebar.success("Aggregated SHAP importance loaded!")
                        else:
                             st.sidebar.warning("Aggregated importance file missing required columns ('original_feature', 'aggregated_importance').")
                             st.session_state['shap_importance_df_agg'] = None # Ensure it's None
                    except Exception as e_agg:
                        st.sidebar.error(f"Failed to load aggregated importance: {e_agg}")
                        st.session_state['shap_importance_df_agg'] = None
                else:
                    st.session_state['shap_importance_df_agg'] = None # Explicitly set to None if file not uploaded

            else:
                st.sidebar.error("Raw importance file missing required columns ('feature', 'feature_importance').")
                st.session_state['shap_importance_df_raw'] = None # Ensure it's None

        except Exception as e_raw:
            st.sidebar.error(f"Failed to load raw importance: {e_raw}")
            st.session_state['shap_importance_df_raw'] = None
            st.session_state['shap_importance_df_agg'] = None


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
if st.session_state.get('shap_importance_df_raw') is not None:
    shap_status = "✅ SHAP Importance: **Loaded (Pre-calculated)**"
    shap_status += f" ({len(st.session_state['shap_importance_df_raw'])} processed features)"
    if st.session_state.get('shap_importance_df_agg') is not None:
         shap_status += f" ({len(st.session_state['shap_importance_df_agg'])} aggregated features)"
    data_summary.append(shap_status)
    # Add note that interactive plots are disabled
    st.sidebar.info("Note: Interactive SHAP plots (Beeswarm, Dependence, Waterfall) are disabled when using pre-calculated importance.", icon="ℹ️")

elif st.session_state.get('shap_data_dict') is not None: # Existing check if full data was loaded
    shap_status = "✅ SHAP Data: **Loaded (Full)**"
    # ... rest of existing status ...
    data_summary.append(shap_status)
else:
    data_summary.append("ℹ️ SHAP Data/Importance: **Not Loaded** (Optional)")

st.markdown("\n".join([f"- {s}" for s in data_summary]))

if st.session_state['pred_df'] is None:
    st.warning("Please upload Prediction/Holdout data using the sidebar.")
else:
    st.info(f"Navigate using the sidebar. Performance pages will analyze the **{active_label.upper()}** dataset.")

st.markdown("---")