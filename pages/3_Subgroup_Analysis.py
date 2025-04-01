import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
# No matplotlib needed if using Plotly versions
# import matplotlib.pyplot as plt
import warnings

# Import necessary functions from your modules/utils
from utils.calculations import calculate_subgroup_means
from utils.plotting import (
    plot_group_summary_stats_plotly,
    plot_rank_metrics_by_group_interactive,
    plot_subgroup_means_interactive
)
from modules.prediction_holdout_analysis import rank_performance_by_group

# --- Page Config ---
st.set_page_config(layout="wide")
st.header("Subgroup Performance Analysis")
st.markdown("""
Analyze model performance broken down by subgroups based on a selected feature.
Select the dataset (Prediction/Reference) and the feature using the sidebar.
You can optionally filter the feature list to show only the most important features based on SHAP values (if loaded).
""")

# --- Determine Active Dataset ---
active_dataset_key = st.session_state.get('active_dataset_key', 'pred')
if active_dataset_key not in ['pred', 'ref']: active_dataset_key = 'pred'
active_df_key = f"{active_dataset_key}_df"
active_df_label = "Prediction/Holdout" if active_dataset_key == 'pred' else "Reference/Training"

# --- Data Check ---
if active_df_key not in st.session_state or st.session_state[active_df_key] is None:
    st.warning(f"The selected dataset ('{active_df_label}') is not loaded. Please load it on the main '🚀 Home' page.")
    st.stop()

active_df = st.session_state[active_df_key]
st.info(f"Analyzing subgroups for **{active_df_label}** dataset.")

# --- Configuration ---
Y_TRUE_COL = 'y_true'
Y_PRED_PROB_COL = 'y_pred_prob'
required_cols_perf = [Y_TRUE_COL, Y_PRED_PROB_COL]

# Check for required columns for performance analysis
missing_perf_cols = [col for col in required_cols_perf if col not in active_df.columns]
if missing_perf_cols:
    st.error(f"Error: Required columns missing from the active {active_df_label} dataset for performance analysis: {missing_perf_cols}.")
    st.stop()


# --- Feature Selection ---
st.sidebar.header("Subgroup Analysis Options")

# --- Get Base Feature List ---
cols_to_exclude = [Y_TRUE_COL, Y_PRED_PROB_COL, 'ContactId', 'EntityId', 'ResponseTimestamp', 'rank', 'bin', 'group', '_merge']
all_available_features = sorted([col for col in active_df.columns if col not in cols_to_exclude])

if not all_available_features:
     st.error(f"No suitable features found in the {active_df_label} dataset for subgroup analysis.")
     st.stop()

# --- Check for SHAP Importance ---
shap_importance_agg_df = st.session_state.get('shap_importance_df_agg')
shap_available = shap_importance_agg_df is not None and not shap_importance_agg_df.empty

# --- Feature Selection Mode ---
feature_selection_mode = st.sidebar.radio(
    "Feature List Filter:",
    options=["All Features", "Top N Important (SHAP)"],
    index=0, # Default to All
    key=f"subgroup_feature_mode_{active_dataset_key}",
    horizontal=True,
    help="Filter features based on SHAP importance (requires SHAP data loaded & aggregated)." if shap_available else "Load SHAP data on Home page to enable Top N filtering.",
    disabled=not shap_available # Disable Top N if SHAP agg unavailable
)

features_to_display = []
top_n_subgroup = 0 # Initialize

if feature_selection_mode == "Top N Important (SHAP)" and shap_available:
    # Get top N features from aggregated SHAP importance
    top_n_subgroup = st.sidebar.slider(
        "Number of Top Features (SHAP)",
        min_value=3,
        max_value=min(50, len(shap_importance_agg_df)),
        value=min(10, len(shap_importance_agg_df)),
        step=1,
        key=f"subgroup_top_n_{active_dataset_key}",
        help="Select how many top features (by aggregated SHAP importance) to show."
    )
    # Get the names of the top N original features
    top_n_shap_features = shap_importance_agg_df.head(top_n_subgroup)['original_feature'].tolist()
    # Filter this list to only include features ACTUALLY present in the active dataframe
    features_to_display = [f for f in top_n_shap_features if f in all_available_features]
    if not features_to_display:
        st.sidebar.warning("None of the top SHAP features found in the current dataset's columns. Showing all features instead.")
        features_to_display = all_available_features
    else:
         st.sidebar.caption(f"Showing {len(features_to_display)} most important features found in this dataset.")

else: # Default to "All Features" or if SHAP unavailable
    features_to_display = all_available_features
    if feature_selection_mode == "Top N Important (SHAP)" and not shap_available:
         st.sidebar.warning("SHAP importance data not available. Showing all features.", icon="⚠️")

# --- Feature Selection Dropdown ---
selected_feature = st.sidebar.selectbox(
    "Select Feature for Subgroup Analysis:",
    options=features_to_display, # Use the potentially filtered list
    index=0,
    key=f"subgroup_feature_select_{active_dataset_key}" # Key per dataset
)

# --- Binning Controls ---
st.sidebar.subheader(f"Binning for '{selected_feature}'")
# Determine if selected feature is numeric or categorical in the active dataset
is_numeric = pd.api.types.is_numeric_dtype(active_df[selected_feature])

numeric_q = None
cat_top_n = None

if is_numeric:
    numeric_q = st.sidebar.slider(
        f"Num Quantile Bins",
        min_value=2, max_value=20, value=5, step=1,
        key=f"num_q_{selected_feature}_{active_dataset_key}" # Unique key per feature AND dataset
    )
    st.sidebar.caption("Numeric features split into quantiles.")
else: # Categorical/Object
    unique_vals = active_df[selected_feature].nunique(dropna=False)
    max_n = min(20, unique_vals)
    cat_top_n = st.sidebar.slider(
        f"Num Top Categories",
        min_value=2, max_value=max_n, value=min(5, max_n), step=1,
        key=f"cat_n_{selected_feature}_{active_dataset_key}" # Unique key
    )
    st.sidebar.caption(f"Top {cat_top_n} shown, others grouped as 'Other'.")

st.sidebar.markdown("---")

# --- Main Panel - Display Analysis for Selected Feature ---
st.subheader(f"Analysis by: {selected_feature}")
# Add info about selection mode if Top N was used
if feature_selection_mode == "Top N Important (SHAP)" and shap_available and selected_feature in features_to_display:
     feature_rank = list(shap_importance_agg_df['original_feature']).index(selected_feature) + 1
     st.caption(f"(Ranked #{feature_rank} by aggregated SHAP importance)")

# --- Calculate Rank Metrics per Group ---
# Use a try-except block as calculation might fail (e.g., bad bins)
rank_metrics_grouped = pd.DataFrame() # Initialize empty
try:
    with warnings.catch_warnings(): # Suppress warnings during calculation if needed
        warnings.simplefilter("ignore")
        rank_metrics_grouped = rank_performance_by_group(
            df=active_df, # Use the selected active dataset
            group_feature=selected_feature,
            y_true_col=Y_TRUE_COL,
            y_pred_col=Y_PRED_PROB_COL,
            num_rank_bins=10, # Deciles within each group
            numeric_q=numeric_q, # Pass binning parameter
            cat_top_n=cat_top_n    # Pass binning parameter
        )
except Exception as e:
     st.error(f"Error calculating rank metrics for subgroups of '{selected_feature}': {e}")


# --- Calculate Subgroup Means Data ---
subgroup_means_df = pd.DataFrame() # Initialize empty
try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        subgroup_means_df = calculate_subgroup_means(
            df=active_df, # Use the selected active dataset
            feature=selected_feature,
            target_col=Y_TRUE_COL,
            pred_col=Y_PRED_PROB_COL,
            numeric_q=numeric_q,
            cat_top_n=cat_top_n
        )
except Exception as e:
    st.error(f"Error calculating subgroup means for '{selected_feature}': {e}")


# --- Display Plots if calculations succeeded ---

# 1. Group Summary Statistics
st.markdown("#### Subgroup Sizes and Baseline Rates")
if not rank_metrics_grouped.empty:
    try:
        fig_summary, _ = plot_group_summary_stats_plotly(
            rank_perf_df=rank_metrics_grouped, group_col='group'
        )
        st.plotly_chart(fig_summary, use_container_width=True)
    except Exception as e:
         st.warning(f"Could not plot group summary statistics: {e}")
else:
    st.info("Group summary statistics require rank metrics to be calculated successfully.")

st.markdown("---")

# 2. Rank Performance Metrics by Group (Side-by-Side)
st.markdown("#### Rank-Based Metrics by Subgroup")
if not rank_metrics_grouped.empty:
    metric_options = {
        'Precision': 'cumulative_precision_within_group',
        'Recall': 'cumulative_recall_within_group',
        'Lift': 'cumulative_lift_within_group'
    }
    col1, col2, col3 = st.columns(3)
    plot_funcs = [plot_rank_metrics_by_group_interactive] * 3 # Reuse the function

    for i, (metric_label, metric_col) in enumerate(metric_options.items()):
        with [col1, col2, col3][i]: # Place plot in correct column
            st.markdown(f"**{metric_label}**")
            try:
                fig = plot_funcs[i]( # Call the plotting function
                    rank_perf_df=rank_metrics_grouped,
                    metric_to_plot=metric_col,
                    group_col='group'
                )
                fig.update_layout(
                    title_text="", # Clear title
                    margin=dict(t=10, b=10, l=10, r=10),
                    # Only show legend on first plot
                    showlegend=(i == 0),
                    legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5) if i == 0 else None
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not plot {metric_label} by group: {e}")
else:
    st.info("Rank performance metrics require rank metrics calculation.")

st.markdown("---")

# 3. Mean Prediction vs Mean Target (Plotly Version)
st.markdown(f"#### Mean Prediction vs. Mean Target")
if not subgroup_means_df.empty:
    try:
        fig_means_plotly = plot_subgroup_means_interactive(
            analysis_df=subgroup_means_df,
            feature_name=selected_feature # Pass feature name for title
        )
        st.plotly_chart(fig_means_plotly, use_container_width=True)
    except Exception as e:
         st.warning(f"Could not plot Mean Prediction vs Target using Plotly: {e}")
else:
     st.info(f"Could not calculate subgroup means for '{selected_feature}'. Cannot display plot.")

st.markdown("---")

# --- (Add placeholders/calls for other analysis plots like residuals/calibration by group if implemented) ---
# st.markdown(f"#### Residual Analysis by '{selected_feature}' Subgroup")
# ... call plot_residuals_by_feature (needs Plotly version or use st.pyplot) ...

# st.markdown(f"#### Calibration by '{selected_feature}' Subgroup")
# ... call calibration_by_feature (needs Plotly version or use st.pyplot) ...