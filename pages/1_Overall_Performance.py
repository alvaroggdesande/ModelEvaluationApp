import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import warnings

# Import calculation functions
from ModelEvaluationApp.utils.calculations import (
    get_metrics_at_threshold,
    get_calibration_data,
    calculate_rank_metrics,
    calculate_descriptive_stats
)
# Import plotting functions
from ModelEvaluationApp.utils.plotting import (
    plot_roc_curve_interactive,
    plot_pr_curve_interactive,
    plot_calibration_interactive,
    plot_rank_metrics_interactive,
    plot_lift_chart_interactive,
    plot_score_distribution_plotly, # Use the plotly version
    plot_distribution_comparison_interactive # For comparing pred vs ref scores
)
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.calibration import calibration_curve

# --- Page Config ---
st.set_page_config(layout="wide")
st.header("Overall Model Performance")
st.markdown("Evaluate aggregate performance metrics for the selected dataset.")

# --- Determine Active Dataset ---
# Default to 'pred' if key is somehow missing or invalid
active_dataset_key = st.session_state.get('active_dataset_key', 'pred')
if active_dataset_key not in ['pred', 'ref']:
    active_dataset_key = 'pred' # Fallback to default

active_df_key = f"{active_dataset_key}_df" # e.g., 'pred_df' or 'ref_df'
active_df_label = "Prediction/Holdout" if active_dataset_key == 'pred' else "Reference/Training"

# --- Data Check ---
if active_df_key not in st.session_state or st.session_state[active_df_key] is None:
    st.warning(f"The selected dataset ('{active_df_label}') is not loaded. Please load it on the main '🚀 Home' page.")
    st.stop()

active_df = st.session_state[active_df_key]
st.info(f"Displaying metrics for **{active_df_label}** dataset.")

# --- Configuration ---
# MAKE SURE THESE MATCH YOUR DATAFRAME COLUMNS
Y_TRUE_COL = 'y_true'
Y_PRED_PROB_COL = 'y_pred_prob'

# --- Data Preparation and Validation ---
required_cols = [Y_TRUE_COL, Y_PRED_PROB_COL]
missing_cols = [col for col in required_cols if col not in active_df.columns]

if missing_cols:
     st.error(f"Error: Required columns missing from the active {active_df_label} dataset: {missing_cols}. Cannot proceed.")
     st.stop()

# Create eval_df from the active dataset for calculations
# Suppress warnings during conversion if necessary
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    eval_df = active_df[[Y_TRUE_COL, Y_PRED_PROB_COL]].copy()
    eval_df[Y_TRUE_COL] = pd.to_numeric(eval_df[Y_TRUE_COL], errors='coerce')
    eval_df[Y_PRED_PROB_COL] = pd.to_numeric(eval_df[Y_PRED_PROB_COL], errors='coerce')
eval_df = eval_df.dropna() # Drop rows where conversion failed

if eval_df.empty:
     st.error(f"No valid data (numeric true labels and prediction scores) remaining in the active {active_df_label} dataset.")
     st.stop()

is_binary = len(eval_df[Y_TRUE_COL].unique()) >= 2


# --- Overall AUC Metrics ---
st.subheader("Overall AUC Metrics")
roc_auc = np.nan
pr_auc = np.nan
if is_binary:
     try:
         with warnings.catch_warnings(): # Suppress potential warnings from sklearn
             warnings.simplefilter("ignore")
             roc_auc = roc_auc_score(eval_df[Y_TRUE_COL], eval_df[Y_PRED_PROB_COL])
             pr_auc = average_precision_score(eval_df[Y_TRUE_COL], eval_df[Y_PRED_PROB_COL])
     except Exception as e:
          st.warning(f"Could not calculate AUCs: {e}")
else:
     st.warning("Only one class present in True labels. AUCs are undefined.")

col1, col2, col3 = st.columns(3)
col1.metric("ROC AUC", f"{roc_auc:.4f}" if not pd.isna(roc_auc) else "N/A")
col2.metric("PR AUC", f"{pr_auc:.4f}" if not pd.isna(pr_auc) else "N/A")
col3.metric("Data Points (used)", f"{len(eval_df):,}")


# --- Threshold Adjustment and Metrics ---
st.subheader("Metrics at Threshold")
threshold = st.slider(
    "Select Probability Threshold",
    min_value=0.0, max_value=1.0, value=0.5, step=0.01,
    key=f"thresh_{active_dataset_key}" # Unique key per dataset
)
metrics = get_metrics_at_threshold(eval_df[Y_TRUE_COL], eval_df[Y_PRED_PROB_COL], threshold)
mcol1, mcol2, mcol3 = st.columns(3)
mcol1.metric("Precision", f"{metrics['precision']:.4f}")
mcol2.metric("Recall", f"{metrics['recall']:.4f}")
mcol3.metric("F1-Score", f"{metrics['f1']:.4f}")
st.write(f"At threshold {threshold:.2f}: TP={metrics['tp']}, FP={metrics['fp']}, FN={metrics['fn']}, TN={metrics['tn']}")


# --- Performance Curves ---
st.subheader("Performance Curves")
if is_binary:
    col_plots1, col_plots2 = st.columns(2)
    with col_plots1:
        fig_roc = plot_roc_curve_interactive(eval_df[Y_TRUE_COL], eval_df[Y_PRED_PROB_COL], title=f"ROC Curve ({active_df_label})")
        st.plotly_chart(fig_roc, use_container_width=True)
    with col_plots2:
        fig_pr = plot_pr_curve_interactive(eval_df[Y_TRUE_COL], eval_df[Y_PRED_PROB_COL], title=f"PR Curve ({active_df_label})")
        st.plotly_chart(fig_pr, use_container_width=True)
else:
     st.warning("Cannot generate performance curves: Only one class present.")


# --- Calibration Plot ---
st.subheader("Calibration")
if is_binary:
    n_bins_cal = st.slider("Number of Bins", 5, 20, 10, 1, key=f"cal_bins_{active_dataset_key}")
    try:
        prob_pred, prob_true = get_calibration_data(eval_df[Y_TRUE_COL], eval_df[Y_PRED_PROB_COL], n_bins=n_bins_cal)
        fig_cal = plot_calibration_interactive(prob_pred, prob_true, title=f"Calibration Plot ({active_df_label})")
        st.plotly_chart(fig_cal, use_container_width=True)
    except ValueError as e:
         st.warning(f"Could not generate calibration plot: {e}")
    except Exception as e:
         st.error(f"An unexpected error occurred during calibration plotting: {e}")
else:
     st.warning("Cannot generate calibration plot: Only one class present.")


# --- Rank-Based Metrics ---
st.subheader("Rank-Based Performance")
if is_binary:
    n_bins_rank = st.slider("Number of Rank Bins (e.g., 10=Deciles)", 5, 20, 10, 1, key=f"rank_bins_{active_dataset_key}")
    try:
        rank_metrics_df = calculate_rank_metrics(
            df=eval_df, # Use the cleaned eval_df for the active dataset
            y_true_col=Y_TRUE_COL,
            y_pred_col=Y_PRED_PROB_COL,
            num_bins=n_bins_rank
        )
        if not rank_metrics_df.empty:
            with st.expander("Show Rank Metrics Table"):
                st.dataframe(rank_metrics_df)

            col_rank1, col_rank2 = st.columns(2)
            with col_rank1:
                fig_rank_pr = plot_rank_metrics_interactive(
                    rank_metrics_df, title=f"Precision & Recall ({active_df_label})"
                )
                st.plotly_chart(fig_rank_pr, use_container_width=True)
            with col_rank2:
                fig_lift = plot_lift_chart_interactive(
                    rank_metrics_df, title=f"Lift Chart ({active_df_label})"
                )
                st.plotly_chart(fig_lift, use_container_width=True)
        else:
            st.warning("Rank metrics calculation resulted in an empty DataFrame.")
    except Exception as e:
        st.error(f"An error occurred during rank metric calculation or plotting: {e}")
else:
    st.warning("Cannot calculate rank-based metrics: Only one class present.")


# --- Prediction Score Distribution ---
st.subheader("Prediction Score Distribution")
active_scores = eval_df[Y_PRED_PROB_COL] # Scores from the active dataset

# Check if the other dataset exists for comparison
comparison_df = None
comparison_label = None
comparison_key = None
if active_dataset_key == 'pred' and 'ref_df' in st.session_state and st.session_state['ref_df'] is not None:
    comparison_df = st.session_state['ref_df']
    comparison_label = "Reference/Training"
    comparison_key = 'ref'
elif active_dataset_key == 'ref' and 'pred_df' in st.session_state and st.session_state['pred_df'] is not None:
    comparison_df = st.session_state['pred_df']
    comparison_label = "Prediction/Holdout"
    comparison_key = 'pred'

comparison_scores = None
if comparison_df is not None and Y_PRED_PROB_COL in comparison_df.columns:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            comparison_scores = pd.to_numeric(comparison_df[Y_PRED_PROB_COL], errors='coerce').dropna()
            if comparison_scores.empty: comparison_scores = None # Treat empty as None
    except Exception:
        comparison_scores = None

# Decide whether to plot comparison or single
plot_comparison = comparison_scores is not None

if plot_comparison:
    st.write(f"Comparing **{active_df_label}** vs. **{comparison_label}** Scores")
    try:
        fig_dist = plot_distribution_comparison_interactive(
            active_scores,
            comparison_scores,
            name1=active_df_label,
            name2=comparison_label,
            feature_name="Prediction Score"
            # nbins = # Optionally add slider later
        )
        st.plotly_chart(fig_dist, use_container_width=True)
    except Exception as e:
         st.error(f"Could not plot score distribution comparison: {e}")

else:
    st.write(f"Displaying **{active_df_label}** Scores")
    try:
        fig_dist_single = plot_score_distribution_plotly(
            scores=active_scores,
            feature_name="Prediction Score",
            title=f"Prediction Score Distribution ({active_df_label})"
            # nbins = # Optionally add slider later
        )
        st.plotly_chart(fig_dist_single, use_container_width=True)
    except Exception as e:
         st.error(f"Could not plot score distribution: {e}")


# --- Descriptive Stats for Active Dataset ---
with st.expander("Show Descriptive Statistics for Active Scores"):
    stats_df = calculate_descriptive_stats(active_scores)
    if stats_df is not None:
        st.dataframe(stats_df)
    else:
        st.warning("Could not calculate descriptive statistics.")