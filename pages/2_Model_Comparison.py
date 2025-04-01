# pages/2_Model_Comparison.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import warnings

# Import calculation functions (reusable)
from utils.calculations import (
    get_metrics_at_threshold,
    get_calibration_data,
    calculate_rank_metrics,
    calculate_descriptive_stats
)
# Import PLOTTING functions - including NEW comparison ones
from utils.plotting import (
    plot_roc_comparison_interactive,        # NEW
    plot_pr_comparison_interactive,         # NEW
    plot_calibration_comparison_interactive,# NEW
    plot_rank_metrics_comparison_interactive, # NEW
    plot_lift_chart_comparison_interactive, # NEW
    plot_distribution_comparison_interactive # Existing, suitable for comparison
)
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.calibration import calibration_curve

# --- Page Config ---
st.set_page_config(layout="wide")
st.header("Model Performance Comparison")
st.markdown("""
Compare model performance metrics between the **Reference (e.g., Training/Test)** dataset
and the **Current (Prediction/Holdout)** dataset side-by-side.

**Load both datasets on the 'Home' page sidebar to enable this comparison.**
""")

# --- Data Availability Check ---
if 'pred_df' not in st.session_state or st.session_state['pred_df'] is None or \
   'ref_df' not in st.session_state or st.session_state['ref_df'] is None:
    st.warning("❗ Both Prediction/Holdout and Reference/Training data must be loaded on the '🚀 Home' page to enable comparison.")
    st.stop()

pred_df_orig = st.session_state['pred_df']
ref_df_orig = st.session_state['ref_df']
st.success("✅ Both Prediction/Holdout and Reference/Training datasets loaded. Proceeding with comparison.")

# --- Configuration ---
Y_TRUE_COL = 'y_true'
Y_PRED_PROB_COL = 'y_pred_prob'
required_cols = [Y_TRUE_COL, Y_PRED_PROB_COL]

# --- Data Preparation and Validation ---
def prepare_eval_df(df, df_name):
    """Helper to prepare and validate a dataframe for evaluation."""
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        st.error(f"Error: Required columns missing from {df_name} dataset: {missing}. Cannot proceed with comparison.")
        return None, False # Indicate failure

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        eval_df = df[required_cols].copy()
        eval_df[Y_TRUE_COL] = pd.to_numeric(eval_df[Y_TRUE_COL], errors='coerce')
        eval_df[Y_PRED_PROB_COL] = pd.to_numeric(eval_df[Y_PRED_PROB_COL], errors='coerce')
    eval_df = eval_df.dropna()

    if eval_df.empty:
        st.error(f"No valid data (numeric true labels and prediction scores) remaining in the {df_name} dataset after cleaning.")
        return None, False # Indicate failure

    is_binary = len(eval_df[Y_TRUE_COL].unique()) >= 2
    if not is_binary:
        st.warning(f"Only one class present in True labels for {df_name}. Some comparison metrics (like AUCs) might be undefined or misleading.")

    return eval_df, is_binary

eval_pred_df, pred_is_binary = prepare_eval_df(pred_df_orig, "Prediction/Holdout")
eval_ref_df, ref_is_binary = prepare_eval_df(ref_df_orig, "Reference/Training")

# Stop if either preparation failed
if eval_pred_df is None or eval_ref_df is None:
    st.stop()

# Use a flag for easier checking if comparison makes sense for AUCs/Curves
can_compare_curves = pred_is_binary and ref_is_binary

# --- Overall AUC Metrics Comparison ---
st.subheader("Overall AUC Metrics")

roc_auc_pred, pr_auc_pred = np.nan, np.nan
roc_auc_ref, pr_auc_ref = np.nan, np.nan

if pred_is_binary:
    try:
        roc_auc_pred = roc_auc_score(eval_pred_df[Y_TRUE_COL], eval_pred_df[Y_PRED_PROB_COL])
        pr_auc_pred = average_precision_score(eval_pred_df[Y_TRUE_COL], eval_pred_df[Y_PRED_PROB_COL])
    except Exception as e: st.warning(f"Could not calculate Prediction AUCs: {e}")
if ref_is_binary:
     try:
         roc_auc_ref = roc_auc_score(eval_ref_df[Y_TRUE_COL], eval_ref_df[Y_PRED_PROB_COL])
         pr_auc_ref = average_precision_score(eval_ref_df[Y_TRUE_COL], eval_ref_df[Y_PRED_PROB_COL])
     except Exception as e: st.warning(f"Could not calculate Reference AUCs: {e}")

col1, col2 = st.columns(2)
with col1:
    st.metric("ROC AUC (Prediction)", f"{roc_auc_pred:.4f}" if not pd.isna(roc_auc_pred) else "N/A")
    st.metric("ROC AUC (Reference)", f"{roc_auc_ref:.4f}" if not pd.isna(roc_auc_ref) else "N/A",
              delta=f"{roc_auc_pred-roc_auc_ref:.4f}" if not pd.isna(roc_auc_pred) and not pd.isna(roc_auc_ref) else None)
with col2:
    st.metric("PR AUC (Prediction)", f"{pr_auc_pred:.4f}" if not pd.isna(pr_auc_pred) else "N/A")
    st.metric("PR AUC (Reference)", f"{pr_auc_ref:.4f}" if not pd.isna(pr_auc_ref) else "N/A",
              delta=f"{pr_auc_pred-pr_auc_ref:.4f}" if not pd.isna(pr_auc_pred) and not pd.isna(pr_auc_ref) else None)

col_size1, col_size2 = st.columns(2)
col_size1.metric("Data Points (Prediction)", f"{len(eval_pred_df):,}")
col_size2.metric("Data Points (Reference)", f"{len(eval_ref_df):,}")


# --- Threshold Adjustment and Metrics Comparison ---
st.subheader("Metrics at Threshold")
threshold = st.slider(
    "Select Probability Threshold for Comparison",
    min_value=0.0, max_value=1.0, value=0.5, step=0.01,
    key="thresh_comp" # Unique key for comparison page slider
)

metrics_pred = get_metrics_at_threshold(eval_pred_df[Y_TRUE_COL], eval_pred_df[Y_PRED_PROB_COL], threshold)
metrics_ref = get_metrics_at_threshold(eval_ref_df[Y_TRUE_COL], eval_ref_df[Y_PRED_PROB_COL], threshold)

mcol1, mcol2, mcol3 = st.columns(3)
with mcol1:
    st.metric("Precision (Pred)", f"{metrics_pred['precision']:.4f}")
    st.metric("Precision (Ref)", f"{metrics_ref['precision']:.4f}", delta=f"{metrics_pred['precision']-metrics_ref['precision']:.4f}")
with mcol2:
    st.metric("Recall (Pred)", f"{metrics_pred['recall']:.4f}")
    st.metric("Recall (Ref)", f"{metrics_ref['recall']:.4f}", delta=f"{metrics_pred['recall']-metrics_ref['recall']:.4f}")
with mcol3:
    st.metric("F1-Score (Pred)", f"{metrics_pred['f1']:.4f}")
    st.metric("F1-Score (Ref)", f"{metrics_ref['f1']:.4f}", delta=f"{metrics_pred['f1']-metrics_ref['f1']:.4f}")

exp_cm = st.expander("Show Confusion Matrices at Threshold")
with exp_cm:
    cm_col1, cm_col2 = st.columns(2)
    with cm_col1:
        st.write(f"**Prediction/Holdout** (Thresh: {threshold:.2f})")
        st.write(f"TP={metrics_pred['tp']}, FP={metrics_pred['fp']}, FN={metrics_pred['fn']}, TN={metrics_pred['tn']}")
    with cm_col2:
        st.write(f"**Reference/Training** (Thresh: {threshold:.2f})")
        st.write(f"TP={metrics_ref['tp']}, FP={metrics_ref['fp']}, FN={metrics_ref['fn']}, TN={metrics_ref['tn']}")


# --- Performance Curves Comparison ---
st.subheader("Performance Curves Comparison")
if can_compare_curves:
    col_plots1, col_plots2 = st.columns(2)
    with col_plots1:
        fig_roc_comp = plot_roc_comparison_interactive(
            eval_pred_df[Y_TRUE_COL], eval_pred_df[Y_PRED_PROB_COL],
            eval_ref_df[Y_TRUE_COL], eval_ref_df[Y_PRED_PROB_COL],
            name1="Prediction", name2="Reference"
        )
        st.plotly_chart(fig_roc_comp, use_container_width=True)
    with col_plots2:
        fig_pr_comp = plot_pr_comparison_interactive(
             eval_pred_df[Y_TRUE_COL], eval_pred_df[Y_PRED_PROB_COL],
             eval_ref_df[Y_TRUE_COL], eval_ref_df[Y_PRED_PROB_COL],
             name1="Prediction", name2="Reference"
         )
        st.plotly_chart(fig_pr_comp, use_container_width=True)
else:
     st.warning("Cannot generate comparison performance curves: Both datasets need binary labels.")


# --- Calibration Plot Comparison ---
st.subheader("Calibration Comparison")
if can_compare_curves:
    n_bins_cal = st.slider("Number of Bins for Calibration", 5, 20, 10, 1, key="cal_bins_comp")
    try:
        prob_pred_pred, prob_true_pred = get_calibration_data(eval_pred_df[Y_TRUE_COL], eval_pred_df[Y_PRED_PROB_COL], n_bins=n_bins_cal)
        prob_pred_ref, prob_true_ref = get_calibration_data(eval_ref_df[Y_TRUE_COL], eval_ref_df[Y_PRED_PROB_COL], n_bins=n_bins_cal)

        fig_cal_comp = plot_calibration_comparison_interactive(
            prob_pred_pred, prob_true_pred,
            prob_pred_ref, prob_true_ref,
            name1="Prediction", name2="Reference"
        )
        st.plotly_chart(fig_cal_comp, use_container_width=True)
    except ValueError as e:
         st.warning(f"Could not generate calibration comparison plot: {e}")
    except Exception as e:
         st.error(f"An unexpected error occurred during calibration plotting: {e}")
else:
     st.warning("Cannot generate calibration comparison plot: Both datasets need binary labels.")


# --- Rank-Based Metrics Comparison ---
st.subheader("Rank-Based Performance Comparison")
if can_compare_curves:
    n_bins_rank = st.slider("Number of Rank Bins (e.g., 10=Deciles)", 5, 20, 10, 1, key="rank_bins_comp")
    try:
        rank_metrics_pred_df = calculate_rank_metrics(eval_pred_df, Y_TRUE_COL, Y_PRED_PROB_COL, n_bins_rank)
        rank_metrics_ref_df = calculate_rank_metrics(eval_ref_df, Y_TRUE_COL, Y_PRED_PROB_COL, n_bins_rank)

        if not rank_metrics_pred_df.empty and not rank_metrics_ref_df.empty:
            with st.expander("Show Rank Metrics Tables"):
                 tcol1, tcol2 = st.columns(2)
                 with tcol1:
                     st.caption("Prediction/Holdout Rank Metrics")
                     st.dataframe(rank_metrics_pred_df)
                 with tcol2:
                     st.caption("Reference/Training Rank Metrics")
                     st.dataframe(rank_metrics_ref_df)

            col_rank1, col_rank2 = st.columns(2)
            with col_rank1:
                fig_rank_pr_comp = plot_rank_metrics_comparison_interactive(
                    rank_metrics_pred_df, rank_metrics_ref_df,
                    name1="Prediction", name2="Reference",
                    title="Precision & Recall vs Top % Targeted"
                )
                st.plotly_chart(fig_rank_pr_comp, use_container_width=True)
            with col_rank2:
                fig_lift_comp = plot_lift_chart_comparison_interactive(
                    rank_metrics_pred_df, rank_metrics_ref_df,
                    name1="Prediction", name2="Reference",
                    title="Lift Chart Comparison"
                )
                st.plotly_chart(fig_lift_comp, use_container_width=True)
        else:
            st.warning("Rank metrics calculation resulted in empty DataFrame(s). Cannot plot comparison.")
    except Exception as e:
        st.error(f"An error occurred during rank metric calculation or plotting: {e}")
else:
    st.warning("Cannot calculate rank-based metrics comparison: Both datasets need binary labels.")


# --- Prediction Score Distribution Comparison ---
st.subheader("Prediction Score Distribution Comparison")
# Data was already prepared and validated above
pred_scores = eval_pred_df[Y_PRED_PROB_COL]
ref_scores = eval_ref_df[Y_PRED_PROB_COL]

try:
    fig_dist_comp = plot_distribution_comparison_interactive(
        pred_scores, # Series 1 is Prediction
        ref_scores,  # Series 2 is Reference
        name1="Prediction/Holdout",
        name2="Reference/Training",
        feature_name="Prediction Score"
        # nbins = # Optionally add slider later
    )
    st.plotly_chart(fig_dist_comp, use_container_width=True)
except Exception as e:
     st.error(f"Could not plot score distribution comparison: {e}")

# --- Descriptive Stats Comparison ---
with st.expander("Show Descriptive Statistics Comparison"):
    stats_pred_df = calculate_descriptive_stats(pred_scores)
    stats_ref_df = calculate_descriptive_stats(ref_scores)

    if stats_pred_df is not None and stats_ref_df is not None:
        stats_comparison = pd.merge(
            stats_pred_df.rename(columns={'Value':'Prediction'}),
            stats_ref_df.rename(columns={'Value':'Reference'}),
            on='Metric',
            how='outer'
        )
        st.dataframe(stats_comparison)
    else:
        st.warning("Could not calculate descriptive statistics for one or both datasets.")