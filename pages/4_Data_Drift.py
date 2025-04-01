# pages/4_Data_Drift.py

import streamlit as st
import pandas as pd
import numpy as np
import warnings

# Import drift calculation and plotting functions
from ModelEvaluationApp.modules.data_drift_analysis import (
    calculate_drift_metrics,
    # plot_drift_summary, # This used matplotlib, let's use the plotly version
    # plot_top_feature_distributions # This used matplotlib grid
)
from ModelEvaluationApp.utils.plotting import (
    plot_distribution_comparison_interactive, # Use the plotly version
    plot_drift_summary_interactive # Use the plotly version
)
# Import other utilities if needed, e.g., for loading SHAP
# from utils.data_loader import load_shap_data

st.set_page_config(layout="wide")
st.header("Data Drift Analysis")
st.markdown("""
Compare the distribution of features between the Reference (e.g., Training) dataset
and the Current (e.g., Prediction/Holdout) dataset to identify potential data drift.
""")

# --- Data Check ---
if 'pred_df' not in st.session_state or st.session_state['pred_df'] is None:
    st.warning("Please load Prediction/Holdout data on the main page sidebar first.")
    st.stop()
if 'ref_df' not in st.session_state or st.session_state['ref_df'] is None:
    st.warning("Please load Reference/Training data on the main page sidebar to enable drift analysis.")
    st.stop()

pred_df = st.session_state['pred_df']
ref_df = st.session_state['ref_df']

# Check if SHAP importance is available
shap_importance_df_raw = st.session_state.get('shap_importance_df_raw')
shap_available = shap_importance_df_raw is not None and not shap_importance_df_raw.empty

# --- Feature Selection for Drift ---
st.subheader("Select Features for Analysis")

# Identify common columns suitable for drift analysis
cols_to_exclude = ['ContactId', 'EntityId', 'ResponseTimestamp', 'y_true', 'y_pred_prob'] # Adapt as needed
common_cols = list(set(pred_df.columns) & set(ref_df.columns))
features_drift_options = sorted([col for col in common_cols if col not in cols_to_exclude])

if not features_drift_options:
     st.warning("No common features found between datasets (excluding ID/target/pred columns) for drift analysis.")
     st.stop()

# Identify common columns suitable for drift analysis
cols_to_exclude = ['ContactId', 'EntityId', 'ResponseTimestamp', 'y_true', 'y_pred_prob'] # Adapt as needed
common_cols = list(set(pred_df.columns) & set(ref_df.columns))
features_drift_options_all = sorted([col for col in common_cols if col not in cols_to_exclude])

if not features_drift_options_all:
     st.warning("No common features found between datasets (excluding ID/target/pred columns) for drift analysis.")
     st.stop()

# --- Feature Selection Logic ---
st.sidebar.subheader("Select Features")
analysis_mode = st.sidebar.radio(
    "Analyze:",
    options=["All Common Features", "Top N Important Features", "Specific Features"],
    index=0 if not shap_available else 1, # Default to Top N if SHAP available
    key="drift_analysis_mode",
    help="Requires SHAP data to be loaded for 'Top N'." if not shap_available else None,
    disabled=not shap_available and "Top N Important Features" in ["Top N Important Features"] # Disable Top N if no SHAP
)

features_to_analyze = []
if analysis_mode == "All Common Features":
    features_to_analyze = features_drift_options_all
    st.sidebar.caption(f"Analyzing all {len(features_to_analyze)} common features.")
elif analysis_mode == "Top N Important Features" and shap_available:
    top_n = st.sidebar.slider(
        "Number of Top Features (by SHAP)", min_value=5, max_value=min(100, len(shap_importance_df_raw)),
        value=min(20, len(shap_importance_df_raw)), step=5, key="drift_top_n"
    )
    # Select features present in SHAP AND common to both datasets
    important_features_sorted = shap_importance_df_raw['feature'].tolist()
    features_to_analyze = [f for f in important_features_sorted if f in features_drift_options_all][:top_n]
    st.sidebar.caption(f"Analyzing top {len(features_to_analyze)} features by SHAP importance.")
elif analysis_mode == "Specific Features":
    features_to_analyze = st.sidebar.multiselect(
        "Select specific features:",
        options=features_drift_options_all,
        default=features_drift_options_all[:min(10, len(features_drift_options_all))]
    )
    st.sidebar.caption(f"Analyzing {len(features_to_analyze)} selected features.")
else: # Fallback or disabled Top N case
    features_to_analyze = features_drift_options_all
    st.sidebar.warning("SHAP data not loaded. Analyzing all common features.")


# --- Calculate Drift Metrics ---
st.subheader("Calculate Drift Metrics")

# Button to trigger calculation
if st.button("Calculate PSI & KS-Test", key="calc_drift_btn"):
    if not features_to_analyze:
        st.warning("Please select at least one feature to analyze.")
    else:
        with st.spinner("Calculating drift metrics..."):
            # Pass the RAW SHAP importance df (post-processor features)
            drift_df, drift_importance_df = calculate_drift_metrics(
                training_df=ref_df,
                prediction_df=pred_df,
                features_to_analyze=features_to_analyze,
                shap_importance_df=shap_importance_df_raw # Use raw importance here
            )
            st.session_state['drift_df'] = drift_df
            # Store the merged df (potentially with importance)
            st.session_state['drift_importance_df'] = drift_importance_df

            st.success(f"Drift metrics calculated for {len(drift_df)} features.")


# --- Display Drift Results ---
drift_df_calculated = st.session_state.get('drift_df')
drift_importance_df_calculated = st.session_state.get('drift_importance_df')

if drift_df_calculated is not None:
    st.markdown("#### Drift Metrics Summary Table")
    # Display the simple drift_df or the merged one if available
    display_df = drift_importance_df_calculated if drift_importance_df_calculated is not None else drift_df_calculated
    st.dataframe(display_df.sort_values('psi', ascending=False, key=abs)) # Show sorted by PSI

    # --- Drift Summary Plot (PSI vs Importance) ---
    st.markdown("#### Drift vs Importance Summary Plot")
    if drift_importance_df_calculated is not None:
         try:
             # Ensure required columns exist from the merge in calculate_drift_metrics
             if all(c in drift_importance_df_calculated for c in ['feature', 'psi', 'feature_importance', 'ks_pvalue']):
                 fig_summary = plot_drift_summary_interactive(drift_importance_df_calculated)
                 st.plotly_chart(fig_summary, use_container_width=True)
             else:
                 st.info("Drift calculation did not produce importance data (SHAP data might be missing or failed to merge). Cannot display summary plot.")
         except Exception as e:
              st.warning(f"Could not generate Drift Summary Plot: {e}")
    else:
         st.info("Load SHAP data and recalculate drift to view the Drift vs Importance Summary Plot.")

    st.markdown("---")

    # --- Distribution Comparison Plot ---
    st.markdown("#### Feature Distribution Comparison")
    if not drift_df_calculated.empty:
        feature_options = drift_df_calculated['feature'].tolist()
        if feature_options:
            # Default to highest PSI feature or first if PSI is NaN
            default_feature_ix = drift_df_calculated['psi'].abs().idxmax() if not drift_df_calculated['psi'].isnull().all() else 0
            default_feature = drift_df_calculated.loc[default_feature_ix, 'feature']

            feat_to_plot = st.selectbox(
                "Select feature to compare distributions:",
                options=feature_options,
                index=feature_options.index(default_feature) if default_feature in feature_options else 0, # Ensure default is valid
                key="dist_feat_select"
            )
            if feat_to_plot:
                 try:
                     # Get PSI/KS values for context
                     metrics_row = drift_df_calculated[drift_df_calculated['feature'] == feat_to_plot].iloc[0]
                     psi_val = metrics_row.get('psi', np.nan)
                     ks_p = metrics_row.get('ks_pvalue', np.nan)
                     st.caption(f"Feature: **{feat_to_plot}** | PSI: {psi_val:.3f} | KS p-value: {ks_p:.3f}")

                     # Optional: Add slider for number of bins
                     n_bins_dist = st.slider(f"Number of bins for '{feat_to_plot}'", 10, 100, 40, 5, key=f"nbins_{feat_to_plot}")

                     # Check if feature exists in original dataframes before plotting
                     if feat_to_plot in ref_df.columns and feat_to_plot in pred_df.columns:
                         fig_dist = plot_distribution_comparison_interactive(
                             ref_df[feat_to_plot],
                             pred_df[feat_to_plot],
                             name1="Reference (Train)",
                             name2="Current (Predict)",
                             feature_name=feat_to_plot,
                             nbins=n_bins_dist
                         )
                         st.plotly_chart(fig_dist, use_container_width=True)
                     else:
                          st.error(f"Feature '{feat_to_plot}' selected for plotting not found in Reference or Prediction data.")

                 except IndexError:
                     st.error(f"Could not find metrics for selected feature '{feat_to_plot}' in drift results.")
                 except Exception as e:
                      st.error(f"Could not plot distribution for '{feat_to_plot}': {e}")
        else:
            st.info("No features available in the calculated drift results to plot distributions.")
    else:
        st.info("Click 'Calculate PSI & KS-Test' to generate drift metrics.")