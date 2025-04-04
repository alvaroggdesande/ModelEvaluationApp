# pages/5_Explainability.py
import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import warnings
import traceback

# Import helpers if needed
from utils.shap_helpers import aggregate_shap_importance

st.set_page_config(layout="wide")
st.header("Model Explainability (SHAP)")
st.markdown("""
Explore model explanations using SHAP values. Understand global feature importance
and local explanations for individual predictions. Feature values are shown inverse-transformed
where possible for better interpretability in relevant plots (Beeswarm, Dependence, Waterfall).

**Requires SHAP data (Values, Features, Base Value, Display Data) to be loaded on the '🚀 Home' page.**
""")

# --- Data Check ---
# Check for the main dictionary AND the essential sampled scores/indices
if 'shap_data_dict' not in st.session_state or st.session_state['shap_data_dict'] is None \
   or 'shap_sampled_scores' not in st.session_state or st.session_state['shap_sampled_scores'] is None \
   or 'shap_sampled_indices' not in st.session_state or st.session_state['shap_sampled_indices'] is None:
    st.warning("❗ Sampled SHAP data (values, features, base, display, scores, indices) not fully loaded. Please load on '🚀 Home' page.")
    st.stop()
# Check if importance was calculated successfully in app.py
if 'shap_importance_df_raw' not in st.session_state or st.session_state['shap_importance_df_raw'] is None:
     st.warning("❗ SHAP importance calculation (from sample) failed or data not loaded properly.")
     st.stop()

# Retrieve loaded SAMPLED SHAP components
shap_data = st.session_state['shap_data_dict']
shap_values = shap_data['shap_values']         # Sampled SHAP values (n_samples_sampled, n_features_total)
feature_names = shap_data['feature_names']     # Full feature names list (n_features_total)
base_value = shap_data['base_value']
processed_data = shap_data['processed_data'] # Sampled display data (DataFrame: n_samples_sampled, n_features_total)
# Get importance calculated from the sample
importance_df_raw = st.session_state['shap_importance_df_raw'] # Keep this for the bar plot
importance_df_agg = st.session_state.get('shap_importance_df_agg')
# Get sampled scores and indices
sampled_scores = st.session_state['shap_sampled_scores']
# sampled_indices = st.session_state['shap_sampled_indices']

n_samples_sampled = shap_values.shape[0]
n_features_total = shap_values.shape[1]

st.success(f"✅ Sampled SHAP data loaded ({n_samples_sampled} samples, {n_features_total} features).")

# --- Sidebar Options ---
st.sidebar.header("Explainability Options")
max_display_features = st.sidebar.slider(
    "Max features for plots:", min_value=5, max_value=min(50, n_features_total),
    value=min(20, n_features_total), key="shap_max_display"
)
st.sidebar.caption(f"Plots use all {n_samples_sampled} samples loaded.")

# --- Global Feature Importance ---
st.subheader("Global Feature Importance (Calculated from Sample)") # Update title
st.markdown("""
Shows the average impact (mean absolute SHAP value) of each feature across the **sampled** data.
Features are ordered by importance.
""")

if importance_df_raw is not None and not importance_df_raw.empty:
    try:
        # Create bar chart directly from importance_df_raw (using Plotly recommended)
        import plotly.express as px
        fig_imp = px.bar(
            importance_df_raw.head(max_display_features),
            x='feature_importance', y='feature', orientation='h',
            title="Global Feature Importance (Sampled Data)",
            labels={'feature_importance': 'Mean Absolute SHAP Value (from Sample)', 'feature': 'Feature'}
        )
        fig_imp.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_imp, use_container_width=True)

        # Optional: Expander for Aggregated Importance Table
        if importance_df_agg is not None and not importance_df_agg.empty:
            with st.expander("Show Aggregated Importance (Original Features - based on Sample)"):
                st.dataframe(importance_df_agg)
        # else: st.caption("Aggregated importance data not available.") # Or if calculation failed

    except Exception as e:
        st.error(f"Could not generate importance bar plot: {e}")
else:
    st.warning("Raw SHAP importance data (from sample) is not available.")

st.markdown("---")

# --- SHAP Summary Plot (Beeswarm) ---
st.subheader("SHAP Summary Plot (Beeswarm - Sampled Data)")
st.markdown(f"""
Visualizes the SHAP value distribution for the **top {max_display_features} most important features** across the **{n_samples_sampled} sampled instances**.
Importance for ordering is determined by mean absolute SHAP within this plot.
- Color: Feature value (requires 'Display Data').
""")

# --- Create Explanation object using FULL Sampled Data ---
# This is the key change: Use the complete sample loaded into session state.
# Ensure 'processed_data' is the DataFrame with column names matching 'feature_names'.
use_display_data_for_color = False
if processed_data is not None:
    if list(processed_data.columns) == feature_names:
         if processed_data.shape == shap_values.shape:
              use_display_data_for_color = True
         else:
              st.warning("Display data shape mismatch with SHAP values. Cannot use for coloring.", icon="⚠️")
    else:
         st.warning("Display data column names mismatch with feature names. Cannot use for coloring.", icon="⚠️")

explanation_full_sample = shap.Explanation(
    values=shap_values,          # (n_samples_sampled, n_features_total)
    base_values=base_value,      # Single value or array matching samples
    data=processed_data if use_display_data_for_color else None, # (n_samples_sampled, n_features_total) - Use only if valid
    feature_names=feature_names  # List (n_features_total)
)
# --- End Explanation object creation ---


# --- Generate Beeswarm Plot using the Explanation object ---
# Let summary_plot handle selecting top N based on the explanation data
try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plt.figure()
        # Pass the full explanation object and max_display
        shap.summary_plot(
            explanation_full_sample,
            max_display=max_display_features, # Let the plot function select top N
            plot_type='dot',                  # Specify beeswarm type
            show=False
        )
        plt.title(f"SHAP Summary Plot - Top Features ({n_samples_sampled} Sampled Instances)")
        plt.xlabel("SHAP Value (Impact on Model Output)")
        plt.tight_layout()
        st.pyplot(plt.gcf())
        plt.clf() # Clear figure after displaying

except Exception as e:
      st.error(f"Could not generate beeswarm plot: {e}")
      st.error(traceback.format_exc()) # Show full error in app

# Display notes about coloring
if processed_data is not None and not use_display_data_for_color:
    st.info("Coloring by feature value unavailable due to data mismatch (check warnings above).")
elif processed_data is None:
     st.info("Load 'Display Data' on Home page sidebar to enable coloring by feature value.")

st.markdown("---")

# --- SHAP Dependence Plots ---
st.subheader("SHAP Dependence Plots (Sampled Data)") # Update title
st.markdown(f"""
Shows how the model's prediction depends on a single feature's value, using the **{n_samples_sampled} sampled instances**.
Vertical dispersion indicates interaction effects. Requires 'Display Data'.
""")

if processed_data is not None: # Check if SAMPLED display data exists
    col_dep1, col_dep2 = st.columns(2)
    with col_dep1:
        feature_to_plot = st.selectbox(
            "Select Feature for Dependence Plot:",
            options=feature_names, index=0, key="dep_feature"
        )
    with col_dep2:
        interaction_options = ['auto'] + feature_names
        interaction_feature = st.selectbox(
            "Select Feature for Interaction Coloring:",
            options=interaction_options, index=0, key="dep_interaction"
        )

    if feature_to_plot:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fig, ax = plt.subplots(figsize=(8, 6))
                # Use SAMPLED SHAP values and SAMPLED display data
                shap.dependence_plot(
                    feature_to_plot,
                    shap_values,      # Full sampled SHAP values
                    processed_data,   # Full sampled display data
                    feature_names=feature_names, # Pass full list for lookup
                    interaction_index=interaction_feature if interaction_feature != 'auto' else 'auto',
                    ax=ax,
                    show=False
                )
                plt.title(f"SHAP Dependence Plot for '{feature_to_plot}' ({n_samples_sampled} Sampled Instances)")
                plt.tight_layout()
                st.pyplot(fig)
                plt.clf()
        except ValueError as ve:
             if "is not in feature_names" in str(ve): st.error(f"Feature '{feature_to_plot}' or interaction '{interaction_feature}' not found.")
             else: st.error(f"Plot error for '{feature_to_plot}': {ve}")
        except Exception as e:
            st.error(f"Unexpected error generating dependence plot: {e}")
else:
    st.info("Load 'Display Data' on Home page for Dependence Plots.")

st.markdown("---")

# --- Local Explanation (Waterfall Plot) ---
st.subheader("Local Explanation (Waterfall Plot - Sampled Data)") # Update title
st.markdown("""
Explains a single prediction **from the sample**. Shows how features contributed to push the prediction
from the base value to the final predicted value for that instance.
""")

# --- Instance Selection within the SAMPLE ---
# Use the sampled_scores loaded in app.py
scores = pd.Series(sampled_scores) # Convert numpy array to Series for easier sorting/indexing
n_samples = len(scores) # Number of samples is now the size of the uploaded sample
sorted_indices_sample = scores.sort_values().index.to_list() # Indices are 0 to n_samples-1

selected_instance_index = None

sel_col1, sel_col2 = st.columns([1, 3])

with sel_col1:
    st.write("**Select by Score (within Sample):**")
    if st.button("Lowest Score", key="btn_lowest_sample"): st.session_state['selected_shap_index_sample'] = sorted_indices_sample[0]
    if st.button("Median Score", key="btn_median_sample"): st.session_state['selected_shap_index_sample'] = sorted_indices_sample[n_samples // 2]
    if st.button("Highest Score", key="btn_highest_sample"): st.session_state['selected_shap_index_sample'] = sorted_indices_sample[-1]

    st.write("**Select by Sample Index:**")
    index_input_key = "waterfall_index_input_sample"
    # Use session state key unique to this sampled approach
    default_index = st.session_state.get('selected_shap_index_sample', 0)
    # Ensure default index is within bounds
    if default_index >= n_samples: default_index = 0

    raw_index = st.number_input(
        f"Index (0 to {n_samples-1}):", # Use sample size limit
        min_value=0, max_value=n_samples-1,
        value=default_index,
        step=1, key=index_input_key
    )
    # Update session state if raw index is changed manually
    if raw_index != st.session_state.get('selected_shap_index_sample', default_index):
         st.session_state['selected_shap_index_sample'] = raw_index

with sel_col2:
    st.write("**Select by Score Rank (within Sample):**")
    # Find the rank of the currently selected index within the sorted sample indices
    current_rank_default = 0
    try:
        current_rank_default = sorted_indices_sample.index(st.session_state.get('selected_shap_index_sample', sorted_indices_sample[0]))
    except ValueError: # If the index isn't in the list somehow, default to 0
        pass

    slider_rank = st.slider(
        "Score Rank (Lowest to Highest in Sample)",
        min_value=0,
        max_value=n_samples - 1, # Use sample size limit
        value=current_rank_default,
        step=1,
        key="score_rank_slider_sample",
    )
    # If slider changes, update the selected index in session state
    new_index_from_slider = sorted_indices_sample[slider_rank]
    try:
        current_score = scores.loc[new_index_from_slider]
        st.caption(f"Selected Rank: {slider_rank} → Score: {current_score:.4f}")
    except Exception as e:
         st.caption(f"Selected Rank: {slider_rank} (Error displaying score: {e})")

    if new_index_from_slider != st.session_state.get('selected_shap_index_sample'):
        st.session_state['selected_shap_index_sample'] = new_index_from_slider
        st.rerun() # Rerun needed to update the number_input

# Determine the final index to plot (within the sample)
instance_index = st.session_state.get('selected_shap_index_sample', raw_index)

# Ensure instance_index is valid before proceeding
if not (0 <= instance_index < n_samples):
    st.error(f"Selected index {instance_index} is out of bounds for the sampled data (0 to {n_samples-1}). Resetting to 0.")
    instance_index = 0
    st.session_state['selected_shap_index_sample'] = 0

st.info(f"Displaying Waterfall plot for **Sample Index: {instance_index}**" +
        (f" (Score: {scores.loc[instance_index]:.4f})" if instance_index in scores.index else ""))

# Generate Waterfall Plot (using data for the instance_index from the SAMPLED arrays)
try:
    instance_shap_values = shap_values[instance_index]
    instance_data = processed_data.iloc[instance_index] if processed_data is not None else None

    explanation = shap.Explanation(
        values=instance_shap_values,
        base_values=base_value,
        data=instance_data, # Uses SAMPLED display data
        feature_names=feature_names
    )

    with warnings.catch_warnings():
         warnings.simplefilter("ignore")
         plt.figure(figsize=(10, max_display_features * 0.25 + 1))
         shap.plots.waterfall(explanation, max_display=max_display_features, show=False)
         plt.title(f"Waterfall Plot for Sample Index {instance_index}")
         plt.tight_layout()
         st.pyplot(plt.gcf())
         plt.clf()

except Exception as e:
     st.error(f"Could not generate waterfall plot for sample index {instance_index}: {e}")
     st.error(traceback.format_exc()) # Show detailed error in app for debugging