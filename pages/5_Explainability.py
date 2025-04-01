# pages/5_Explainability.py
import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import warnings

# Import helpers if needed
from ModelEvaluationApp.utils.shap_helpers import aggregate_shap_importance

st.set_page_config(layout="wide")
st.header("Model Explainability (SHAP)")
st.markdown("""
Explore model explanations using SHAP values. Understand global feature importance
and local explanations for individual predictions. Feature values are shown inverse-transformed
where possible for better interpretability in relevant plots (Beeswarm, Dependence, Waterfall).

**Requires SHAP data (Values, Features, Base Value, Display Data) to be loaded on the '🚀 Home' page.**
""")

# --- Data Check ---
if 'shap_data_dict' not in st.session_state or st.session_state['shap_data_dict'] is None:
    st.warning("❗ SHAP data not loaded. Please load it on the '🚀 Home' page sidebar.")
    st.stop()
# Ensure importance_df_raw exists, crucial for feature ordering
if 'shap_importance_df_raw' not in st.session_state or st.session_state['shap_importance_df_raw'] is None:
     st.warning("❗ SHAP importance calculation failed or data not loaded. Cannot determine feature order.")
     st.stop()

# Retrieve loaded SHAP components
shap_data = st.session_state['shap_data_dict']
shap_values = shap_data['shap_values']         # Full SHAP values
feature_names = shap_data['feature_names']     # Full feature names list
base_value = shap_data['base_value']
processed_data = shap_data['processed_data'] # Full display data (inverse transformed where possible)
importance_df_raw = st.session_state['shap_importance_df_raw'] # Importance based on full SHAP values
importance_df_agg = st.session_state.get('shap_importance_df_agg') # Might be None

n_samples_total = shap_values.shape[0]
n_features_total = shap_values.shape[1]

st.success(f"✅ SHAP data loaded ({n_samples_total} samples, {n_features_total} features). Generating explanations...")

# --- Sidebar Options ---
st.sidebar.header("Explainability Options")
max_display_features = st.sidebar.slider(
    "Max features for plots:", min_value=5, max_value=min(50, n_features_total),
    value=min(20, n_features_total), key="shap_max_display"
)
# Sampling limit
MAX_SAMPLES_DEFAULT = 10000
max_samples_for_plots = st.sidebar.number_input(
    f"Max samples for Beeswarm/Dependence plots:",
    min_value=100, max_value=n_samples_total,
    value=min(MAX_SAMPLES_DEFAULT, n_samples_total),
    step=100,
    key="shap_max_samples",
    help=f"Limits samples for performance. Full data has {n_samples_total} samples."
)


# --- Global Feature Importance ---
st.subheader("Global Feature Importance")
st.markdown("""
Shows the average impact (mean absolute SHAP value) of each feature (as seen by the model)
on the model output magnitude. Features are ordered by importance, and less important features
are automatically grouped.
""")

if importance_df_raw is not None and not importance_df_raw.empty:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # --- Create SHAP Explanation object for consistency ---
            # This often helps SHAP plots behave as expected, especially with aggregation
            explanation_full = shap.Explanation(
                values=shap_values,
                base_values=base_value,
                data=processed_data, # Include display data if available
                feature_names=feature_names
            )

            plt.figure() # Create a figure context for SHAP
            # Pass the Explanation object to the bar plot
            shap.summary_plot(
                explanation_full,     # Pass Explanation object
                plot_type='bar',
                max_display=max_display_features,
                show=False
            )
            plt.title("Global Feature Importance (Processed Features)")
            plt.xlabel("Mean Absolute SHAP Value (|SHAP value|)")
            plt.tight_layout()
            st.pyplot(plt.gcf())
            plt.clf() # Clear the figure

        # --- Optional: Expander for Aggregated Importance Table ---
        # ... (expander code remains the same) ...

    except Exception as e:
        st.error(f"Could not generate global importance bar plot: {e}")

else:
    st.warning("Raw SHAP importance data is not available.")


st.markdown("---")


# --- Apply Sampling Logic ONCE before Beeswarm and Dependence ---
shap_values_sampled = shap_values
processed_data_sampled = processed_data
sample_indices = None

if n_samples_total > max_samples_for_plots:
    st.info(f"Large dataset detected ({n_samples_total} samples). Displaying beeswarm/dependence plots using a random sample of {max_samples_for_plots} instances for performance.", icon="ℹ️")
    np.random.seed(42)
    sample_indices = np.random.choice(n_samples_total, max_samples_for_plots, replace=False)
    shap_values_sampled = shap_values[sample_indices, :]
    if processed_data is not None:
        processed_data_sampled = processed_data.iloc[sample_indices]
else:
    st.info(f"Displaying beeswarm/dependence plots using all {n_samples_total} samples.", icon="ℹ️")


# --- SHAP Summary Plot (Beeswarm) ---
st.subheader("SHAP Summary Plot (Beeswarm)")
st.markdown(f"""
Visualizes the SHAP value distribution for the **top {max_display_features} most important features** across the {'sampled (' + str(max_samples_for_plots) + ')' if sample_indices is not None else 'full'} dataset.
- Each point is a SHAP value for a specific feature and sample.
- Y-axis: Features (ordered by global importance based on the **full dataset**).
- X-axis: SHAP value (impact on prediction).
- Color: Feature value (High=Red, Low=Blue). Requires 'Display Data'.
- *Note: This plot only shows top features. See the bar plot above for the aggregated impact of 'Other Features'.*
""")

# --- Get Top N Feature Indices and Names based on FULL importance_df_raw ---
top_feature_names = importance_df_raw.head(max_display_features)['feature'].tolist()
try:
    # Find indices in the original full list
    top_feature_indices_original = [feature_names.index(f) for f in top_feature_names]
except ValueError as e:
    st.error(f"Error finding indices for top features. Feature name mismatch? Error: {e}")
    # Fallback: Use first N indices, but warn user order might be wrong
    top_feature_indices_original = list(range(max_display_features))
    top_feature_names = feature_names[:max_display_features] # Adjust names to match fallback
    st.warning("Using first {max_display_features} features due to index error. Order may not reflect true importance.")


# --- Slice the (potentially sampled) data to only include top features ---
# Use the indices found in the original list to slice columns of SAMPLED data
shap_values_top_features_sampled = shap_values_sampled[:, top_feature_indices_original]

processed_data_top_features_sampled = None
if processed_data_sampled is not None:
    try:
        # Select columns using the actual feature names corresponding to the indices
        processed_data_top_features_sampled = processed_data_sampled[top_feature_names]
    except KeyError as e:
         st.warning(f"Could not select display data for top features (needed for coloring). Plotting without color. Error: {e}", icon="⚠️")
         processed_data_top_features_sampled = None # Plot without color
    except Exception as e:
         st.warning(f"An unexpected error occurred slicing display data: {e}", icon="⚠️")
         processed_data_top_features_sampled = None


# --- Generate Beeswarm Plot ---
plot_title_suffix = f"(Sampled: {shap_values_sampled.shape[0]} instances)" if sample_indices is not None else "(Full Data)"
use_color = processed_data_top_features_sampled is not None

try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plt.figure() # Create a figure context for SHAP

        # --- Create Explanation object with SLICED/SAMPLED data ---
        # This is often more reliable for plotting functions
        explanation_beeswarm = shap.Explanation(
             values=shap_values_top_features_sampled,
             # Base values usually scalar, don't slice. Use original base_value.
             base_values=base_value,
             data=processed_data_top_features_sampled, # Sliced display data for coloring
             feature_names=top_feature_names # Only names for the top features
         )

        # Pass the Explanation object containing only top-feature data
        shap.summary_plot(
            explanation_beeswarm,
            plot_type='dot',
            # max_display is implicitly handled by the data slicing now
            show=False
        )

        plt.title(f"SHAP Summary Plot - Top {len(top_feature_names)} Features {plot_title_suffix}")
        plt.xlabel("SHAP Value (Impact on Model Output)")
        plt.tight_layout()
        st.pyplot(plt.gcf())
        plt.clf() # Clear the figure

except Exception as e:
      st.error(f"Could not generate beeswarm plot: {e}")

# Display note if coloring was not possible
if processed_data is not None and not use_color: # Check if display data was loaded but slicing failed
    st.info("Coloring by feature value unavailable (issue selecting top features from display data).")
elif processed_data is None:
     st.info("💡 Load the 'Display Data' on the Home page sidebar to enable coloring by feature value in the beeswarm plot.")


st.markdown("---")


# --- SHAP Dependence Plots ---
st.subheader("SHAP Dependence Plots")
st.markdown(f"""
Shows how the model's prediction for a single feature depends on its value, using the
{'sampled (' + str(max_samples_for_plots) + ')' if sample_indices is not None else 'full'} dataset.
Vertical dispersion indicates interaction effects. Requires 'Display Data'.
""")

if processed_data_sampled is not None: # Check if sampled display data exists
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
                # Use sampled SHAP values and sampled display data
                shap.dependence_plot(
                    feature_to_plot,
                    shap_values_sampled,      # Sampled SHAP values
                    processed_data_sampled,   # Sampled display data
                    feature_names=feature_names, # Pass full list for lookup
                    interaction_index=interaction_feature if interaction_feature != 'auto' else 'auto',
                    ax=ax,
                    show=False
                )
                plot_title_suffix = f"(Sampled: {max_samples_for_plots} instances)" if sample_indices is not None else "(Full Data)"
                plt.title(f"SHAP Dependence Plot for '{feature_to_plot}' {plot_title_suffix}")
                plt.tight_layout()
                st.pyplot(fig)
                plt.clf()
        except ValueError as ve:
            # Specific check for feature not found, common in dependence plot
             if "is not in feature_names" in str(ve):
                  st.error(f"Feature '{feature_to_plot}' or interaction feature '{interaction_feature}' not found in the provided feature names list.")
             else:
                  st.error(f"Could not generate dependence plot for '{feature_to_plot}': {ve}")
        except Exception as e:
            st.error(f"An unexpected error occurred during dependence plot generation: {e}")
else:
    st.info("💡 Load the 'Display Data' on the Home page sidebar to enable Dependence Plots.")


st.markdown("---")


# --- Local Explanation (Waterfall Plot) ---
st.subheader("Local Explanation (Waterfall Plot)")
st.markdown("""
Explains a single prediction. Shows how features contributed to push the prediction
from the base value (average prediction) to the final predicted value for that instance.
Select an instance using the options below.
""")

# --- Check if Prediction Data is available for score-based selection ---
pred_df = st.session_state.get('pred_df')
scores_available = False
sorted_indices = None
scores = None

if pred_df is not None and 'y_pred_prob' in pred_df.columns:
    if len(pred_df) == shap_values.shape[0]:
        try:
            scores = pd.to_numeric(pred_df['y_pred_prob'], errors='coerce').dropna()
            # Ensure scores align with shap_values after potential drops
            if len(scores) == shap_values.shape[0]:
                # Get indices sorted by score
                sorted_indices = scores.sort_values().index.to_list() # Sorts ascending (Lowest to Highest)
                scores_available = True
            else:
                 st.warning("Prediction scores length mismatch after cleaning. Cannot use score-based selection.", icon="⚠️")
        except Exception as e:
            st.warning(f"Could not process prediction scores for selection: {e}", icon="⚠️")
    else:
        st.warning("Prediction data rows mismatch SHAP values rows. Cannot use score-based selection.", icon="⚠️")
else:
    st.warning("Load Prediction/Holdout data with 'y_pred_prob' column on Home page for score-based selection.", icon="⚠️")


# --- Instance Selection ---
n_samples = shap_values.shape[0]
selected_instance_index = None # Initialize

sel_col1, sel_col2 = st.columns([1, 3]) # Adjust column ratios as needed

with sel_col1:
    st.write("**Select by Category:**")
    if st.button("Lowest Score", key="btn_lowest", disabled=not scores_available):
        if sorted_indices:
            st.session_state['selected_shap_index'] = sorted_indices[0]
    if st.button("Median Score", key="btn_median", disabled=not scores_available):
         if sorted_indices:
             st.session_state['selected_shap_index'] = sorted_indices[len(sorted_indices) // 2]
    if st.button("Highest Score", key="btn_highest", disabled=not scores_available):
         if sorted_indices:
             st.session_state['selected_shap_index'] = sorted_indices[-1]
    st.write("**Select by Index:**")
    # Keep the index input as a fallback or precise selection method
    index_input_key = "waterfall_index_input"
    raw_index = st.number_input(
        f"Index (0 to {n_samples-1}):",
        min_value=0, max_value=n_samples-1,
        value=st.session_state.get('selected_shap_index', 0), # Default to session state or 0
        step=1, key=index_input_key
    )
    # Update session state if raw index is changed manually
    if raw_index != st.session_state.get('selected_shap_index', 0):
         st.session_state['selected_shap_index'] = raw_index


with sel_col2:
    st.write("**Select by Score Rank:**")
    if scores_available and sorted_indices:
        # Slider represents rank (0=lowest, N-1=highest)
        slider_rank = st.slider(
            "Score Rank (Lowest to Highest)",
            min_value=0,
            max_value=n_samples - 1,
            value=sorted_indices.index(st.session_state.get('selected_shap_index', sorted_indices[0])), # Find rank of current index
            step=1,
            key="score_rank_slider",
            # format_func=... REMOVED THIS LINE
            # The default format for the slider value (the rank number) is fine.
        )
        # If slider changes, update the selected index in session state
        new_index_from_slider = sorted_indices[slider_rank]

        # --- Display the score corresponding to the selected rank ---
        try:
            current_score = scores.loc[new_index_from_slider]
            st.caption(f"Selected Rank: {slider_rank} → Score: {current_score:.4f}") # Display score separately
        except KeyError:
            st.caption(f"Selected Rank: {slider_rank} (Score unavailable)")
        except Exception as e:
             st.caption(f"Selected Rank: {slider_rank} (Error displaying score: {e})")
        # --- End score display ---


        if new_index_from_slider != st.session_state.get('selected_shap_index'):
            st.session_state['selected_shap_index'] = new_index_from_slider
            # Rerun slightly to update the number input field if slider was moved
            st.rerun()

    else:
        st.caption("Score-based slider disabled. Load Prediction data with scores.")

# Determine the final index to plot
instance_index = st.session_state.get('selected_shap_index', raw_index) # Use session state if available

st.info(f"Displaying Waterfall plot for **Sample Index: {instance_index}**" +
        (f" (Score: {scores.loc[instance_index]:.4f})" if scores_available and instance_index in scores.index else ""))

# --- Generate Waterfall Plot ---
# IMPORTANT: Waterfall plot uses the FULL (unsampled) data for the specific instance
if instance_index >= 0 and instance_index < n_samples_total: # Check against total samples
    try:
        # Get data for the specific instance from the ORIGINAL, UNSAMPLED arrays/dfs
        instance_shap_values = shap_values[instance_index]
        instance_data = processed_data.iloc[instance_index] if processed_data is not None else None

        explanation = shap.Explanation(
            values=instance_shap_values,
            base_values=base_value,
            data=instance_data, # This uses the potentially inverse-transformed display data
            feature_names=feature_names
        )

        with warnings.catch_warnings():
             warnings.simplefilter("ignore")
             plt.figure(figsize=(10, max_display_features * 0.25 + 1))
             # shap.plots.waterfall handles showing top N + Other for the instance
             shap.plots.waterfall(explanation, max_display=max_display_features, show=False)
             plt.title(f"Waterfall Plot for Sample Index {instance_index}")
             plt.tight_layout()
             st.pyplot(plt.gcf())
             plt.clf()

    except IndexError:
         st.error(f"Selected index {instance_index} is out of bounds for SHAP/display data. Max index is {n_samples_total-1}.")
    except Exception as e:
         st.error(f"Could not generate waterfall plot for index {instance_index}: {e}")
         # import traceback
         # st.error(traceback.format_exc()) # Uncomment for detailed debugging

else:
    st.warning("Invalid sample index selected.")