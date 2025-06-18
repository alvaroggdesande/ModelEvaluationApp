# pages/6_Impact_Simulator.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
import plotly.express as px # For the Net Result vs N Contacted plot

# Import your plotting and calculation utils
# Ensure these paths and function names are correct based on your project structure
try:
    from utils.plotting import (
        plot_roc_curve_interactive,
        plot_pr_curve_interactive,
        plot_rank_metrics_interactive,
        plot_lift_chart_interactive,
        plot_distribution_comparison_interactive
    )
    from utils.calculations import calculate_rank_metrics
except ImportError:
    st.error("Could not import utility functions. Please ensure utils/plotting.py and utils/calculations.py are correctly set up.")
    st.stop()


st.set_page_config(layout="wide")
st.title("Fundraising Campaign Impact Simulator")

# --- GLOBAL-LIKE SIMULATION PARAMETERS (from sidebar) ---
st.sidebar.header("Scenario Base Settings")
TOTAL_POPULATION_INPUT = st.sidebar.number_input(
    "Total Potential Donors in List",
    value=1000, min_value=50, max_value=100000, step=50, key="total_pop_sim" # Min value 50 for more stable plots
)
ACTUAL_DONORS_COUNT_INPUT = st.sidebar.number_input(
    "Actual Donors if All Contacted",
    value=100, min_value=5, max_value=TOTAL_POPULATION_INPUT, step=5, key="actual_donors_sim" # Min value 5
)

if ACTUAL_DONORS_COUNT_INPUT > TOTAL_POPULATION_INPUT:
    st.sidebar.error("Actual donors cannot exceed total population.")
    st.stop()
if ACTUAL_DONORS_COUNT_INPUT <= 0 or TOTAL_POPULATION_INPUT <= 0:
    st.sidebar.error("Population and donor counts must be positive.")
    st.stop()


# Assign to more convenient variable names for use in the script
TOTAL_POPULATION = TOTAL_POPULATION_INPUT
ACTUAL_DONORS_COUNT = ACTUAL_DONORS_COUNT_INPUT
ACTUAL_NON_DONORS_COUNT = TOTAL_POPULATION - ACTUAL_DONORS_COUNT

st.markdown(f"""
This simulator demonstrates how different quality prediction models can impact the outcome
of a fundraising campaign.

**Current Scenario:**
- We have a list of **{TOTAL_POPULATION:,}** potential donors.
- If we contact everyone, **{ACTUAL_DONORS_COUNT:,}** people are expected to donate.
""")

# --- Helper function to generate simulated data ---
# This function is NOT cached because its output depends on sidebar inputs TOTAL_POPULATION and ACTUAL_DONORS_COUNT
def load_or_generate_simulated_data(total_pop_func_arg, actual_donors_func_arg):
    actual_non_donors_func_arg = total_pop_func_arg - actual_donors_func_arg
    if actual_donors_func_arg <= 0 or actual_non_donors_func_arg < 0: # Basic validation
        st.error("Invalid population/donor counts for simulation.")
        return pd.DataFrame()

    person_ids = np.arange(1, total_pop_func_arg + 1)
    y_true = np.array([1] * actual_donors_func_arg + [0] * actual_non_donors_func_arg)
    np.random.seed(42) # for reproducibility
    np.random.shuffle(y_true)

    sim_df = pd.DataFrame({'person_id': person_ids, 'y_true': y_true})

    model_scenarios = {
        "Perfect Model (AUC=1.0)": "perfect", # Deterministic

        # Aiming for AUC ~0.90-0.95 (Still very distinct, minimal overlap)
        "Excellent Model (AUC ~0.92)": (0.68, 0.15, 0.38, 0.15), # Donor(mean,std), Non-Donor(mean,std). Increased non-donor mean slightly.

        # Aiming for AUC ~0.80-0.85 (Good separation, but clear overlap now)
        "Good Model (AUC ~0.83)": (0.65, 0.16, 0.42, 0.16), # Increased std for more overlap, means closer.

        # Aiming for AUC ~0.70-0.75 (Significant overlap, clearly not "great")
        "Fair Model (AUC ~0.72)": (0.60, 0.18, 0.45, 0.18), # Means even closer, std larger.

        # Aiming for AUC ~0.60-0.65 (Very heavy overlap, model is struggling)
        "Poor Model (AUC ~0.63)": (0.55, 0.20, 0.50, 0.20), # Means very close, large std.

        # Aiming for AUC ~0.50 (Effectively random)
        "Random Guessing (AUC ~0.50)": (0.50, 0.22, 0.50, 0.22) # Same means, wide std.
    }

    for name, params in model_scenarios.items():
        scores = np.zeros(total_pop_func_arg)
        if params == "perfect":
            # For a perfect model, assign high scores to actual donors, low to non-donors
            # Ensure NO OVERLAP for AUC=1
            min_score_donor = 0.51 # Min score for a donor
            max_score_non_donor = 0.49 # Max score for a non-donor
            for i, actual in enumerate(sim_df['y_true']):
                if actual == 1: # Donor
                    scores[i] = np.random.uniform(min_score_donor, 0.999)
                else: # Non-donor
                    scores[i] = np.random.uniform(0.001, max_score_non_donor)
        else:
            d_mean, d_std, nd_mean, nd_std = params
            for i, actual in enumerate(sim_df['y_true']):
                if actual == 1:
                    scores[i] = np.random.normal(d_mean, d_std)
                else:
                    scores[i] = np.random.normal(nd_mean, nd_std)
        sim_df[name] = np.clip(scores, 0.001, 0.999) # Clip to avoid exactly 0 or 1
    return sim_df

# --- Load or Generate Data ---
sim_data_df = load_or_generate_simulated_data(TOTAL_POPULATION, ACTUAL_DONORS_COUNT)

if sim_data_df.empty:
    st.error("Failed to generate simulation data based on current settings.")
    st.stop()

# --- Inputs for Campaign ---
st.sidebar.header("Campaign Settings")
cost_per_letter = st.sidebar.number_input("Cost per Letter (€)", value=1.00, min_value=0.01, step=0.01, format="%.2f")
avg_donation_amount = st.sidebar.number_input("Average Donation Amount (€)", value=25.00, min_value=1.00, step=1.00, format="%.2f")

model_names_list = [col for col in sim_data_df.columns if col not in ['person_id', 'y_true']]
selected_model_name = st.sidebar.selectbox("Select Model Simulation", model_names_list)

# Calculate and display AUC for the selected model simulation
y_true_for_auc = sim_data_df['y_true']
y_pred_prob_for_auc = sim_data_df[selected_model_name]
auc_score = np.nan
if len(y_true_for_auc.unique()) > 1: # AUC is defined for binary classes
    try:
        auc_score = roc_auc_score(y_true_for_auc, y_pred_prob_for_auc)
        st.sidebar.metric("Selected Model AUC", f"{auc_score:.3f}")
    except ValueError:
        st.sidebar.caption("AUC not computable (e.g., only one class).")
else:
    st.sidebar.caption("AUC not computable (only one class in y_true).")


# Slider for number of people to target
max_targetable_slider = TOTAL_POPULATION
min_val_slider = min(max(1, int(TOTAL_POPULATION*0.01)), max_targetable_slider) # Min 1% or 1, but not more than total
val_slider = min(max(min_val_slider, int(TOTAL_POPULATION*1)), max_targetable_slider) # Default 20% or min_val_slider
step_slider = max(1, int(TOTAL_POPULATION*0.01)) # Step 1% or 1

num_to_target = st.sidebar.slider(
    f"Number to contact (Top {TOTAL_POPULATION:,} by score):",
    min_value=min_val_slider,
    max_value=max_targetable_slider,
    value=val_slider,
    step=step_slider,
    key="num_target_slider"
)

# --- Calculations for Interactive Outcome ---
working_df = sim_data_df[['person_id', 'y_true', selected_model_name]].copy()
working_df = working_df.sort_values(by=selected_model_name, ascending=False)

targeted_df = working_df.head(num_to_target)
# Ensure tail calculation is correct even if num_to_target is TOTAL_POPULATION
not_targeted_df = working_df.iloc[num_to_target:] # Use iloc for safer slicing after head

# Confusion Matrix components
TP = targeted_df['y_true'].sum()
FP = len(targeted_df) - TP

FN = not_targeted_df['y_true'].sum()
TN = ACTUAL_NON_DONORS_COUNT - FP

# Metrics
precision = TP / num_to_target if num_to_target > 0 else 0
recall = TP / ACTUAL_DONORS_COUNT if ACTUAL_DONORS_COUNT > 0 else 0
overall_positive_rate = ACTUAL_DONORS_COUNT / TOTAL_POPULATION if TOTAL_POPULATION > 0 else 0
lift = precision / overall_positive_rate if overall_positive_rate > 0 else 0

# Financials
letters_sent = num_to_target
cost_of_mailing = letters_sent * cost_per_letter
donations_received_model = TP * avg_donation_amount
net_gain_loss_model = donations_received_model - cost_of_mailing

# Baseline: Contacting Everyone
cost_of_mailing_all = TOTAL_POPULATION * cost_per_letter
donations_received_all = ACTUAL_DONORS_COUNT * avg_donation_amount
net_gain_loss_all = donations_received_all - cost_of_mailing_all

# --- Display Section 1: Interactive Campaign Outcome ---
st.subheader(f"Interactive Outcome: Targeting Top {num_to_target:,} ({num_to_target/TOTAL_POPULATION:.1%} of List)")
st.markdown(f"Using **{selected_model_name}** to rank individuals.")

# --- Row 1: Performance and Confusion Matrix ---
# This row will contain the non-financial metrics and the confusion matrix
row1_col1, row1_col2 = st.columns(2)

with row1_col1:
    st.markdown("##### Performance at this Targeting Level")
    st.metric("Precision (Donors among contacted)", f"{precision:.2%}",
              help="Of all the people contacted, what percentage were actual donors? (TP / (TP + FP))")
    st.metric("Recall (Donors found out of all possible)", f"{recall:.2%}",
              help="Of all the actual donors in the full list, what percentage did we successfully contact? (TP / (TP + FN))")
    st.metric("Lift over Random", f"{lift:.2f}x",
              help="How many times better is this model at finding donors compared to random selection?")

with row1_col2:
    st.markdown("##### Confusion Matrix")
    cm_data = {
        f'Predicted Donor (Targeted: {num_to_target:,})': [f"{TP:,}", f"{FP:,}"],
        f'Predicted Non-Donor (Not Targeted: {(TOTAL_POPULATION - num_to_target):,})': [f"{FN:,}", f"{TN:,}"]
    }
    cm_df = pd.DataFrame(cm_data, index=[f'Actual Donor ({ACTUAL_DONORS_COUNT:,})', f'Actual Non-Donor ({ACTUAL_NON_DONORS_COUNT:,})'])
    st.table(cm_df) # Using st.table as you mentioned you haven't changed this yet

    # Explicitly show TP, FP, FN, TN with explanations (optional, but helpful)
    st.markdown("###### Understanding the Matrix Cell Values:")
    cm_breakdown_col1, cm_breakdown_col2 = st.columns(2)
    with cm_breakdown_col1:
        st.markdown(f"**TP (True Positives): {TP:,}**<br><small>Actual donors correctly contacted.</small>", unsafe_allow_html=True)
        st.markdown(f"**FP (False Positives): {FP:,}**<br><small>Non-donors mistakenly contacted.</small>", unsafe_allow_html=True)

    with cm_breakdown_col2:
        st.markdown(f"**FN (False Negatives): {FN:,}**<br><small>Actual donors we missed.</small>", unsafe_allow_html=True)
        st.markdown(f"**TN (True Negatives): {TN:,}**<br><small>Non-donors correctly not contacted.</small>", unsafe_allow_html=True)

# Add a more prominent separator before the financial comparison
st.markdown("<hr style='border:1px solid #e0e0e0; margin-top:1.5em; margin-bottom:1.5em;'>", unsafe_allow_html=True)

# --- Row 2: Financial Comparison ---
st.markdown("##### Campaign Financials: Model vs. Baseline") # A common subheader for the financial section
fin_col1, fin_col2 = st.columns(2) # New columns specifically for aligned financials

with fin_col1:
    # Centered sub-subheader for this column
    st.markdown("<p style='text-align:center; font-weight:bold;'>Model-Based Approach</p>", unsafe_allow_html=True)
    st.metric("Letters Sent", f"{letters_sent:,}")
    st.metric("Cost of Mailing", f"{cost_of_mailing:,.2f} €")
    st.metric("Donations Received", f"{donations_received_model:,.2f} €")
    delta_vs_baseline_net = net_gain_loss_model - net_gain_loss_all
    st.metric("Net Financial Result", f"{net_gain_loss_model:,.2f} €",
              delta=f"{delta_vs_baseline_net:,.2f} € vs Baseline",
              help="Difference in net result compared to contacting everyone.")

with fin_col2:
    # Centered sub-subheader for this column
    st.markdown("<p style='text-align:center; font-weight:bold;'>Baseline (Contacting Everyone)</p>", unsafe_allow_html=True)
    st.metric("Letters Sent (All)", f"{TOTAL_POPULATION:,}")
    st.metric("Cost of Mailing (All)", f"{cost_of_mailing_all:,.2f} €")
    st.metric("Donations Received (All)", f"{donations_received_all:,.2f} €")
    st.metric("Net Financial Result (All)", f"{net_gain_loss_all:,.2f} €")

st.markdown("---") # Standard separator after this section

# --- Display Section 2: Selected Model's Overall Characteristics ---
st.subheader(f"Overall Characteristics of '{selected_model_name}'")
st.markdown(f"Evaluated on the full list of {TOTAL_POPULATION:,} individuals.")

eval_df_sim = sim_data_df[['y_true', selected_model_name]].copy()
eval_df_sim = eval_df_sim.rename(columns={selected_model_name: 'y_pred_prob'})

# Score Distribution Plot
st.markdown("##### Score Distribution by Actual Outcome")
st.markdown("How well does this model separate actual donors from actual non-donors?") # Added context
scores_donors = sim_data_df[sim_data_df['y_true'] == 1][selected_model_name]
scores_non_donors = sim_data_df[sim_data_df['y_true'] == 0][selected_model_name]

if not scores_donors.empty or not scores_non_donors.empty:
    fig_dist = plot_distribution_comparison_interactive(
        scores_donors,
        scores_non_donors,
        name1=f"Actual Donors ({len(scores_donors):,})",
        name2=f"Actual Non-Donors ({len(scores_non_donors):,})",
        feature_name="Model Prediction Score"
    )

    # --- ADDING THE VERTICAL LINE ---
    # working_df is already sorted by selected_model_name descending from earlier calculations
    threshold_score_for_vline = None
    annotation_text_vline = ""

    if num_to_target == 0:
        # If no one is targeted, conceptually the threshold is above the max score (e.g., 1.0 or slightly higher for vis)
        # Or, more practically, don't draw a line or draw it at 1.0.
        # Let's draw it at 1.0 if nothing is targeted.
        threshold_score_for_vline = 1.0
        annotation_text_vline = "Target Threshold (None Targeted)"
    elif num_to_target >= TOTAL_POPULATION and TOTAL_POPULATION > 0:
        # If everyone (or more than everyone, due to slider step) is targeted,
        # the threshold is effectively the minimum score in the dataset, or 0.
        # Let's use the minimum score of the last person included.
        threshold_score_for_vline = working_df[selected_model_name].min()
        annotation_text_vline = f"Target Threshold (All Targeted): {threshold_score_for_vline:.2f}"
    elif num_to_target > 0 and num_to_target < TOTAL_POPULATION:
        # The (num_to_target)-th person (0-indexed) has the threshold score
        # This is the score of the LAST person included in the 'num_to_target' group
        threshold_score_for_vline = working_df[selected_model_name].iloc[num_to_target - 1]
        annotation_text_vline = f"Target Threshold: {threshold_score_for_vline:.3f}" # More precision for score

    if threshold_score_for_vline is not None:
        fig_dist.add_vline(
            x=threshold_score_for_vline,
            line_width=2,
            line_dash="dash",
            line_color="red",
            annotation_text=annotation_text_vline,
            annotation_position="top left" # Adjust as needed (top right, bottom left, etc.)
        )
    # --- END OF ADDING VERTICAL LINE ---

    st.plotly_chart(fig_dist, use_container_width=True)
    # Add a small explanation for the red line
    if threshold_score_for_vline is not None:
        st.caption(f"The vertical red line indicates the model score threshold ({threshold_score_for_vline:.3f}). Individuals with scores at or above this threshold are included in the 'Number to contact'.")

else:
    st.warning("Not enough data to plot score distributions.")

# --- Display Section 3: Exploring Different Targeting Levels ---
st.subheader("Exploring Net Financial Result vs. Number Contacted")
st.markdown(f"Shows the financial outcome for **{selected_model_name}** if we vary the number of people contacted.")

results_over_n = []
# Dynamic target_levels for the line chart
n_points_line_chart = 20 # Number of points on the line chart
min_target_level_chart = max(1, int(TOTAL_POPULATION * 0.01)) # Start from at least 1% or 1 person
if TOTAL_POPULATION <= n_points_line_chart: # If population is small, use all integer steps
    target_levels = np.arange(min_target_level_chart, TOTAL_POPULATION + 1) # Ensure it starts from min_target_level_chart
    if not target_levels.size: # If TOTAL_POPULATION < min_target_level_chart, make it just TOTAL_POPULATION
        target_levels = np.array([TOTAL_POPULATION]) if TOTAL_POPULATION > 0 else np.array([1])
else:
    target_levels = np.unique(np.linspace(min_target_level_chart, TOTAL_POPULATION, num=n_points_line_chart, dtype=int))

if not np.any(target_levels): # If target_levels is empty or all zeros
    st.warning("Cannot generate target levels for the net result chart with current population settings.")
    plot_df_net_result = pd.DataFrame() # Ensure it's defined
else:
    for n_contact in target_levels:
        if n_contact == 0 and TOTAL_POPULATION > 0: continue
        if n_contact > TOTAL_POPULATION : continue # Should be handled by linspace, but good check

        # Ensure working_df is defined and sorted correctly from earlier in your script
        # working_df should be sim_data_df sorted by selected_model_name
        _targeted_df = working_df.head(n_contact)
        _tp = _targeted_df['y_true'].sum()
        _cost = n_contact * cost_per_letter
        _donations = _tp * avg_donation_amount
        results_over_n.append({
            'N Contacted': n_contact,
            'Net Result (€)': _donations - _cost,
            'Donors Found': _tp,
            '% Contacted': (n_contact / TOTAL_POPULATION) * 100 if TOTAL_POPULATION > 0 else 0
        })
    plot_df_net_result = pd.DataFrame(results_over_n)

if not plot_df_net_result.empty:
    fig_net_result = px.line(plot_df_net_result, x='N Contacted', y='Net Result (€)', markers=True,
                  hover_data={'Donors Found': True, '% Contacted': ':.1f%'},
                  labels={'N Contacted': f'Number Contacted (out of {TOTAL_POPULATION:,})'},
                  title=f"Net Financial Result vs. Number Contacted ({selected_model_name})")
    fig_net_result.add_hline(y=net_gain_loss_all, line_dash="dot",
                  annotation_text=f"Baseline (Contact All): {net_gain_loss_all:,.0f} €",
                  annotation_position="bottom right")

    # --- CORRECTED VLINE ---
    # The x-value for the vline should be num_to_target (the absolute number from the slider)
    # The annotation text can still show the percentage for clarity
    current_targeting_absolute = num_to_target # This is the value from your slider
    annotation_text_vline = f"Current Target: {num_to_target:,} ({num_to_target/TOTAL_POPULATION:.0%})" if TOTAL_POPULATION > 0 else f"Current Target: {num_to_target:,}"

    fig_net_result.add_vline(
        x=current_targeting_absolute, # Use the absolute number of contacts
        line_width=2,
        line_dash="dash",
        line_color="red",
        annotation_text=annotation_text_vline,
        annotation_position="top right" # You might need to adjust this position
    )
    st.plotly_chart(fig_net_result, use_container_width=True)

else:
    st.warning("Could not generate data for the 'Net Financial Result vs. Number Contacted' chart.")


# ROC, PR, Rank-based plots
num_rank_bins_sim = st.select_slider( # select_slider is better for few discrete choices
            "Number of Rank Bins for Curves:",
            options=[5, 10, 20], value=10, key="rank_bins_sim_curves"
        )
if len(eval_df_sim['y_true'].unique()) > 1: # Check if calculable
    col_curves1, col_curves2 = st.columns(2)
    with col_curves1:
        fig_roc = plot_roc_curve_interactive(eval_df_sim['y_true'], eval_df_sim['y_pred_prob'], title=f"ROC Curve")
        st.plotly_chart(fig_roc, use_container_width=True)

        fig_pr = plot_pr_curve_interactive(eval_df_sim['y_true'], eval_df_sim['y_pred_prob'], title=f"PR Curve")
        st.plotly_chart(fig_pr, use_container_width=True)

    with col_curves2:

        rank_metrics_df_sim = calculate_rank_metrics(
            df=eval_df_sim,
            y_true_col='y_true',
            y_pred_col='y_pred_prob',
            num_bins=num_rank_bins_sim
        )
        if not rank_metrics_df_sim.empty:
            current_targeting_percentage_on_plot = (num_to_target / TOTAL_POPULATION) * 100 if TOTAL_POPULATION > 0 else 0
            annotation_text = f"Targeting Top {num_to_target / TOTAL_POPULATION:.0%}" if TOTAL_POPULATION > 0 else "Targeting 0%"

            fig_rank_pr = plot_rank_metrics_interactive(rank_metrics_df_sim, title=f"Precision & Recall vs. Top %")
            fig_rank_pr.add_vline(x=current_targeting_percentage_on_plot, line_width=2, line_dash="dash", line_color="red",
                                  annotation_text=annotation_text, annotation_position="bottom right")
            st.plotly_chart(fig_rank_pr, use_container_width=True)

            fig_lift = plot_lift_chart_interactive(rank_metrics_df_sim, title=f"Lift Chart")
            fig_lift.add_vline(x=current_targeting_percentage_on_plot, line_width=2, line_dash="dash", line_color="red",
                               annotation_text=annotation_text, annotation_position="top right") # Adjusted position
            st.plotly_chart(fig_lift, use_container_width=True)
        else:
            st.warning("Could not calculate rank metrics for these curves (e.g., too few data points or bins).")
else:
    st.warning("Overall performance curves (ROC, PR, etc.) cannot be generated as there's only one class in the 'Actual Donors' data for the current scenario settings.")


st.markdown("---")


st.markdown("---")
st.subheader("How to Interpret This Simulator:")
st.markdown("""
- **Scenario Settings:** Adjust `Total Potential Donors` and `Actual Donors if All Contacted` in the sidebar to match different campaign sizes or baseline conversion rates.
- **Model Quality (AUC):** In the sidebar, select different `Model Simulation` types. Notice how the "Selected Model AUC" changes. Generally, higher AUC models (like "Excellent Model") will show better separation in the "Score Distribution" plot and lead to better financial outcomes or efficiency.
- **"Perfect Model (AUC=1)":** Use this to discuss **overfitting**. While it looks ideal here (perfect separation), explain that if a model performs *this* perfectly on data it was *trained* on, it might have just memorized that data and could fail on new, unseen data. This highlights the need for proper testing on holdout sets.
- **Targeting Slider (`Number to contact`):** This is the key interactive element. As you move it:
    - **Observe Changes in Section 1:**
        - **Precision vs. Recall:** If you target few people (slider to the left), precision might be high (for good models) but recall (total donors found) will be low. As you target more (slider to the right), recall increases, but precision often drops.
        - **Confusion Matrix:** See how TP, FP, FN, TN numbers change. This directly shows who you're correctly/incorrectly targeting or missing.
        - **Financials:** The "Net Financial Result" is crucial. Targeting too few might mean low net gain because you miss donors. Targeting too many increases costs and might also reduce net gain if precision drops significantly.
    - **Observe Highlight on Overall Curves (Section 2):** The red dashed line on the "Precision & Recall vs. Top %" and "Lift Chart" shows where your current `Number to contact` falls on the overall model performance curves. This connects your specific targeting decision to the model's general capabilities.
- **Score Distribution Plot (Section 2):** This plot visualizes how well the selected model distinguishes between actual donors and non-donors based on their scores. Better models show more separation between the two distributions.
- **Net Financial Result vs. Number Contacted (Section 3):** This line chart summarizes the financial impact across all possible targeting levels for the selected model. It often reveals an optimal range for `Number to contact` that maximizes net financial result. Compare this chart for different model qualities.

This hands-on, scenario-based approach aims to make abstract machine learning concepts (like AUC, precision, recall) tangible and demonstrate their direct relevance to NGO operations and decision-making.
""")