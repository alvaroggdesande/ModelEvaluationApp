import streamlit as st
import pandas as pd
import numpy as np
import warnings
from typing import Optional
from scipy import stats
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, \
                            precision_score, recall_score, f1_score, confusion_matrix
from sklearn.calibration import calibration_curve
# Import your custom calculation functions
from modules.prediction_holdout_analysis import (custom_sort_key,
                                                                    calculate_rank_metrics, 
                                                                    rank_performance_by_group, 
                                                                    _create_group_column)
from modules.data_drift_analysis import (calculate_psi_with_nan, calculate_categorical_psi_with_nan, calculate_ks_test)

# --- Wrappers or direct use of your functions ---

# Example: Get metrics at a specific threshold
def get_metrics_at_threshold(y_true, y_pred_prob, threshold):
    y_pred_binary = (np.array(y_pred_prob) >= threshold).astype(int)
    precision = precision_score(y_true, y_pred_binary, zero_division=0)
    recall = recall_score(y_true, y_pred_binary, zero_division=0)
    f1 = f1_score(y_true, y_pred_binary, zero_division=0)
    cm = confusion_matrix(y_true, y_pred_binary)
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
    return {
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn
    }

# Expose your rank metric function (can be called directly)
calculate_rank_metrics = calculate_rank_metrics # Already imported

# Expose your subgroup rank metric function
rank_performance_by_group = rank_performance_by_group # Already imported

# Expose your drift calculation functions if needed for on-the-fly calc
calculate_psi = calculate_psi_with_nan # Rename for clarity maybe
calculate_cateogircal_psi = calculate_categorical_psi_with_nan
calculate_ks = calculate_ks_test

# Function for calibration data
def get_calibration_data(y_true, y_pred_prob, n_bins=10, strategy='uniform'):
    prob_true, prob_pred = calibration_curve(y_true, y_pred_prob, n_bins=n_bins, strategy=strategy)
    return prob_pred, prob_true


def calculate_descriptive_stats(scores: pd.Series) -> Optional[pd.DataFrame]:
    """
    Calculates descriptive statistics, skew, kurtosis, diversity metrics,
    and normality test results for prediction scores.

    Args:
        scores: A pandas Series containing the prediction scores.

    Returns:
        A pandas DataFrame containing the calculated statistics, or None if input is invalid.
    """
    if not isinstance(scores, pd.Series):
        scores = pd.Series(scores) # Convert if NumPy array or list

    scores = pd.to_numeric(scores, errors='coerce').dropna() # Ensure numeric and remove NaNs

    if scores.empty:
        warnings.warn("Cannot calculate descriptive stats: Input scores are empty or all NaN.")
        return None

    # Standard descriptive statistics
    desc = scores.describe()
    skew = scores.skew()
    kurt = scores.kurtosis()
    unique_count = scores.nunique()
    total_count = scores.count() # Use count after dropping NaNs

    # Calculate diversity metrics only if there are scores
    gini = np.nan
    entropy_val = np.nan
    if total_count > 0:
        # Frequency distribution for Gini/Entropy
        counts = scores.value_counts(normalize=False) # Get counts directly
        p = counts / total_count # Normalize to get probabilities
        if len(p) > 1 : # Gini requires at least 2 unique values
             gini = 1 - np.sum(p**2)
        # Entropy requires counts, not probabilities for scipy.stats.entropy
        if len(counts) > 0:
             entropy_val = stats.entropy(counts.values, base=2) # Use counts.values

    # Normality test (Shapiro-Wilk) - sample if too large
    shapiro_stat, shapiro_p = np.nan, np.nan
    sample_size_for_shapiro = min(len(scores), 4999) # Shapiro has limitations around N=5000
    if 3 <= sample_size_for_shapiro <= 4999: # Shapiro requires N between 3 and 5000
        try:
            # Run on a sample if the dataset is large
            scores_sample = scores.sample(n=sample_size_for_shapiro, random_state=42) if len(scores) > 5000 else scores
            shapiro_stat, shapiro_p = stats.shapiro(scores_sample)
        except Exception as e:
            warnings.warn(f"Shapiro-Wilk test failed: {e}")
            # Keep NaN values

    additional_stats = {
        'unique_count': unique_count,
        'skewness': skew,
        'kurtosis': kurt,
        'gini_coefficient': gini,
        'entropy_bits': entropy_val, # Rename for clarity
        'shapiro_w': shapiro_stat,
        'shapiro_p': shapiro_p
    }

    # Combine and format as DataFrame
    stats_df = pd.concat([desc, pd.Series(additional_stats, name='value')])
    stats_df = stats_df.reset_index().rename(columns={'index': 'Metric', 0: 'Value'}) # Make it a 2-column DF

    return stats_df

# Page 3. Subgroup metrics
def calculate_subgroup_means(
    df: pd.DataFrame,
    feature: str,
    target_col: str,
    pred_col: str,
    numeric_q: Optional[int] = 5, # Use Optional for clarity
    cat_top_n: Optional[int] = 7, # Use Optional for clarity
) -> pd.DataFrame:
    """
    Calculates mean target, mean prediction, and count across subgroups defined by a feature.

    Creates bins for numeric features or groups top categories for categorical ones.
    Handles NaN values as a separate group and sorts the results.

    Args:
        df: DataFrame containing the data.
        feature: Feature column name to group by.
        target_col: True outcome column name.
        pred_col: Prediction score/value column name.
        numeric_q: Number of quantile bins for numeric features. Use None if feature is not numeric or no binning desired.
        cat_top_n: Number of top categories for categorical features ('Other' for the rest). Use None if not categorical.

    Returns:
        pd.DataFrame: Aggregated results per group, sorted appropriately.
                      Columns: 'group', 'mean_prediction', 'mean_target', 'count'.
    """
    if feature not in df.columns:
        raise ValueError(f"Feature '{feature}' not found in DataFrame.")
    if target_col not in df.columns:
         raise ValueError(f"Target column '{target_col}' not found.")
    if pred_col not in df.columns:
         raise ValueError(f"Prediction column '{pred_col}' not found.")

    temp_df = df[[feature, target_col, pred_col]].copy() # Work with necessary columns

    # --- Determine binning parameters based on feature type ---
    is_numeric_feat = pd.api.types.is_numeric_dtype(temp_df[feature])
    q_to_use = numeric_q if is_numeric_feat else None
    n_to_use = cat_top_n if not is_numeric_feat else None

    # --- Create group column ---
    try:
        temp_df['group'] = _create_group_column(
            temp_df, feature, numeric_q=q_to_use, cat_top_n=n_to_use, include_nan=True
        )
    except Exception as e:
         raise ValueError(f"Failed to create groups for feature '{feature}': {e}")

    # Ensure target and prediction are numeric for aggregation
    temp_df[target_col] = pd.to_numeric(temp_df[target_col], errors='coerce')
    temp_df[pred_col] = pd.to_numeric(temp_df[pred_col], errors='coerce')

    # --- Aggregate ---
    analysis_df = temp_df.groupby('group', observed=False).agg(
        mean_prediction=(pred_col, "mean"),
        mean_target=(target_col, "mean"),
        count=(feature, "size") # Count rows in the original feature column for this group
    ).reset_index()

    # Fill potential NaN aggregates if a group ended up empty or had all NaNs in target/pred
    analysis_df['mean_prediction'] = analysis_df['mean_prediction'].fillna(0)
    analysis_df['mean_target'] = analysis_df['mean_target'].fillna(0)


    # --- Sort using the custom key ---
    try:
        # Ensure group column is suitable type for sorting key (e.g., category or object)
        analysis_df['sort_key'] = analysis_df['group'].astype(object).apply(custom_sort_key)
        analysis_df = analysis_df.sort_values('sort_key').drop(columns='sort_key')
    except Exception as e:
         warnings.warn(f"Could not sort groups for feature '{feature}', returning unsorted. Error: {e}")

    return analysis_df.reset_index(drop=True) # Ensure index is clean