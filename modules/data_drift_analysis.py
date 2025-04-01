# datadrift_analysis.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats as sp_stats
from typing import Optional, Union, List, Tuple, Dict, Any
import math
import warnings
import re

# --- Constants ---
EPSILON = 1e-10 # Small value to avoid division by zero
DEFAULT_PSI_BUCKETS = 10
DEFAULT_FIG_SIZE_WIDE = (12, 7)
DEFAULT_FIG_SIZE_GRID = (15, 10)

# --- PSI Calculation Functions (Including NaN) ---

def calculate_psi_with_nan(
    expected_series: pd.Series,
    actual_series: pd.Series,
    buckets: int = DEFAULT_PSI_BUCKETS
) -> float:
    """
    Calculate the PSI (Population Stability Index) for a numerical feature,
    INCLUDING missing values as a separate bin.

    Args:
        expected_series: 1D pandas Series of training/reference values (can contain NaNs).
        actual_series: 1D pandas Series of prediction/current values (can contain NaNs).
        buckets: Number of bins for non-NaN values.

    Returns:
        The calculated PSI value. Returns 0.0 if inputs are identical including NaNs.
    """

    # --- 1. Calculate NaN percentages ---
    total_expected_count = len(expected_series)
    total_actual_count = len(actual_series)

    if total_expected_count == 0 or total_actual_count == 0:
        warnings.warn("One or both series are empty, returning PSI=nan.")
        return np.nan # Cannot calculate PSI if one set is empty

    nan_expected_count = expected_series.isnull().sum()
    nan_actual_count = actual_series.isnull().sum()

    nan_perc_expected = nan_expected_count / total_expected_count
    nan_perc_actual = nan_actual_count / total_actual_count

    # --- 2. Define bins based on non-NaN expected data ---
    expected_data_nonan = expected_series.dropna()
    actual_data_nonan = actual_series.dropna() # Also needed later

    psi_value = 0.0

    if not expected_data_nonan.empty:
        min_val = expected_data_nonan.min() # Use actual min/max
        max_val = expected_data_nonan.max()

        if np.isclose(min_val, max_val): # Handle constant non-NaN data
            breakpoints = [min_val, max_val + EPSILON]
            actual_buckets = 1
        else:
            # Use linspace for even bins; consider pd.qcut for quantile bins if preferred
            breakpoints = np.linspace(min_val, max_val, buckets + 1)
            actual_buckets = buckets
        # Ensure the last breakpoint includes the max value robustly
        breakpoints[-1] = max_val + EPSILON

        # --- 3. Calculate PSI for non-NaN bins ---
        for i in range(actual_buckets):
            # Count observations IN THE BIN (from non-NaN data)
            expected_bin_count = ((expected_data_nonan >= breakpoints[i]) & (expected_data_nonan < breakpoints[i+1])).sum()
            actual_bin_count = ((actual_data_nonan >= breakpoints[i]) & (actual_data_nonan < breakpoints[i+1])).sum()

            # Calculate percentage RELATIVE TO TOTAL (including NaNs)
            expected_perc_bin = expected_bin_count / total_expected_count
            actual_perc_bin = actual_bin_count / total_actual_count

            # Add PSI component for this bin
            # Handle cases where one percentage is zero
            term = (expected_perc_bin - actual_perc_bin) * np.log((expected_perc_bin + EPSILON) / (actual_perc_bin + EPSILON))
            psi_value += term

    else:
        warnings.warn(f"Feature '{expected_series.name}' has no non-NaN values in expected data. PSI based only on NaN%.")
        # If expected is all NaN, psi calculation below will handle it

    # --- 4. Add PSI component for the NaN bin ---
    if not np.isclose(nan_perc_expected, nan_perc_actual):
         psi_value += (nan_perc_expected - nan_perc_actual) * np.log((nan_perc_expected + EPSILON) / (nan_perc_actual + EPSILON))
    # else: NaNs are stable, add 0

    # Handle potential negative PSI due to floating point issues near zero
    return max(0.0, psi_value)


def calculate_categorical_psi_with_nan(
    train_series: pd.Series,
    pred_series: pd.Series
) -> float:
    """
    Calculate PSI for a categorical feature, INCLUDING missing values
    as a distinct category.

    Args:
        train_series: pandas Series of training/reference categorical values.
        pred_series: pandas Series of prediction/current categorical values.

    Returns:
        The calculated PSI value.
    """
    total_expected_count = len(train_series)
    total_actual_count = len(pred_series)

    if total_expected_count == 0 or total_actual_count == 0:
        warnings.warn("One or both series are empty, returning PSI=nan.")
        return np.nan

    # Calculate distributions INCLUDING NaNs, normalized by total count
    train_dist = train_series.value_counts(normalize=True, dropna=False)
    pred_dist = pred_series.value_counts(normalize=True, dropna=False)

    psi_value = 0.0
    all_categories = set(train_dist.index).union(set(pred_dist.index))

    for cat in all_categories:
        expected_perc = train_dist.get(cat, 0)
        actual_perc = pred_dist.get(cat, 0)

        if not np.isclose(expected_perc, actual_perc):
            term = (expected_perc - actual_perc) * np.log((expected_perc + EPSILON) / (actual_perc + EPSILON))
            psi_value += term

    return max(0.0, psi_value) # Ensure non-negative

# --- KS Test Helper ---

def calculate_ks_test(
    expected_series: pd.Series,
    actual_series: pd.Series
) -> Tuple[Optional[float], Optional[float]]:
    """
    Calculates the 2-sample Kolmogorov-Smirnov test statistic and p-value
    ONLY for numeric features after attempting numeric conversion and dropping NaNs.

    Args:
        expected_series: Series from the reference dataset.
        actual_series: Series from the current dataset.

    Returns:
        A tuple (ks_statistic, p_value). Returns (np.nan, np.nan) if the
        feature is not numeric or if KS test cannot be performed.
    """
    ks_stat, ks_pvalue = np.nan, np.nan # Default

    # --- Check 1: Is the feature fundamentally numeric? ---
    # Check the original dtype before dropping NaNs, as dropna might change it
    if not pd.api.types.is_numeric_dtype(expected_series.dtype) and \
       not pd.api.types.is_numeric_dtype(actual_series.dtype):
         # If neither original series looks numeric, skip KS test
         # Allow if one was numeric and the other object that might be convertible
         # A stricter check might be:
         # if not pd.api.types.is_numeric_dtype(expected_series.dtype):
         #     return ks_stat, ks_pvalue # Skip if reference isn't numeric
         pass # Proceed cautiously, conversion attempt below

    # --- Check 2: Attempt conversion and drop remaining NaNs/non-numerics ---
    try:
        # Convert to numeric, coercing errors to NaN
        train_data_numeric = pd.to_numeric(expected_series, errors='coerce').dropna()
        pred_data_numeric = pd.to_numeric(actual_series, errors='coerce').dropna()
    except Exception as e:
        warnings.warn(f"Error during numeric conversion for feature '{expected_series.name}'. Skipping KS. Error: {e}")
        return ks_stat, ks_pvalue # Return NaN defaults


    # --- Check 3: Proceed only if data remains after conversion/dropping ---
    if not train_data_numeric.empty and not pred_data_numeric.empty:
        # Check for constant data which ks_2samp dislikes
        if len(np.unique(train_data_numeric)) > 1 and len(np.unique(pred_data_numeric)) > 1:
             try:
                 # Ensure data is float for ks_2samp if it became integer
                 ks_stat, ks_pvalue = sp_stats.ks_2samp(
                     train_data_numeric.astype(float),
                     pred_data_numeric.astype(float)
                 )
             except Exception as e:
                  warnings.warn(f"KS test failed for feature '{expected_series.name}'. Error: {e}")
                  # Keep defaults ks_stat, ks_pvalue = np.nan, np.nan
        else:
             # Handle cases where one or both are constant after cleaning
             # You could add more sophisticated checks here if needed
             if np.allclose(train_data_numeric.mean(), pred_data_numeric.mean()):
                 ks_stat, ks_pvalue = 0.0, 1.0 # No difference detectable by KS if constants are same
             else:
                 ks_stat, ks_pvalue = 1.0, 0.0 # Max difference if constants differ (KS not ideal)
    # else: If one/both are empty after conversion/dropping, return NaN defaults

    return ks_stat, ks_pvalue

# --- Main Drift Calculation Function ---

def calculate_drift_metrics(
    training_df: pd.DataFrame,
    prediction_df: pd.DataFrame,
    features_to_analyze: List[str],
    shap_importance_df: Optional[pd.DataFrame] = None,
    psi_buckets: int = DEFAULT_PSI_BUCKETS
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Calculates PSI (incl. NaN) and KS-test metrics for specified features
    between a training and prediction dataframe. Optionally merges with SHAP importance.

    Args:
        training_df: DataFrame representing the training/reference distribution.
        prediction_df: DataFrame representing the prediction/current distribution.
        features_to_analyze: List of column names (features) to analyze for drift.
        shap_importance_df: Optional DataFrame with 'feature' and 'feature_importance' columns.
                            Feature importance should be a non-negative numeric value.
        psi_buckets: Number of bins for numerical PSI calculation.

    Returns:
        A tuple containing:
        - drift_df: DataFrame with 'feature', 'psi', 'ks_stat', 'ks_pvalue'.
        - drift_importance_df: drift_df merged with shap_importance_df (if provided),
                               sorted by importance, otherwise None.
    """
    datadrift_results = {}

    for feature in features_to_analyze:
        psi_value, ks_stat, ks_pvalue = np.nan, np.nan, np.nan # Defaults

        if feature not in training_df.columns:
            warnings.warn(f"Feature '{feature}' not found in training_df. Skipping.")
            continue
        if feature not in prediction_df.columns:
            warnings.warn(f"Feature '{feature}' not found in prediction_df. Skipping.")
            continue

        train_series = training_df[feature]
        pred_series = prediction_df[feature]

        # --- Compute PSI (including NaNs) ---
        try:
            if pd.api.types.is_numeric_dtype(train_series):
                 psi_value = calculate_psi_with_nan(train_series, pred_series, buckets=psi_buckets)
            elif pd.api.types.is_object_dtype(train_series) or pd.api.types.is_categorical_dtype(train_series):
                 psi_value = calculate_categorical_psi_with_nan(train_series, pred_series)
            else:
                 warnings.warn(f"Skipping PSI for feature '{feature}' due to unsupported data type: {train_series.dtype}")
        except Exception as e:
             warnings.warn(f"Error calculating PSI for feature '{feature}': {e}")

        # --- Compute KS Test (on non-NaNs) ---
        ks_stat, ks_pvalue = calculate_ks_test(train_series, pred_series)

        datadrift_results[feature] = {"psi": psi_value, "ks_stat": ks_stat, "ks_pvalue": ks_pvalue}

    drift_df = pd.DataFrame.from_dict(datadrift_results, orient='index')
    drift_df = drift_df.reset_index().rename(columns={'index': 'feature'})
    drift_df = drift_df.dropna(subset=['psi']) # Remove rows where PSI calculation failed entirely

    # --- Merge with SHAP Importance ---
    drift_importance_df = None
    if shap_importance_df is not None:
        if not all(col in shap_importance_df.columns for col in ['feature', 'feature_importance']):
            warnings.warn("shap_importance_df must contain 'feature' and 'feature_importance' columns. Skipping merge.")
        elif not pd.api.types.is_numeric_dtype(shap_importance_df['feature_importance']):
             warnings.warn("'feature_importance' column must be numeric. Skipping merge.")
        else:
            # Ensure importance is non-negative
            shap_importance_df['feature_importance'] = shap_importance_df['feature_importance'].abs()
            drift_importance_df = pd.merge(drift_df, shap_importance_df, on='feature', how='left')
            # Fill missing importance with 0 and sort
            drift_importance_df['feature_importance'] = drift_importance_df['feature_importance'].fillna(0)
            drift_importance_df = drift_importance_df.sort_values('feature_importance', ascending=False).reset_index(drop=True)

    return drift_df, drift_importance_df


# --- Visualization Functions ---

def plot_distribution_drift(
    training_series: pd.Series,
    prediction_series: pd.Series,
    feature_name: str,
    psi_value: Optional[float],
    ks_pvalue: Optional[float],
    ax: plt.Axes
):
    """Plots overlaid distributions for a single feature on a given matplotlib Axes."""
    train_data_nonan = training_series.dropna()
    pred_data_nonan = prediction_series.dropna()
    nan_perc_train = training_series.isnull().mean() * 100
    nan_perc_pred = prediction_series.isnull().mean() * 100

    plot_title = f"{feature_name}\n"
    if psi_value is not None:
        plot_title += f"PSI: {psi_value:.3f} | "
    if ks_pvalue is not None:
        plot_title += f"KS p-val: {ks_pvalue:.3f}\n"
    plot_title += f"NaN %: {nan_perc_train:.1f}% -> {nan_perc_pred:.1f}%"


    if pd.api.types.is_numeric_dtype(training_series):
        if not train_data_nonan.empty:
            sns.kdeplot(train_data_nonan, ax=ax, label='Train (Expected)', fill=True, warn_singular=False)
        if not pred_data_nonan.empty:
            sns.kdeplot(pred_data_nonan, ax=ax, label='Predict (Actual)', fill=True, warn_singular=False)
        ax.set_xlabel(feature_name)
        ax.set_ylabel("Density")

    elif pd.api.types.is_object_dtype(training_series) or pd.api.types.is_categorical_dtype(training_series):
        # For categorical, prepare data for grouped bar chart (top N + Other maybe needed for clarity?)
        # Simple version: plot all categories found
        train_counts = train_series.value_counts(normalize=True, dropna=False).rename('Train')
        pred_counts = pred_series.value_counts(normalize=True, dropna=False).rename('Predict')
        df_plot = pd.concat([train_counts, pred_counts], axis=1).fillna(0)
        df_plot.plot(kind='bar', ax=ax) # Plot all categories for direct comparison
        ax.set_ylabel("Proportion")
        ax.tick_params(axis='x', rotation=45, ha='right')
    else:
        ax.text(0.5, 0.5, f"Unsupported dtype: {training_series.dtype}", ha='center', va='center', transform=ax.transAxes)

    ax.set_title(plot_title, fontsize=10)
    ax.legend()
    ax.grid(True, alpha=0.3)


def plot_top_feature_distributions(
    training_df: pd.DataFrame,
    prediction_df: pd.DataFrame,
    drift_df: pd.DataFrame, # Needs 'feature', 'psi', 'ks_pvalue' columns
    n_features_to_plot: int = 12,
    sort_by: str = 'psi', # or 'feature_importance' if available and merged
    figsize: Tuple[int, int] = DEFAULT_FIG_SIZE_GRID
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Plots the distribution drift for the top N features based on PSI or Importance.

    Args:
        training_df: DataFrame representing the training/reference distribution.
        prediction_df: DataFrame representing the prediction/current distribution.
        drift_df: DataFrame containing drift metrics ('feature', 'psi', 'ks_pvalue', optionally 'feature_importance').
        n_features_to_plot: Maximum number of features to plot.
        sort_by: Column in drift_df to sort by ('psi' or 'feature_importance').
        figsize: Overall figure size.

    Returns:
        A tuple containing:
        - fig: The matplotlib Figure object.
        - axes: A numpy array of the matplotlib Axes objects.
    """
    if sort_by not in drift_df.columns:
         warnings.warn(f"Sort column '{sort_by}' not found in drift_df. Defaulting to 'psi'.")
         sort_by = 'psi'
         if 'psi' not in drift_df.columns:
              raise ValueError("drift_df must contain at least 'feature' and 'psi' columns.")

    # Sort and select top N features
    top_features_df = drift_df.sort_values(sort_by, ascending=False, key=abs).head(n_features_to_plot)
    features_to_plot = top_features_df['feature'].tolist()

    if not features_to_plot:
        warnings.warn("No features selected for plotting.")
        return plt.figure(figsize=figsize), np.array([]) # Return empty figure/axes

    n_cols = min(4, len(features_to_plot)) # Max 4 columns
    n_rows = math.ceil(len(features_to_plot) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes_flat = axes.flatten()

    for i, feature in enumerate(features_to_plot):
        if i >= len(axes_flat): break # Should not happen with ceil logic, but safety check

        ax = axes_flat[i]
        metrics = top_features_df[top_features_df['feature'] == feature].iloc[0]
        plot_distribution_drift(
            training_df[feature],
            prediction_df[feature],
            feature,
            metrics.get('psi'), # Use .get for safety
            metrics.get('ks_pvalue'),
            ax
        )

    # Hide any unused subplots
    for j in range(i + 1, len(axes_flat)):
        fig.delaxes(axes_flat[j])

    plt.tight_layout()
    return fig, axes


def plot_drift_summary(
    drift_importance_df: pd.DataFrame,
    psi_threshold_minor: float = 0.1,
    psi_threshold_major: float = 0.25,
    importance_threshold_quantile: float = 0.75, # Label features in top 25% importance
    labeling_psi_threshold: float = 0.15, # Also label if PSI exceeds this, regardless of importance
    ks_alpha: float = 0.05,
    figsize: Tuple[int, int] = DEFAULT_FIG_SIZE_WIDE
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plots PSI vs. Feature Importance, highlighting significant drift.

    Args:
        drift_importance_df: DataFrame with 'feature', 'psi', 'feature_importance', 'ks_pvalue'.
        psi_threshold_minor: PSI threshold line for minor drift.
        psi_threshold_major: PSI threshold line for major drift.
        importance_threshold_quantile: Quantile for defining 'high importance' for labeling.
        labeling_psi_threshold: PSI value above which features are always labeled.
        ks_alpha: Significance level for KS p-value highlighting.
        figsize: Figure size.

    Returns:
        A tuple containing:
        - fig: The matplotlib Figure object.
        - ax: The matplotlib Axes object.
    """
    if not all(col in drift_importance_df.columns for col in ['feature', 'psi', 'feature_importance', 'ks_pvalue']):
        raise ValueError("drift_importance_df must include 'feature', 'psi', 'feature_importance', 'ks_pvalue'.")

    temp_df = drift_importance_df.copy()
    temp_df['ks_significant'] = temp_df['ks_pvalue'] < ks_alpha
    temp_df['ks_label'] = temp_df['ks_significant'].map({True: f'p < {ks_alpha}', False: f'p >= {ks_alpha}'})

    fig, ax = plt.subplots(figsize=figsize)

    scatter = sns.scatterplot(
        data=temp_df,
        x='feature_importance',
        y='psi',
        hue='ks_label',
        size='feature_importance',
        sizes=(50, 800),
        style='ks_significant',
        palette={'p < 0.05': 'red', f'p >= {ks_alpha}': 'green'},
        alpha=0.7,
        ax=ax
    )

    ax.axhline(psi_threshold_minor, color='orange', linestyle='--', label=f'PSI={psi_threshold_minor} (Minor Drift)')
    ax.axhline(psi_threshold_major, color='red', linestyle='--', label=f'PSI={psi_threshold_major} (Major Drift)')

    ax.set_xlabel("Mean Absolute SHAP Value (Importance)")
    ax.set_ylabel("Population Stability Index (PSI)")
    ax.set_title("Feature Drift (PSI) vs. Feature Importance (SHAP)")
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.5)

    # --- Text Labeling ---
    texts = []
    importance_threshold_value = temp_df['feature_importance'].quantile(importance_threshold_quantile)

    for i, row in temp_df.iterrows():
        # Label if high importance OR high PSI OR significant KS + moderate PSI/Importance? Be selective.
        is_high_importance = row['feature_importance'] >= importance_threshold_value
        is_high_psi = row['psi'] >= labeling_psi_threshold

        if is_high_importance or is_high_psi:
             texts.append(ax.text(row['feature_importance'], row['psi'], f" {row['feature']}", fontsize=9))

    # Use adjustText if available
    if texts:
        try:
            from adjustText import adjust_text
            adjust_text(texts, ax=ax, expand_points=(1.1, 1.2), expand_text=(1.1, 1.2),
                        arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))
        except ImportError:
            warnings.warn("adjustText library not found. Install for better label placement: pip install adjustText")
            pass # Continue without adjusting text

    # --- Legend Handling ---
    handles, labels = ax.get_legend_handles_labels()
    # Filter out size legend items if needed (can be verbose)
    filtered_handles = [h for h, l in zip(handles, labels) if not l.startswith("feature_importance")]
    filtered_labels = [l for l in labels if not l.startswith("feature_importance")]
    ax.legend(handles=filtered_handles, labels=filtered_labels, title="Legend", bbox_to_anchor=(1.03, 1), loc='upper left')

    plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout for legend

    return fig, ax