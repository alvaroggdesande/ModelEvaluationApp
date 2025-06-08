import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score # Added AUCs
from typing import Optional, Union, List, Tuple, Dict, Any # Enhanced typing
import re
import warnings # To handle potential warnings from qcut or plotting

# --- Constants for Plotting ---
DEFAULT_FIG_SIZE = (10, 6)



def _format_bin_edge(value: float) -> str:
    """Formats a bin edge value for better readability."""
    if pd.isna(value):
        return "NaN"
    # If the number is large or has no significant decimal part, format as integer
    if abs(value) >= 1000 or np.isclose(value, round(value)):
        return f"{value:,.0f}" # Add comma separators for thousands
    # Otherwise, use a reasonable number of decimal places (e.g., 2)
    else:
        return f"{value:.2f}"
# --- Helper Function for Consistent Binning ---

def _create_group_column(
    df: pd.DataFrame,
    feature: str,
    numeric_q: Optional[int] = None,
    cat_top_n: Optional[int] = None,
    include_nan: bool = True
) -> pd.Series:
    """
    Creates a grouping Series based on a feature column, handling numeric/categorical types and NaNs.
    """
    if feature not in df.columns:
        raise ValueError(f"Feature '{feature}' not found in DataFrame.")

    col = df[feature]
    # Initialize with object dtype to avoid inheriting restrictive categories early
    group_col = pd.Series(index=df.index, dtype=object)

    # Identify NaNs upfront
    nan_mask = col.isna()
    non_nan_col = col[~nan_mask]

    if pd.api.types.is_numeric_dtype(col):
        if numeric_q is None:
            raise ValueError(f"numeric_q must be specified for numeric feature '{feature}'")
        if not non_nan_col.empty:
            try:
                bins, intervals = pd.qcut(non_nan_col, q=numeric_q, duplicates="drop", retbins=True, labels=False)
                interval_strs = [f"{_format_bin_edge(intervals[i])}-{_format_bin_edge(intervals[i+1])}" for i in range(len(intervals)-1)]
                label_map = {idx: label for idx, label in enumerate(interval_strs)}
                # Use .loc for safer assignment
                group_col.loc[~nan_mask] = bins.map(label_map)
                # Fill any remaining NaNs in the non-NaN section (rare, but safety)
                group_col.loc[~nan_mask] = group_col.loc[~nan_mask].fillna("ErrorBinning")
            except ValueError as e:
                 warnings.warn(f"Could not create {numeric_q} bins for feature '{feature}'. Grouping as single bin. Error: {e}")
                 group_col.loc[~nan_mask] = "AllNumeric"
        else:
            warnings.warn(f"Feature '{feature}' contains only NaN values.")

    elif pd.api.types.is_object_dtype(col) or pd.api.types.is_categorical_dtype(col): # Handles both object and category inputs
        if cat_top_n is None:
             raise ValueError(f"cat_top_n must be specified for categorical/object feature '{feature}'")
        if not non_nan_col.empty:
            cat_counts = non_nan_col.value_counts(dropna=False) # Check counts
            # Ensure cat_top_n is not larger than the number of unique non-NaN values
            actual_top_n = min(cat_top_n, len(cat_counts))
            if actual_top_n < cat_top_n:
                warnings.warn(f"Feature '{feature}' has only {actual_top_n} unique non-NaN values. Using top {actual_top_n}.")

            top_categories = cat_counts.index[:actual_top_n]

            # Convert the non-NaN part to object type BEFORE applying .where
            # This prevents the Categorical error when creating "Other"
            processed_groups = non_nan_col.astype(object).where(
                non_nan_col.isin(top_categories), other="Other"
            )

            # Assign the results back using .loc
            group_col.loc[~nan_mask] = processed_groups

        else:
             warnings.warn(f"Feature '{feature}' contains only NaN values.")

    else:
        raise TypeError(f"Unsupported data type for feature '{feature}': {col.dtype}")

    # Handle NaNs based on include_nan flag
    if include_nan:
        group_col.loc[nan_mask] = "NaN" # Use .loc for assignment
    # No 'else' needed as we started with a full index Series

    # Convert the final result to category for efficiency AFTER processing
    return group_col.astype('category')

# --- Custom Sorting Key for Group Labels ---
def custom_sort_key(label: Union[str, float, int, None, pd.Interval]) -> float:
    """
    Provides a sort key for group labels including numeric intervals (e.g., '-100--50'),
    categories, 'NaN', 'Other', etc. Handles thousands separators.
    """
    if pd.isna(label):
        return float('inf')  # NaNs consistently last

    # --- Handle non-string types first ---
    if isinstance(label, pd.Interval): # Should ideally not happen if labels are strings
        # Fallback if Interval objects are somehow passed
        return label.left if label.left is not None else float('-inf')
    if isinstance(label, (int, float)): # Handle raw numbers
        return label

    # --- Process string labels ---
    label_str = str(label).strip() # Ensure string and remove whitespace

    # --- Handle Special String Cases ---
    if label_str == "NaN":
        return float('inf')
    if label_str == "Other":
        return float('inf') - 1
    if label_str == "AllNumeric":
        return float('-inf') # Sort this first if it occurs
    if label_str == "ErrorBinning":
        return float('inf') - 2 # Before 'Other'

    # --- Attempt to Parse Numeric Interval Strings ---
    # 1. Remove thousands separators (commas) to simplify parsing
    label_str_no_comma = label_str.replace(',', '')

    # 2. Use regex to find the first number (integer or float, potentially negative)
    #    at the beginning of the string (after removing commas).
    #    Regex handles optional minus, digits, optional decimal part.
    match = re.match(r'^(-?\d+\.?\d*)', label_str_no_comma)

    if match:
        try:
            left_edge_str = match.group(1)
            # 3. Convert the extracted number string to float
            return float(left_edge_str)
        except (ValueError, TypeError):
            # If conversion fails for some reason, fall through to treat as category
            pass

    # --- Fallback for Regular Categorical Strings (that didn't parse as intervals) ---
    # Assign a default sort position (e.g., alphabetically after special cases
    # but before numeric intervals). You could adjust this value if needed.
    # Using -inf + 1 places them near the beginning but after 'AllNumeric'.
    # Consider `return hash(label_str)` if a consistent but arbitrary order is okay.
    return float('-inf') + 1
    return float('-inf') + 1 # Default sort position for unmatched strings

# --- Refactored Analysis Functions ---

def analyze_features_subgroups(
    df: pd.DataFrame,
    feature: str,
    target_col: str,
    pred_col: str,
    numeric_q: int = 5, # Default q=5
    cat_top_n: int = 7, # Default top 7
    figsize: Tuple[int, int] = DEFAULT_FIG_SIZE
) -> Tuple[pd.DataFrame, plt.Figure, plt.Axes]:
    """
    Analyzes mean target and prediction values across subgroups defined by a feature.

    Creates bins for numeric features or groups top categories for categorical ones.
    Handles NaN values as a separate group. Plots results on dual axes.

    Args:
        df: DataFrame containing the data.
        feature: Feature column name to group by.
        target_col: True outcome column name.
        pred_col: Prediction score/value column name.
        numeric_q: Number of quantile bins for numeric features.
        cat_top_n: Number of top categories for categorical features ('Other' for the rest).
        figsize: Figure size for the plot.

    Returns:
        A tuple containing:
        - analysis_df: DataFrame with aggregated results per group.
        - fig: The matplotlib Figure object.
        - ax1: The primary matplotlib Axes object (for counts).
    """
    temp_df = df.copy()
    temp_df['group'] = _create_group_column(
        temp_df, feature, numeric_q=numeric_q, cat_top_n=cat_top_n, include_nan=True
    )

    analysis_df = temp_df.groupby('group', observed=False).agg( # observed=False includes empty bins
        mean_prediction=(pred_col, "mean"),
        mean_target=(target_col, "mean"),
        count=(pred_col, "size") # Use size to include NaNs in prediction col if any
    ).reset_index()

    # Sort using the custom key
    analysis_df['sort_key'] = analysis_df['group'].apply(custom_sort_key)
    analysis_df = analysis_df.sort_values('sort_key').drop(columns='sort_key')
    analysis_df = analysis_df.reset_index(drop=True) # Ensure index is 0, 1, 2...

    # Plot
    fig, ax1 = plt.subplots(figsize=figsize)
    fig.suptitle(f"Subgroup Analysis by {feature}") # Use suptitle for main title

    x_labels = analysis_df["group"].astype(str) # Ensure labels are strings
    x_range = range(len(x_labels))

    # Bar plot for count on primary y-axis
    ax1.bar(x_range, analysis_df["count"], alpha=0.3, color="grey", label="Count")
    ax1.set_xlabel(f"{feature} Bins/Categories")
    ax1.set_ylabel("Count", color="grey")
    ax1.tick_params(axis="y", labelcolor="grey")
    ax1.set_xticks(x_range)
    ax1.set_xticklabels(x_labels, rotation=45, ha="right")

    # Secondary axis: line plots for mean values
    ax2 = ax1.twinx()
    ax2.plot(x_range, analysis_df["mean_prediction"], marker="o", color="blue", label="Mean Prediction")
    ax2.plot(x_range, analysis_df["mean_target"], marker="x", color="green", label="Mean Target")
    ax2.set_ylabel("Mean Value", color="black")
    ax2.tick_params(axis="y", labelcolor="black")
    ax2.grid(False) # Turn off grid for secondary axis if desired

    # Combine legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left") # Place legend consistently

    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout slightly for suptitle

    return analysis_df, fig, ax1 # Return df, fig, and primary axis


def plot_residuals_by_feature(
    df: pd.DataFrame,
    feature: str,
    y_true_col: str = "y_true",
    y_pred_col: str = "y_pred",
    numeric_q: int = 10,
    cat_top_n: int = 10,
    figsize: Tuple[int, int] = DEFAULT_FIG_SIZE
) -> Tuple[pd.DataFrame, plt.Figure, plt.Axes]:
    """
    Computes residuals (y_true - y_pred) and plots their mean by subgroups defined by a feature.

    Uses quantile bins for numeric features and top N / 'Other' for categoricals. Includes NaN group.

    Args:
        df: DataFrame with true and predicted values.
        feature: The feature to stratify by.
        y_true_col: Column name for true values.
        y_pred_col: Column name for predicted values.
        numeric_q: Number of quantile bins for numeric features.
        cat_top_n: Number of top categories for categorical features.
        figsize: Figure size for the plot.

    Returns:
        A tuple containing:
        - group_stats: DataFrame with aggregated residual stats per group.
        - fig: The matplotlib Figure object.
        - ax: The matplotlib Axes object.
    """
    temp_df = df.copy()
    # Compute residuals, handle potential non-numeric types if y_true/y_pred are objects
    temp_df['residual'] = pd.to_numeric(temp_df[y_true_col], errors='coerce') - pd.to_numeric(temp_df[y_pred_col], errors='coerce')
    temp_df = temp_df.dropna(subset=['residual']) # Drop rows where residual couldn't be calculated

    temp_df['group'] = _create_group_column(
        temp_df, feature, numeric_q=numeric_q, cat_top_n=cat_top_n, include_nan=True
    )

    # Aggregate residual statistics
    group_stats = temp_df.groupby("group", observed=False)["residual"].agg(["mean", "std", "count"]).reset_index()

    # Sort using the custom key
    group_stats['sort_key'] = group_stats['group'].apply(custom_sort_key)
    group_stats = group_stats.sort_values('sort_key').drop(columns='sort_key')
    group_stats = group_stats.reset_index(drop=True)

    # Plot mean residual with error bars (std)
    fig, ax = plt.subplots(figsize=figsize)
    x_range = range(len(group_stats))
    ax.bar(x_range, group_stats["mean"], yerr=group_stats["std"],
            color=sns.color_palette("viridis", len(group_stats)), # Use seaborn palette
            edgecolor="k", capsize=5, alpha=0.7, label="Mean Residual") # Use plt.bar for better error bar control

    ax.axhline(0, color='red', linestyle='--', lw=1, label='Zero Residual') # Add zero line
    ax.set_xticks(x_range)
    ax.set_xticklabels(group_stats["group"].astype(str), rotation=45, ha="right")
    ax.set_xlabel(f"{feature} Group")
    ax.set_ylabel("Mean Residual (y_true - y_pred)")
    ax.set_title(f"Residual Analysis by {feature}")
    ax.legend()
    plt.tight_layout()

    return group_stats, fig, ax


def calibration_by_feature(
    df: pd.DataFrame,
    feature: str,
    y_true_col: str = "y_true",
    pred_col: str = "PredictionScore",
    numeric_q: int = 10,
    cat_top_n: int = 10,
    figsize: Tuple[int, int] = DEFAULT_FIG_SIZE
) -> Tuple[pd.DataFrame, plt.Figure, plt.Axes]:
    """
    Calculates and plots the calibration point (mean prediction vs. observed rate)
    for each subgroup defined by a feature on a single chart.

    Args:
        df: DataFrame containing the feature, true labels, and predicted scores.
        feature: The feature to stratify by.
        y_true_col: True outcome column name.
        pred_col: Prediction score column name.
        numeric_q: Number of quantile bins if feature is numeric.
        cat_top_n: Number of top categories if feature is categorical.
        figsize: Figure size for the plot.

    Returns:
        A tuple containing:
        - calibration_stats: DataFrame with aggregated calibration stats per group.
        - fig: The matplotlib Figure object.
        - ax: The matplotlib Axes object.
    """
    temp_df = df.copy()
    temp_df['group'] = _create_group_column(
        temp_df, feature, numeric_q=numeric_q, cat_top_n=cat_top_n, include_nan=True
    )

    # Ensure y_true is numeric (0 or 1)
    temp_df[y_true_col] = pd.to_numeric(temp_df[y_true_col], errors='coerce')
    # Ensure pred_col is numeric
    temp_df[pred_col] = pd.to_numeric(temp_df[pred_col], errors='coerce')
    # Drop rows where conversion failed or grouping failed
    temp_df = temp_df.dropna(subset=[y_true_col, pred_col, 'group'])


    calibration_stats = temp_df.groupby("group", observed=False).agg(
        mean_pred=(pred_col, "mean"),
        observed_rate=(y_true_col, "mean"),
        count=(y_true_col, "size")
    ).reset_index()

    # Sort using the custom key - helpful for consistent plotting if needed elsewhere
    calibration_stats['sort_key'] = calibration_stats['group'].apply(custom_sort_key)
    calibration_stats = calibration_stats.sort_values('sort_key').drop(columns='sort_key')
    calibration_stats = calibration_stats.reset_index(drop=True)

    # Plot calibration per group
    fig, ax = plt.subplots(figsize=figsize)
    # Use scatter plot, size points by count?
    sizes = np.maximum(calibration_stats["count"] / calibration_stats["count"].max() * 500, 50) # Scale size
    scatter = ax.scatter(calibration_stats["mean_pred"], calibration_stats["observed_rate"], s=sizes,
                         alpha=0.7, label="Group Calibration Points", zorder=3) # zorder=3 to be on top
    ax.plot([0,1], [0,1], "k--", label="Perfect Calibration", zorder=2)

    # Add labels for groups (optional, can get crowded)
    texts = []
    for i, row in calibration_stats.iterrows():
        texts.append(ax.text(row["mean_pred"], row["observed_rate"], f" {row['group']}", fontsize=8, va='center'))

    # Use adjustText if available and needed
    try:
        from adjustText import adjust_text
        adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))
    except ImportError:
        warnings.warn("adjustText library not found. Install for better label placement: pip install adjustText")
        pass # Continue without adjusting text

    ax.set_xlabel("Mean Predicted Score (within group)")
    ax.set_ylabel("Observed Outcome Rate (within group)")
    ax.set_title(f"Group-Level Calibration by {feature}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    return calibration_stats, fig, ax


def performance_by_group(
    df: pd.DataFrame,
    group_feature: str,
    y_true_col: str = "y_true",
    y_pred_col: str = "PredictionScore",
    thresholds: Union[float, int, List[float]] = 0.5,
    numeric_q: Optional[int] = 10, # Make q optional, use None to skip binning numeric
    cat_top_n: Optional[int] = 10, # Make n optional, use None to skip grouping categorical
) -> pd.DataFrame:
    """
    Computes classification performance (Precision, Recall, F1, ROC AUC, PR AUC)
    for each group defined by a feature, for one or several thresholds.

    If the grouping feature is numeric and 'numeric_q' is provided, bins into quantiles.
    If categorical and 'cat_top_n' is provided, groups into top N + 'Other'.
    Handles NaN values as a separate "NaN" group.

    Args:
        df: DataFrame with true labels and predicted probabilities.
        group_feature: Column name for the grouping feature.
        y_true_col: True outcome column name (binary 0/1).
        y_pred_col: Predicted probability column name.
        thresholds: A single threshold, an int N for N thresholds [0,1], or a list of thresholds.
        numeric_q: Number of quantile bins for numeric features. If None, uses raw numeric values.
        cat_top_n: Number of top categories for cat features. If None, uses raw categories.

    Returns:
        DataFrame: Performance metrics (Precision, Recall, F1, ROC AUC, PR AUC, count) by group and threshold.
                   AUC metrics are calculated once per group (independent of threshold).
    """
    temp_df = df.copy()

    # Create grouping variable using helper function (pass None to skip binning/grouping)
    if numeric_q is not None or cat_top_n is not None:
        temp_df['group'] = _create_group_column(
            temp_df, group_feature, numeric_q=numeric_q, cat_top_n=cat_top_n, include_nan=True
        )
    else:
        # Use raw feature values, ensuring consistent NaN handling
        temp_df['group'] = temp_df[group_feature]
        if temp_df['group'].isna().any():
            temp_df['group'] = temp_df['group'].astype(object).fillna("NaN") # Convert potential NAs to string 'NaN'
        temp_df['group'] = temp_df['group'].astype('category')


    # Determine threshold list
    if isinstance(thresholds, int):
        threshold_list = np.linspace(0, 1, thresholds)
    elif isinstance(thresholds, (float, np.floating)): # Check for numpy float too
        threshold_list = [thresholds]
    elif isinstance(thresholds, list):
        threshold_list = thresholds
    else:
        raise ValueError("Invalid type for thresholds parameter.")

    results = []
    unique_groups = temp_df["group"].unique()

    for grp in unique_groups:
        grp_df = temp_df[temp_df["group"] == grp]
        y_true = grp_df[y_true_col]
        y_pred_prob = grp_df[y_pred_col]
        count = len(grp_df)

        # Calculate AUCs once per group (if enough data)
        roc_auc = np.nan
        pr_auc = np.nan
        if len(y_true.unique()) > 1 and count > 1: # Need both classes and more than 1 sample for AUC
            try:
                roc_auc = roc_auc_score(y_true, y_pred_prob)
                pr_auc = average_precision_score(y_true, y_pred_prob)
            except Exception as e:
                warnings.warn(f"Could not calculate AUC for group '{grp}'. Error: {e}")


        # Calculate threshold-dependent metrics
        for thresh in threshold_list:
            y_pred_bin = (y_pred_prob >= thresh).astype(int)
            precision = precision_score(y_true, y_pred_bin, zero_division=0)
            recall = recall_score(y_true, y_pred_bin, zero_division=0)
            f1 = f1_score(y_true, y_pred_bin, zero_division=0)
            results.append({
                "group": grp, # Use 'group' consistently
                "threshold": thresh,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "roc_auc": roc_auc, # Add AUCs
                "pr_auc": pr_auc,
                "count": count
            })

    results_df = pd.DataFrame(results)
    # Add original feature name back for clarity if needed, though 'group' is the active column
    results_df[group_feature] = results_df['group'] # Add original feature name mapping

    return results_df


def plot_metrics_by_threshold(
     perf_df: pd.DataFrame,
     group_col: str = "group", # Use the 'group' column created
     figsize_factor: float = 5.0
 ) -> Tuple[plt.Figure, List[plt.Axes]]:
    """
    Plots Precision, Recall, and F1-score by threshold for different groups.

    Args:
        perf_df: DataFrame output from `performance_by_group`.
        group_col: The column in perf_df identifying the groups (usually 'group').
        figsize_factor: Height factor per metric subplot.

    Returns:
        A tuple containing:
        - fig: The matplotlib Figure object.
        - axes: A list of the matplotlib Axes objects for each metric subplot.
    """
    metrics = ["precision", "recall", "f1_score"]
    # AUC metrics are constant per group, not plotted against threshold here
    threshold_list = sorted(perf_df["threshold"].unique())
    unique_groups = perf_df[group_col].unique()

    # Sort groups using the custom key
    groups_sorted = sorted(unique_groups, key=custom_sort_key)

    # Create one figure with subplots for each metric.
    fig, axes = plt.subplots(len(metrics), 1,
                             figsize=(DEFAULT_FIG_SIZE[0], figsize_factor * len(metrics)),
                             sharex=True)

    if len(metrics) == 1: # Handle case of single metric
        axes = [axes]

    # Determine a color palette
    palette = sns.color_palette("viridis", len(groups_sorted))
    color_map = {group: color for group, color in zip(groups_sorted, palette)}

    for ax, metric in zip(axes, metrics):
        for grp in groups_sorted:
            grp_data = perf_df[perf_df[group_col] == grp].sort_values("threshold")
            if not grp_data.empty:
                ax.plot(grp_data["threshold"], grp_data[metric], marker=".", # Use smaller marker
                        label=f"{grp}" if len(groups_sorted) < 15 else None, # Avoid huge legend
                        color=color_map[grp])
        ax.set_title(f"{metric.capitalize()} vs. Threshold")
        # ax.set_xlabel("Threshold") # Only needed on bottom plot due to sharex=True
        ax.set_ylabel(metric.capitalize())
        if len(groups_sorted) < 15: # Only show legend if manageable
             ax.legend(title=group_col, bbox_to_anchor=(1.02, 1), loc='upper left')
        ax.grid(True, alpha=0.4)

    axes[-1].set_xlabel("Threshold") # Set x-label only on the last subplot
    plt.tight_layout(rect=[0, 0, 0.88, 1]) # Adjust layout for potential legend

    return fig, axes


def plot_gains_chart(
    df: Optional[pd.DataFrame] = None,
    y_true: Optional[Union[pd.Series, np.ndarray, List]] = None,
    pred: Optional[Union[pd.Series, np.ndarray, List]] = None,
    y_true_col: str = "y_true",
    pred_col: str = "PredictionScore",
    num_bins: int = 10,
    figsize: Tuple[int, int] = DEFAULT_FIG_SIZE
) -> Tuple[pd.DataFrame, plt.Figure, plt.Axes]:
    """
    Plots a Gains chart. Accepts data as DataFrame or separate Series/arrays.

    Sorts data by prediction score (descending), bins into quantiles (deciles by default),
    and calculates cumulative gains.

    Args:
        df: DataFrame containing true labels and prediction scores.
        y_true: True outcome values (if df is not provided).
        pred: Predicted score values (if df is not provided).
        y_true_col: Column name for true outcomes in df.
        pred_col: Column name for predicted scores in df.
        num_bins: Number of bins (quantiles) to split the sorted data.
        figsize: Figure size for the plot.

    Returns:
        A tuple containing:
        - gains_df: DataFrame with gains statistics per bin.
        - fig: The matplotlib Figure object.
        - ax: The matplotlib Axes object.

    Raises:
        ValueError: If neither df nor both y_true and pred are provided, or if columns not found.
    """
    if df is not None:
        if y_true_col not in df.columns or pred_col not in df.columns:
            raise ValueError(f"Columns '{y_true_col}' or '{pred_col}' not found in DataFrame.")
        y_true_series = df[y_true_col]
        pred_series = df[pred_col]
    elif y_true is not None and pred is not None:
        y_true_series = pd.Series(y_true) if not isinstance(y_true, pd.Series) else y_true
        pred_series = pd.Series(pred) if not isinstance(pred, pd.Series) else pred
    else:
        raise ValueError("Provide either a DataFrame or both y_true and pred inputs.")

    if len(y_true_series) != len(pred_series):
        raise ValueError("Inputs y_true and pred must have the same length.")

    data = pd.DataFrame({
        "y_true": pd.to_numeric(y_true_series, errors='coerce'),
        "PredictionScore": pd.to_numeric(pred_series, errors='coerce')
    }).dropna() # Drop rows where conversion failed

    if data.empty:
        raise ValueError("No valid data remaining after handling NaNs in y_true/PredictionScore.")

    data_sorted = data.sort_values(by="PredictionScore", ascending=False).reset_index(drop=True)

    # Create quantile bins based on the sorted index
    try:
         data_sorted["bin_rank"] = pd.qcut(data_sorted.index, q=num_bins, labels=False, duplicates="drop")
         # Create labels like "Decile 1", "Decile 2", ...
         data_sorted["bin_label"] = [f"Bin {i+1}" for i in data_sorted["bin_rank"]]
    except ValueError as e:
         warnings.warn(f"Could not create {num_bins} bins for Gains chart, possibly due to duplicate scores. Error: {e}. Try fewer bins.")
         # Fallback or raise error? For now, return empty df and None fig/ax
         return pd.DataFrame(), None, None # Indicate failure


    gains_df = data_sorted.groupby("bin_label", observed=False).agg(
        total=("y_true", "size"),
        positives=("y_true", "sum")
    ).reset_index()
    # Ensure bins are sorted correctly (Bin 1, Bin 2, ..., Bin 10)
    gains_df["bin_num"] = gains_df["bin_label"].str.extract(r'(\d+)').astype(int)
    gains_df = gains_df.sort_values("bin_num").drop(columns="bin_num")

    gains_df["cumulative_positives"] = gains_df["positives"].cumsum()
    gains_df["cumulative_total"] = gains_df["total"].cumsum()

    total_population = gains_df["total"].sum()
    total_positives = gains_df["positives"].sum()

    gains_df["percentage_population"] = gains_df["cumulative_total"] / total_population if total_population > 0 else 0
    gains_df["percentage_positives"] = gains_df["cumulative_positives"] / total_positives if total_positives > 0 else 0

    # Add baseline percentage for comparison
    gains_df["baseline_percentage"] = gains_df["percentage_population"]

    # Add Lift
    overall_positive_rate = total_positives / total_population if total_population > 0 else 0
    gains_df["positive_rate_bin"] = gains_df["positives"] / gains_df["total"] if total_population > 0 else 0
    gains_df["lift"] = gains_df["positive_rate_bin"] / overall_positive_rate if overall_positive_rate > 0 else 0


    # Plotting
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(gains_df["percentage_population"] * 100, gains_df["percentage_positives"] * 100, marker="o", label="Gains Curve")
    ax.plot([0, 100], [0, 100], "k--", label="Baseline (Random Model)")
    ax.set_xlabel("Cumulative Percentage of Population (%)")
    ax.set_ylabel("Cumulative Percentage of Positives Captured (%)")
    ax.set_title("Gains Chart")
    ax.legend()
    ax.grid(True, alpha=0.4)
    plt.tight_layout()

    return gains_df, fig, ax

def calculate_rank_metrics(
    df: pd.DataFrame,
    y_true_col: str = "y_true",
    y_pred_col: str = "PredictionScore",
    num_bins: int = 10 # e.g., 10 for deciles, 20 for ventiles, 100 for percentiles
) -> pd.DataFrame:
    """
    Calculates rank-based performance metrics (Precision, Recall, Lift) across bins
    (e.g., deciles).

    Args:
        df: DataFrame with true labels and predicted probabilities.
        y_true_col: True outcome column name (binary 0/1).
        y_pred_col: Predicted probability column name.
        num_bins: Number of equal-sized bins to divide the data into based on rank.

    Returns:
        pd.DataFrame: Metrics calculated cumulatively for each bin
                      (Top N%, where N corresponds to bin boundaries).
                      Includes: bin, count, positives, cumulative_count,
                      cumulative_positives, cumulative_precision, cumulative_recall,
                      cumulative_lift.
    """
    if not pd.api.types.is_numeric_dtype(df[y_true_col]):
         df[y_true_col] = pd.to_numeric(df[y_true_col], errors='coerce')
    if not pd.api.types.is_numeric_dtype(df[y_pred_col]):
         df[y_pred_col] = pd.to_numeric(df[y_pred_col], errors='coerce')

    # Drop rows where essential columns are NaN after potential coercion
    temp_df = df[[y_true_col, y_pred_col]].dropna().copy()
    if temp_df.empty:
        warnings.warn("DataFrame empty after dropping NaNs in true/pred columns. Returning empty metrics.")
        return pd.DataFrame()

    # Sort by prediction score descending
    temp_df = temp_df.sort_values(by=y_pred_col, ascending=False).reset_index(drop=True)

    # Assign bin labels (handle potential non-unique scores causing uneven bins)
    # Use rank first to handle ties, then qcut on ranks for more even bins
    temp_df['rank'] = temp_df[y_pred_col].rank(method='first', ascending=False)
    try:
        # Add small epsilon to avoid issues with exact boundaries if needed
        temp_df['bin'] = pd.qcut(temp_df['rank'], q=num_bins, labels=False, duplicates='drop') + 1 # Bins 1 to num_bins
    except ValueError as e:
         warnings.warn(f"Could not create {num_bins} bins, possibly due to ties or few data points. Returning fewer bins or empty DF. Error: {e}")
         # Fallback: try fewer bins or just return empty
         try:
             n_bins_reduced = max(1, num_bins // 2)
             warnings.warn(f"Trying {n_bins_reduced} bins instead.")
             temp_df['bin'] = pd.qcut(temp_df['rank'], q=n_bins_reduced, labels=False, duplicates='drop') + 1
             num_bins = n_bins_reduced # Update num_bins to actual number created
         except:
              return pd.DataFrame()


    # Aggregate per bin
    agg_df = temp_df.groupby('bin').agg(
        count=(y_true_col, 'size'),
        positives=(y_true_col, 'sum')
    ).reset_index()

    # Calculate cumulative metrics
    agg_df['cumulative_count'] = agg_df['count'].cumsum()
    agg_df['cumulative_positives'] = agg_df['positives'].cumsum()

    total_positives = agg_df['positives'].sum()
    total_count = agg_df['count'].sum()
    overall_positive_rate = total_positives / total_count if total_count > 0 else 0

    # Avoid division by zero
    agg_df['cumulative_precision'] = agg_df['cumulative_positives'] / agg_df['cumulative_count'].replace(0, np.nan)
    agg_df['cumulative_recall'] = agg_df['cumulative_positives'] / total_positives if total_positives > 0 else 0
    agg_df['cumulative_lift'] = agg_df['cumulative_precision'] / overall_positive_rate if overall_positive_rate > 0 else np.nan

    # Add percentage of population for clarity
    agg_df['percentage_population'] = agg_df['cumulative_count'] / total_count if total_count > 0 else 0
    # Add bin labels like Top 10%, Top 20%
    agg_df['bin_label'] = (agg_df['bin'] * (100 / num_bins)).round(1).astype(str) + '%'


    return agg_df

def plot_lift_chart(
    rank_metrics_df: pd.DataFrame,
    figsize: Tuple[int, int] = DEFAULT_FIG_SIZE
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plots the Cumulative Lift curve based on rank metrics.

    Args:
        rank_metrics_df: DataFrame output from calculate_rank_metrics.
        figsize: Figure size.

    Returns:
        Tuple containing the Figure and Axes objects.
    """
    if not all(col in rank_metrics_df.columns for col in ['percentage_population', 'cumulative_lift', 'bin_label']):
         raise ValueError("Input DataFrame must contain 'percentage_population', 'cumulative_lift', and 'bin_label' columns.")

    fig, ax = plt.subplots(figsize=figsize)

    # Use percentage population for x-axis
    x_values = rank_metrics_df['percentage_population'] * 100
    y_values = rank_metrics_df['cumulative_lift']

    ax.plot(x_values, y_values, marker='o', label='Cumulative Lift')
    ax.axhline(1, color='grey', linestyle='--', label='Baseline (Lift = 1)')

    ax.set_xlabel("Top % Population Targeted")
    ax.set_ylabel("Cumulative Lift")
    ax.set_title("Lift Chart")
    ax.legend()
    ax.grid(True, alpha=0.5)

    # Optional: Annotate x-axis with bin labels if not too crowded
    # ax.set_xticks(x_values)
    # ax.set_xticklabels(rank_metrics_df['bin_label']) # Can get crowded

    plt.tight_layout()
    return fig, ax

def rank_performance_by_group(
    df: pd.DataFrame,
    group_feature: str,
    y_true_col: str = "y_true",
    y_pred_col: str = "PredictionScore",
    num_rank_bins: int = 10, # e.g., 10 for deciles
    numeric_q: Optional[int] = 10,
    cat_top_n: Optional[int] = 10,
) -> pd.DataFrame:
    """
    Computes rank-based classification performance metrics (Precision, Recall, Lift)
    separately for each subgroup defined by a feature.

    Metrics are calculated cumulatively across rank bins (e.g., deciles) *within each subgroup*.

    Args:
        df: DataFrame with true labels and predicted probabilities.
        group_feature: Column name for the grouping feature.
        y_true_col: True outcome column name (binary 0/1).
        y_pred_col: Predicted probability column name.
        num_rank_bins: Number of equal-sized rank bins (e.g., 10 for deciles).
        numeric_q: Number of quantile bins for numeric grouping features. If None, uses raw values.
        cat_top_n: Number of top categories for categorical grouping features. If None, uses raw values.

    Returns:
        pd.DataFrame: Rank-based metrics (cumulative_precision, cumulative_recall,
                      cumulative_lift, etc.) calculated per rank bin within each group.
    """
    temp_df = df.copy()

    # 1. Create the primary grouping column
    if numeric_q is not None or cat_top_n is not None:
        temp_df['group'] = _create_group_column(
            temp_df, group_feature, numeric_q=numeric_q, cat_top_n=cat_top_n, include_nan=True
        )
    else:
        # Use raw feature values, ensuring consistent NaN handling
        temp_df['group'] = temp_df[group_feature]
        if temp_df['group'].isna().any():
            temp_df['group'] = temp_df['group'].astype(object).fillna("NaN")
        temp_df['group'] = temp_df['group'].astype('category')

    # --- Ensure prediction and true columns are numeric ---
    if not pd.api.types.is_numeric_dtype(temp_df[y_true_col]):
         temp_df[y_true_col] = pd.to_numeric(temp_df[y_true_col], errors='coerce')
    if not pd.api.types.is_numeric_dtype(temp_df[y_pred_col]):
         temp_df[y_pred_col] = pd.to_numeric(temp_df[y_pred_col], errors='coerce')
    # Drop rows where essential columns became NaN after coercion
    temp_df = temp_df.dropna(subset=[y_true_col, y_pred_col, 'group'])


    all_results = []
    unique_groups = temp_df["group"].unique()

    # 2. Iterate through each subgroup
    for grp in unique_groups:
        grp_df = temp_df[temp_df["group"] == grp].copy()
        group_total_count = len(grp_df)
        group_total_positives = grp_df[y_true_col].sum()
        group_overall_positive_rate = group_total_positives / group_total_count if group_total_count > 0 else 0

        if group_total_count == 0:
            warnings.warn(f"Group '{grp}' is empty. Skipping rank metrics calculation.")
            continue

        # 3. Calculate rank metrics *within this subgroup*
        # Sort by prediction score descending within the group
        grp_df = grp_df.sort_values(by=y_pred_col, ascending=False).reset_index(drop=True)

        # Assign rank bins within the group
        grp_df['rank_within_group'] = grp_df[y_pred_col].rank(method='first', ascending=False)
        try:
            # Bins are 1, 2, ..., num_rank_bins
            grp_df['rank_bin'] = pd.qcut(grp_df['rank_within_group'], q=num_rank_bins, labels=False, duplicates='drop') + 1
            actual_bins_created = grp_df['rank_bin'].nunique() # Check how many bins were actually made
        except ValueError:
            warnings.warn(f"Could not create {num_rank_bins} rank bins within group '{grp}'. Skipping rank metrics for this group.")
            continue # Skip to next group if binning fails

        # Aggregate per rank bin within the group
        agg_grp_df = grp_df.groupby('rank_bin').agg(
            count_in_rank_bin=(y_true_col, 'size'),
            positives_in_rank_bin=(y_true_col, 'sum')
        ).reset_index()

        # Calculate cumulative metrics within the group
        agg_grp_df['cumulative_count_within_group'] = agg_grp_df['count_in_rank_bin'].cumsum()
        agg_grp_df['cumulative_positives_within_group'] = agg_grp_df['positives_in_rank_bin'].cumsum()

        # Precision, Recall, Lift *relative to the group's statistics*
        agg_grp_df['cumulative_precision_within_group'] = agg_grp_df['cumulative_positives_within_group'] / agg_grp_df['cumulative_count_within_group'].replace(0, np.nan)
        agg_grp_df['cumulative_recall_within_group'] = agg_grp_df['cumulative_positives_within_group'] / group_total_positives if group_total_positives > 0 else 0
        agg_grp_df['cumulative_lift_within_group'] = agg_grp_df['cumulative_precision_within_group'] / group_overall_positive_rate if group_overall_positive_rate > 0 else np.nan

        # Add percentage population within group for plotting x-axis
        agg_grp_df['percentage_population_within_group'] = agg_grp_df['cumulative_count_within_group'] / group_total_count if group_total_count > 0 else 0

        # Add group identifier and group totals for context
        agg_grp_df['group'] = grp
        agg_grp_df['group_total_count'] = group_total_count
        agg_grp_df['group_total_positives'] = group_total_positives

        all_results.append(agg_grp_df)

    if not all_results: # If no groups had valid results
        warnings.warn("No rank metrics calculated for any group.")
        # Return empty df with expected columns
        return pd.DataFrame(columns=[
            'rank_bin', 'count_in_rank_bin', 'positives_in_rank_bin',
            'cumulative_count_within_group', 'cumulative_positives_within_group',
            'cumulative_precision_within_group', 'cumulative_recall_within_group',
            'cumulative_lift_within_group', 'percentage_population_within_group',
            'group', 'group_total_count', 'group_total_positives'
        ])

    # Concatenate results from all groups
    final_results_df = pd.concat(all_results, ignore_index=True)

    # Add original feature name for reference
    final_results_df[group_feature] = final_results_df['group']

    # Sort results for better readability (optional)
    final_results_df['group_sort_key'] = final_results_df['group'].apply(custom_sort_key)
    final_results_df = final_results_df.sort_values(['group_sort_key', 'rank_bin']).drop(columns='group_sort_key')

    return final_results_df

def plot_rank_metrics_by_group(
    rank_perf_df: pd.DataFrame,
    metric_to_plot: str, # e.g., 'cumulative_precision_within_group'
    group_col: str = "group",
    figsize: Tuple[int, int] = DEFAULT_FIG_SIZE
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plots a specific rank-based metric (Precision, Recall, or Lift) vs.
    Top % Population Targeted, comparing different subgroups.

    Args:
        rank_perf_df: DataFrame output from `rank_performance_by_group`.
        metric_to_plot: The name of the metric column to plot on the Y-axis
                        (e.g., 'cumulative_precision_within_group',
                         'cumulative_recall_within_group', 'cumulative_lift_within_group').
        group_col: The column in rank_perf_df identifying the groups (usually 'group').
        figsize: Figure size.

    Returns:
        Tuple containing the Figure and Axes objects.
    """
    required_cols = ['percentage_population_within_group', metric_to_plot, group_col]
    if not all(col in rank_perf_df.columns for col in required_cols):
        raise ValueError(f"Input DataFrame must contain columns: {required_cols}")

    unique_groups = rank_perf_df[group_col].unique()
    groups_sorted = sorted(unique_groups, key=custom_sort_key) # Sort groups for legend/plotting order

    fig, ax = plt.subplots(figsize=figsize)

    # Use seaborn for easier plotting with hue for groups
    sns.lineplot(
        data=rank_perf_df,
        x='percentage_population_within_group',
        y=metric_to_plot,
        hue=group_col,
        hue_order=groups_sorted, # Ensure consistent order
        marker='.', # Add markers
        ax=ax
    )

    # Improve labels based on metric name
    metric_label = metric_to_plot.replace('cumulative_', '').replace('_within_group', '').replace('_', ' ').title()
    ax.set_xlabel("Top % Population Targeted (within group)")
    ax.set_ylabel(f"Cumulative {metric_label}")
    ax.set_title(f"{metric_label} vs. Top % Targeted by Group: {group_col}")

    # Add baseline for lift if plotting lift
    if 'lift' in metric_to_plot.lower():
        ax.axhline(1, color='grey', linestyle='--', label='Baseline (Lift = 1)')
        # Ensure baseline label is included if legend exists
        handles, labels = ax.get_legend_handles_labels()
        if 'Baseline (Lift = 1)' not in labels:
             # Manually add handle if seaborn didn't pick it up (less common with lineplot)
             from matplotlib.lines import Line2D
             handles.append(Line2D([0], [0], color='grey', linestyle='--', lw=1))
             labels.append('Baseline (Lift = 1)')
             ax.legend(handles=handles, labels=labels, title=group_col, bbox_to_anchor=(1.03, 1), loc='upper left')
        else:
             ax.legend(title=group_col, bbox_to_anchor=(1.03, 1), loc='upper left')

    else:
        # Adjust legend position
        ax.legend(title=group_col, bbox_to_anchor=(1.03, 1), loc='upper left')

    # Format x-axis as percentage
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x*100:.0f}%'))
    ax.set_xlim(left=0) # Start x-axis at 0

    ax.grid(True, alpha=0.5)
    plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout for legend

    return fig, ax

def plot_group_summary_stats(
    rank_perf_df: pd.DataFrame,
    group_col: str = "group",
    figsize: Tuple[int, int] = (12, 5)
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Plots the total count and baseline positive rate for each subgroup.

    Provides context for interpreting rank-based performance plots.

    Args:
        rank_perf_df: DataFrame output from `rank_performance_by_group`.
        group_col: The column identifying the groups (usually 'group').
        figsize: Figure size for the entire figure containing two subplots.

    Returns:
        Tuple containing the Figure and an array of the two Axes objects.
    """
    required_cols = [group_col, 'group_total_count', 'group_total_positives']
    if not all(col in rank_perf_df.columns for col in required_cols):
        raise ValueError(f"Input DataFrame must contain columns: {required_cols}")

    # 1. Get unique group summary stats
    group_summary = rank_perf_df[[
        group_col, 'group_total_count', 'group_total_positives'
    ]].drop_duplicates().copy()

    # 2. Calculate baseline positive rate per group
    group_summary['group_positive_rate'] = (
        group_summary['group_total_positives'] / group_summary['group_total_count'].replace(0, np.nan)
    )
    group_summary['group_positive_rate'] = group_summary['group_positive_rate'].fillna(0)

    # 3. Sort groups for consistent plotting
    group_summary['sort_key'] = group_summary[group_col].apply(custom_sort_key)
    group_summary = group_summary.sort_values('sort_key').reset_index(drop=True)

    # 4. Create Figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle(f'Subgroup Summary Statistics: {group_col}', fontsize=14)

    # --- Plot 1: Group Counts ---
    ax1 = axes[0]
    groups = group_summary[group_col].astype(str)
    counts = group_summary['group_total_count']
    bars1 = ax1.bar(groups, counts, color=sns.color_palette("viridis", len(groups)))
    ax1.set_title('Group Size (Total Count)')
    ax1.set_ylabel('Count')
    # Set rotation for x-tick labels directly
    ax1.tick_params(axis='x', rotation=45) # Remove ha='right'
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    ax1.bar_label(bars1, fmt='{:,.0f}', padding=3, fontsize=8)


    # --- Plot 2: Group Positive Rate ---
    ax2 = axes[1]
    rates = group_summary['group_positive_rate']
    bars2 = ax2.bar(groups, rates, color=sns.color_palette("viridis", len(groups)))
    ax2.set_title('Baseline Positive Rate by Group')
    ax2.set_ylabel('Positive Rate (Positives / Count)')
    # Set rotation for x-tick labels directly
    ax2.tick_params(axis='x', rotation=45) # Remove ha='right'
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    ax2.bar_label(bars2, fmt='{:.2%}', padding=3, fontsize=8)


    plt.tight_layout(rect=[0, 0, 1, 0.95])

    return fig, axes