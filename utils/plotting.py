import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from scipy import stats
import warnings
from typing import Optional, Tuple
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
# Import your matplotlib plotting functions if you want wrappers
from modules.prediction_holdout_analysis import plot_group_summary_stats, custom_sort_key
from modules.data_drift_analysis import plot_distribution_drift
from .theme import PLOT_COLOR_SEQUENCE, SECONDARY_BACKGROUND_COLOR, TEXT_COLOR, GRID_COLOR, BACKGROUND_COLOR

# --- Interactive Plots using Plotly ---

def plot_roc_curve_interactive(y_true, y_pred_prob, title="ROC Curve"):
    """Generates an interactive ROC curve plot using Plotly."""
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    fig = px.area(
        x=fpr, y=tpr,
        title=f'{title} (AUC = {roc_auc:.2f})',
        labels=dict(x='False Positive Rate', y='True Positive Rate'),
        width=700, height=500
    )
    fig.add_shape(type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1)
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    fig.update_xaxes(constrain='domain')
    return fig

def plot_pr_curve_interactive(y_true, y_pred_prob, title="Precision-Recall Curve"):
    """Generates an interactive Precision-Recall curve plot using Plotly."""
    precision, recall, _ = precision_recall_curve(y_true, y_pred_prob)
    pr_auc = average_precision_score(y_true, y_pred_prob)
    fig = px.area(
        x=recall, y=precision,
        title=f'{title} (AUC = {pr_auc:.2f})',
        labels=dict(x='Recall', y='Precision'),
        width=700, height=500
    )
    fig.add_hline(y=np.mean(y_true), line=dict(dash='dash'), name='Baseline') # Baseline based on prevalence
    fig.update_yaxes(range=[0, 1.05])
    fig.update_xaxes(range=[0, 1])
    return fig

def plot_calibration_interactive(prob_pred, prob_true, title="Calibration Plot"):
    """Generates an interactive calibration plot using Plotly."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=prob_pred, y=prob_true, mode='lines+markers', name='Calibration curve'))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Perfect Calibration', line=dict(dash='dash')))
    fig.update_layout(
        title=title,
        xaxis_title='Mean Predicted Probability',
        yaxis_title='Fraction of Positives',
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1]),
        width=700, height=500
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig

def plot_rank_metrics_interactive(rank_metrics_df, title="Precision and Recall by Top % Targeted"):
    """Plots interactive Precision and Recall vs Top % using Plotly."""
    color_map = {
        'Precision': PLOT_COLOR_SEQUENCE[0],  # e.g. the first default Plotly color
        'Recall':    PLOT_COLOR_SEQUENCE[6],  # e.g. the seventh default Plotly color
    }
    pop_pct = rank_metrics_df['percentage_population'] * 100
    fig = go.Figure()

    # precision trace
    fig.add_trace(go.Scatter(
        x=pop_pct,
        y=rank_metrics_df['cumulative_precision'],
        mode='lines+markers',
        name='Precision',
        line=dict(color=color_map['Precision']),
        marker=dict(color=color_map['Precision'])
    ))

    # recall trace
    fig.add_trace(go.Scatter(
        x=pop_pct,
        y=rank_metrics_df['cumulative_recall'],
        mode='lines+markers',
        name='Recall',
        line=dict(color=color_map['Recall']),
        marker=dict(color=color_map['Recall'])
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Top % Population Targeted",
        yaxis_title="Precision / Recall",
        yaxis=dict(range=[0, 1.05]),
        xaxis=dict(range=[0, 101]),
        width=700, height=500,
        hovermode="x unified"
    )
    return fig

def plot_lift_chart_interactive(rank_metrics_df, title="Lift Chart"):
    """Plots interactive Lift chart using Plotly."""
    fig = go.Figure()
    pop_pct = rank_metrics_df['percentage_population'] * 100
    fig.add_trace(go.Scatter(x=pop_pct, y=rank_metrics_df['cumulative_lift'],
                             mode='lines+markers', name='Cumulative Lift'))
    fig.add_hline(y=1, line=dict(dash='dash', color='grey'), name='Baseline (Lift=1)')
    fig.update_layout(
        title=title,
        xaxis_title="Top % Population Targeted",
        yaxis_title="Cumulative Lift",
        xaxis=dict(range=[0, 101]),
        width=700, height=500,
        hovermode="x unified"
    )
    return fig


def plot_rank_metrics_by_group_interactive(rank_perf_df, metric_to_plot, group_col="group"):
     """Plots a specific rank-based metric vs. Top % Targeted by group using Plotly."""
     required_cols = ['percentage_population_within_group', metric_to_plot, group_col]
     if not all(col in rank_perf_df.columns for col in required_cols):
         raise ValueError(f"Input DataFrame must contain columns: {required_cols}")

     metric_label = metric_to_plot.replace('cumulative_', '').replace('_within_group', '').replace('_', ' ').title()
     # REMOVED title generation here, will be set by markdown in Streamlit page

     # Use Plotly Express for easy coloring by group
     fig = px.line(
         rank_perf_df,
         x='percentage_population_within_group',
         y=metric_to_plot,
         color=group_col,
         markers=True,
         labels={ # Map column names to readable labels
             'percentage_population_within_group': 'Top % Population Targeted (within group)',
             metric_to_plot: f"Cumulative {metric_label}",
             group_col: 'Group' # Use 'Group' for legend title
         },
         title=None # <<< Explicitly set initial title to None
     )

     # Add baseline for lift
     if 'lift' in metric_to_plot.lower():
         fig.add_hline(y=1, line=dict(dash='dash', color='grey'), name='Baseline (Lift=1)', annotation_text="Baseline")

     fig.update_layout(
         xaxis=dict(tickformat='.0%'), # Format x-axis as percentage
         hovermode="x unified",
         legend_title_text='Group' # Set legend title here
     )
     fig.update_xaxes(range=[0, 1.01])
     if 'precision' in metric_to_plot.lower() or 'recall' in metric_to_plot.lower():
          fig.update_yaxes(range=[0, 1.05])

     return fig


def plot_score_distribution_plotly(
    scores: pd.Series,
    feature_name: str = "Prediction Score",
    bins: int = 30,
    show_kde: bool = True,
    show_normal_fit: bool = True,
    title: str = "Prediction Score Distribution"
) -> go.Figure:
    """
    Plots an interactive histogram of prediction scores with optional KDE
    and normal distribution fit overlays using Plotly.

    Args:
        scores: Pandas Series of prediction scores.
        feature_name: Name of the score/feature for axis labeling.
        bins: Number of bins for the histogram.
        show_kde: If True, overlay a Kernel Density Estimate.
        show_normal_fit: If True, overlay a fitted Normal distribution curve.
        title: Title for the plot.

    Returns:
        Plotly Figure object.
    """
    scores_clean = scores.dropna()
    if scores_clean.empty:
        # Return an empty figure with a message if no data
        fig = go.Figure()
        fig.update_layout(title=title, xaxis_title=feature_name, yaxis_title="Density",
                          annotations=[dict(text="No valid score data to display",
                                            showarrow=False, xref='paper', yref='paper')])
        return fig

    fig = go.Figure()

    # 1. Histogram
    fig.add_trace(go.Histogram(
        x=scores_clean,
        nbinsx=bins,
        name='Histogram',
        histnorm='probability density', # Normalize to density
        #marker_color='lightblue', # Match user example style
        opacity=0.75,
        marker_line=dict(width=1) # Add edge color
    ))

    # Calculate plot range for KDE/Normal fit
    xmin, xmax = scores_clean.min(), scores_clean.max()
    x_range = np.linspace(xmin, xmax, 200) # More points for smooth curves

    # 2. KDE (Optional)
    if show_kde:
        try:
            kde = stats.gaussian_kde(scores_clean)
            y_kde = kde(x_range)
            fig.add_trace(go.Scatter(
                x=x_range,
                y=y_kde,
                mode='lines',
                name='KDE',
                line=dict(color='skyblue', width=2) # Match user example style
            ))
        except Exception as e:
            warnings.warn(f"Could not generate KDE plot: {e}")

    # 3. Normal Fit (Optional)
    if show_normal_fit:
        try:
            mu, std = stats.norm.fit(scores_clean)
            y_norm = stats.norm.pdf(x_range, mu, std)
            fig.add_trace(go.Scatter(
                x=x_range,
                y=y_norm,
                mode='lines',
                name=f'Normal fit: μ={mu:.2f}, σ={std:.2f}',
                line=dict(color='red', width=2) # Match user example style
            ))
        except Exception as e:
            warnings.warn(f"Could not generate Normal fit plot: {e}")


    # Layout configuration
    fig.update_layout(
        title=title,
        xaxis_title=feature_name,
        yaxis_title="Density",
        bargap=0.01, # Small gap between bars
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
        width=700, height=500 # Can be overridden by use_container_width
    )

    return fig

#Page 2. Model comparison

# --- >>> NEW Comparison Plots <<< ---

def plot_roc_comparison_interactive(y_true1, y_prob1, y_true2, y_prob2, name1="Dataset 1", name2="Dataset 2", title="ROC Curve Comparison"):
    """Generates an interactive ROC curve plot comparing two datasets."""
    fpr1, tpr1, _ = roc_curve(y_true1, y_prob1)
    roc_auc1 = auc(fpr1, tpr1)
    fpr2, tpr2, _ = roc_curve(y_true2, y_prob2)
    roc_auc2 = auc(fpr2, tpr2)

    fig = go.Figure()

    # Plot Dataset 1
    fig.add_trace(go.Scatter(
        x=fpr1, y=tpr1, mode='lines', name=f'{name1} (AUC = {roc_auc1:.2f})',
        line=dict(color=PLOT_COLOR_SEQUENCE[0], width=2) # Use first theme color
    ))

    # Plot Dataset 2
    fig.add_trace(go.Scatter(
        x=fpr2, y=tpr2, mode='lines', name=f'{name2} (AUC = {roc_auc2:.2f})',
        line=dict(color=PLOT_COLOR_SEQUENCE[6], width=2) # Use another theme color (e.g., 7th)
    ))

    # Add diagonal baseline
    fig.add_shape(type='line', line=dict(dash='dash', color=GRID_COLOR), x0=0, x1=1, y0=0, y1=1)

    fig.update_layout(
        title=title,
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        xaxis=dict(constrain='domain'),
        yaxis=dict(scaleanchor="x", scaleratio=1),
        legend=dict(yanchor="bottom", y=0.01, xanchor="right", x=0.99),
        width=700, height=500 # Default size
    )
    return fig

def plot_pr_comparison_interactive(y_true1, y_prob1, y_true2, y_prob2, name1="Dataset 1", name2="Dataset 2", title="Precision-Recall Curve Comparison"):
    """Generates an interactive PR curve plot comparing two datasets."""
    precision1, recall1, _ = precision_recall_curve(y_true1, y_prob1)
    pr_auc1 = average_precision_score(y_true1, y_prob1)
    precision2, recall2, _ = precision_recall_curve(y_true2, y_prob2)
    pr_auc2 = average_precision_score(y_true2, y_prob2)

    fig = go.Figure()

    # Plot Dataset 1
    fig.add_trace(go.Scatter(
        x=recall1, y=precision1, mode='lines', name=f'{name1} (AUC = {pr_auc1:.2f})',
        line=dict(color=PLOT_COLOR_SEQUENCE[0], width=2)
    ))

    # Plot Dataset 2
    fig.add_trace(go.Scatter(
        x=recall2, y=precision2, mode='lines', name=f'{name2} (AUC = {pr_auc2:.2f})',
        line=dict(color=PLOT_COLOR_SEQUENCE[6], width=2)
    ))

    # Add baselines (might differ if prevalence differs)
    baseline1 = np.mean(y_true1)
    baseline2 = np.mean(y_true2)
    fig.add_hline(y=baseline1, line=dict(dash='dash', color=PLOT_COLOR_SEQUENCE[0], width=1),
                  annotation_text=f"{name1} Baseline ({baseline1:.2f})", annotation_position="bottom right")
    fig.add_hline(y=baseline2, line=dict(dash='dot', color=PLOT_COLOR_SEQUENCE[6], width=1),
                  annotation_text=f"{name2} Baseline ({baseline2:.2f})", annotation_position="top right")


    fig.update_layout(
        title=title,
        xaxis_title='Recall',
        yaxis_title='Precision',
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1.05]),
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
        width=700, height=500
    )
    return fig

def plot_calibration_comparison_interactive(prob_pred1, prob_true1, prob_pred2, prob_true2, name1="Dataset 1", name2="Dataset 2", title="Calibration Plot Comparison"):
    """Generates an interactive calibration plot comparing two datasets."""
    fig = go.Figure()

    # Plot Dataset 1
    fig.add_trace(go.Scatter(x=prob_pred1, y=prob_true1, mode='lines+markers', name=name1,
                             line=dict(color=PLOT_COLOR_SEQUENCE[0]), marker=dict(symbol='circle')))

    # Plot Dataset 2
    fig.add_trace(go.Scatter(x=prob_pred2, y=prob_true2, mode='lines+markers', name=name2,
                             line=dict(color=PLOT_COLOR_SEQUENCE[6]), marker=dict(symbol='x')))

    # Add Ideal calibration line
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Perfect Calibration',
                             line=dict(dash='dash', color=GRID_COLOR)))

    fig.update_layout(
        title=title,
        xaxis_title='Mean Predicted Probability',
        yaxis_title='Fraction of Positives',
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1], scaleanchor="x", scaleratio=1),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        width=700, height=500
    )
    return fig

def plot_rank_metrics_comparison_interactive(rank_df1, rank_df2, name1="Dataset 1", name2="Dataset 2", title="Precision & Recall vs Top % Targeted"):
    """Plots interactive Precision and Recall vs Top % comparing two datasets."""
    fig = go.Figure()

    # Dataset 1
    pop_pct1 = rank_df1['percentage_population'] * 100
    fig.add_trace(go.Scatter(x=pop_pct1, y=rank_df1['cumulative_precision'], mode='lines+markers',
                             name=f'Precision ({name1})', line=dict(color=PLOT_COLOR_SEQUENCE[0]), marker=dict(symbol='circle')))
    fig.add_trace(go.Scatter(x=pop_pct1, y=rank_df1['cumulative_recall'], mode='lines+markers',
                             name=f'Recall ({name1})', line=dict(color=PLOT_COLOR_SEQUENCE[0], dash='dot'), marker=dict(symbol='circle-open'))) # Dotted line for recall

    # Dataset 2
    pop_pct2 = rank_df2['percentage_population'] * 100
    fig.add_trace(go.Scatter(x=pop_pct2, y=rank_df2['cumulative_precision'], mode='lines+markers',
                             name=f'Precision ({name2})', line=dict(color=PLOT_COLOR_SEQUENCE[6]), marker=dict(symbol='x')))
    fig.add_trace(go.Scatter(x=pop_pct2, y=rank_df2['cumulative_recall'], mode='lines+markers',
                             name=f'Recall ({name2})', line=dict(color=PLOT_COLOR_SEQUENCE[6], dash='dot'), marker=dict(symbol='x-open'))) # Dotted line for recall

    fig.update_layout(
        title=title,
        xaxis_title="Top % Population Targeted",
        yaxis_title="Precision / Recall",
        yaxis=dict(range=[0, 1.05]),
        xaxis=dict(range=[0, 101]),
        width=700, height=500,
        hovermode="x unified",
        legend=dict(yanchor="bottom", y=0.01, xanchor="right", x=0.99)
    )
    return fig

def plot_lift_chart_comparison_interactive(rank_df1, rank_df2, name1="Dataset 1", name2="Dataset 2", title="Lift Chart Comparison"):
    """Plots interactive Lift chart comparing two datasets."""
    fig = go.Figure()

    # Dataset 1
    pop_pct1 = rank_df1['percentage_population'] * 100
    fig.add_trace(go.Scatter(x=pop_pct1, y=rank_df1['cumulative_lift'], mode='lines+markers',
                             name=f'{name1}', line=dict(color=PLOT_COLOR_SEQUENCE[0])))

    # Dataset 2
    pop_pct2 = rank_df2['percentage_population'] * 100
    fig.add_trace(go.Scatter(x=pop_pct2, y=rank_df2['cumulative_lift'], mode='lines+markers',
                             name=f'{name2}', line=dict(color=PLOT_COLOR_SEQUENCE[6])))

    # Baseline
    fig.add_hline(y=1, line=dict(dash='dash', color=GRID_COLOR), name='Baseline (Lift=1)')

    fig.update_layout(
        title=title,
        xaxis_title="Top % Population Targeted",
        yaxis_title="Cumulative Lift",
        xaxis=dict(range=[0, 101]),
        width=700, height=500,
        hovermode="x unified",
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99)
    )
    return fig

# Page 3. Subgroup analysis
def plot_group_summary_stats_plotly(
    rank_perf_df: pd.DataFrame,
    group_col: str = "group",
    figsize_width: int = 1000, # Control overall width
    subplot_height: int = 400 # Height per subplot
) -> Tuple[go.Figure, None]: # Return fig, None (no axes array like mpl)
    """
    Plots the total count and baseline positive rate for each subgroup using Plotly.

    Args:
        rank_perf_df: DataFrame output from `rank_performance_by_group`.
        group_col: The column identifying the groups (usually 'group').
        figsize_width: Width of the combined figure.
        subplot_height: Height of each individual subplot.

    Returns:
        Tuple containing the Plotly Figure object and None.
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
    ).fillna(0)

    # 3. Sort groups for consistent plotting
    group_summary['sort_key'] = group_summary[group_col].apply(custom_sort_key)
    group_summary = group_summary.sort_values('sort_key').reset_index(drop=True)
    groups = group_summary[group_col].astype(str) # Use string labels

    # 4. Create Figure with two subplots (side-by-side)
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=('Group Size (Total Count)', 'Baseline Positive Rate by Group'))

    # --- Plot 1: Group Counts ---
    fig.add_trace(
        go.Bar(x=groups, y=group_summary['group_total_count'], name='Count', showlegend=False),
        row=1, col=1
    )
    # Add text labels for counts
    fig.update_traces(text=group_summary['group_total_count'].map('{:,.0f}'.format), textposition='outside', row=1, col=1)


    # --- Plot 2: Group Positive Rate ---
    fig.add_trace(
        go.Bar(x=groups, y=group_summary['group_positive_rate'], name='Positive Rate', showlegend=False),
        row=1, col=2
    )
    # Add text labels for rates (formatted as percentage)
    fig.update_traces(text=group_summary['group_positive_rate'].map('{:.2%}'.format), textposition='outside', row=1, col=2)


    # Update layout
    fig.update_layout(
        title_text=f'Subgroup Summary Statistics: {group_col}',
        height=subplot_height + 100, # Adjust total height based on subplot height + title/margins
        width=figsize_width,
        bargap=0.2,
        margin=dict(t=100) # Add top margin for main title
    )
    # Update axes specifically
    fig.update_yaxes(title_text="Count", row=1, col=1)
    fig.update_yaxes(title_text="Positive Rate", tickformat=".0%", row=1, col=2) # Format y-axis as percentage
    fig.update_xaxes(tickangle=45, row=1, col=1)
    fig.update_xaxes(tickangle=45, row=1, col=2)

    # Apply theme colors (optional - template might handle this)
    # fig.update_traces(marker_color=PLOT_COLOR_SEQUENCE[0], row=1, col=1) # Example
    # fig.update_traces(marker_color=PLOT_COLOR_SEQUENCE[0], row=1, col=2) # Example

    return fig, None # Return None for axes array consistent with Matplotlib return sig

def plot_subgroup_means_interactive(
    analysis_df: pd.DataFrame,
    feature_name: str,
    figsize_width: int = 800,
    subplot_height: int = 500
) -> go.Figure:
    """
    Creates an interactive Plotly chart showing subgroup counts (bars)
    and mean prediction/target (lines) on dual axes.

    Args:
        analysis_df: DataFrame output from calculate_subgroup_means. Must contain
                     'group', 'count', 'mean_prediction', 'mean_target'.
        feature_name: Name of the feature used for grouping (for title).
        figsize_width: Width of the figure.
        subplot_height: Height of the plot area.

    Returns:
        Plotly Figure object.
    """
    required_cols = ['group', 'count', 'mean_prediction', 'mean_target']
    if not all(col in analysis_df.columns for col in required_cols):
        raise ValueError(f"Input DataFrame must contain columns: {required_cols}")

    # Use make_subplots to create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Ensure group labels are strings for plotting categories
    group_labels = analysis_df['group'].astype(str)

    # 1. Add Count Bars (Primary Y-axis)
    fig.add_trace(
        go.Bar(
            x=group_labels,
            y=analysis_df['count'],
            name='Count',
            marker_color=PLOT_COLOR_SEQUENCE[2], # Example: Use 3rd color from theme for bars
            opacity=0.4 # Make bars semi-transparent
        ),
        secondary_y=False,
    )

    # 2. Add Mean Prediction Line (Secondary Y-axis)
    fig.add_trace(
        go.Scatter(
            x=group_labels,
            y=analysis_df['mean_prediction'],
            mode='lines+markers',
            name='Mean Prediction',
            line=dict(color=PLOT_COLOR_SEQUENCE[0]) # Example: Use 1st color
        ),
        secondary_y=True,
    )

    # 3. Add Mean Target Line (Secondary Y-axis)
    fig.add_trace(
        go.Scatter(
            x=group_labels,
            y=analysis_df['mean_target'],
            mode='lines+markers',
            name='Mean Target',
            line=dict(color=PLOT_COLOR_SEQUENCE[1]), # Example: Use 2nd color
            marker=dict(symbol='x') # Use 'x' marker for target
        ),
        secondary_y=True,
    )

    # 4. Configure Layout
    fig.update_layout(
        title_text=f"Subgroup Analysis by {feature_name}",
        xaxis_title=f"{feature_name} Bins/Categories",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        hovermode="x unified",
        height=subplot_height,
        width=figsize_width,
        plot_bgcolor=BACKGROUND_COLOR, # Use theme color if desired
        # margin=dict(t=50, b=100) # Adjust margins if needed for labels
    )

    # Configure Y-Axes
    fig.update_yaxes(
        title_text="Count",
        secondary_y=False,
        showgrid=True, gridcolor=BACKGROUND_COLOR, # Match theme grid
        color=TEXT_COLOR # Match theme text color
    )
    fig.update_yaxes(
        title_text="Mean Value",
        secondary_y=True,
        showgrid=False, # Avoid overlaying grids usually
        range=[0, max(analysis_df['mean_prediction'].max(), analysis_df['mean_target'].max()) * 1.1], # Auto-range secondary axis
        color=TEXT_COLOR
    )

    # Configure X-Axis Ticks
    fig.update_xaxes(tickangle=45, color=TEXT_COLOR)

    return fig

# page 4. Data drift
def plot_drift_summary_interactive(drift_importance_df, ks_alpha=0.05):
    """Plots interactive PSI vs. Importance using Plotly."""
    if not all(col in drift_importance_df.columns for col in ['feature', 'psi', 'feature_importance', 'ks_pvalue']):
        raise ValueError("Input DataFrame must include 'feature', 'psi', 'feature_importance', 'ks_pvalue'.")

    temp_df = drift_importance_df.copy().dropna(subset=['psi', 'feature_importance', 'ks_pvalue']) # Ensure no NaNs for plotting
    temp_df['ks_significant'] = temp_df['ks_pvalue'] < ks_alpha
    temp_df['ks_label'] = temp_df['ks_significant'].map({True: f'p < {ks_alpha}', False: f'p >= {ks_alpha}'})
    temp_df['size_scaled'] = np.maximum(temp_df['feature_importance'] * 2000, 5) # Scale size for plotting

    fig = px.scatter(
        temp_df,
        x='feature_importance',
        y='psi',
        size='size_scaled',
        color='ks_label',
        symbol='ks_significant',
        hover_name='feature', # Show feature name on hover
        hover_data={'psi': ':.3f', 'ks_pvalue':':.3f', 'feature_importance':':.3f', 'size_scaled': False}, # Customize hover data
        color_discrete_map={'p < 0.05': 'red', f'p >= {ks_alpha}': 'green'},
        symbol_map={True: 'x', False: 'circle'},
        title="Feature Drift (PSI) vs. Feature Importance (SHAP)",
        labels={
            'feature_importance': "Mean Absolute SHAP Value (Importance)",
            'psi': "Population Stability Index (PSI)",
            'ks_label': "KS p-value"
        }
    )

    # Add threshold lines
    fig.add_hline(y=0.1, line=dict(dash='dash', color='orange'), name='PSI=0.1', annotation_text="Minor Drift")
    fig.add_hline(y=0.25, line=dict(dash='dash', color='red'), name='PSI=0.25', annotation_text="Major Drift")

    fig.update_layout(legend_title_text="Legend")
    return fig

def plot_distribution_comparison_interactive(
    series1: pd.Series,
    series2: pd.Series,
    name1: str = "Reference (Train)",
    name2: str = "Current (Predict)",
    feature_name: str = "Feature",
    nbins: Optional[int] = None # Allow specifying number of bins
) -> go.Figure:
    """
    Plots interactive overlaid histograms/density plots using Plotly
    with adjustable transparency.

    Args:
        series1: Data series for the first distribution (e.g., training).
        series2: Data series for the second distribution (e.g., prediction).
        name1: Label for the first dataset.
        name2: Label for the second dataset.
        feature_name: Name of the feature for axis labeling and title.
        nbins: Optional number of bins for the histogram.

    Returns:
        Plotly Figure object.
    """
    # Ensure Series have names for concatenation if needed, drop NaNs
    s1_clean = series1.dropna()
    s1_clean.name = 'value'
    s2_clean = series2.dropna()
    s2_clean.name = 'value'

    if s1_clean.empty and s2_clean.empty:
        fig = go.Figure()
        fig.update_layout(title=f"Distribution Comparison: {feature_name}", xaxis_title=feature_name, yaxis_title="Density",
                          annotations=[dict(text="No valid data to display for either dataset",
                                            showarrow=False, xref='paper', yref='paper')])
        return fig

    # Create separate dataframes for Plotly Express
    df1 = pd.DataFrame({'value': s1_clean, 'Dataset': name1})
    df2 = pd.DataFrame({'value': s2_clean, 'Dataset': name2})
    df_plot = pd.concat([df1, df2], ignore_index=True)

    # Define colors from theme (use first two colors)
    color_map = {
        name1: PLOT_COLOR_SEQUENCE[0], # e.g., your primary blue/purple
        name2: PLOT_COLOR_SEQUENCE[6]  # e.g., your secondary orange/red
    }

    # --- Create Histogram with Adjusted Opacity ---
    fig = px.histogram(
        df_plot,
        x="value",
        color="Dataset",
        color_discrete_map=color_map, # Assign specific colors
        marginal="rug", # Keep rug plot
        histnorm='probability density',
        barmode='overlay',
        opacity=0.3, # <<< ADJUST OPACITY HERE (0.0 to 1.0) - 0.5 or 0.6 is usually good
        nbins=nbins, # Use specified nbins if provided
        title=f"Distribution Comparison: {feature_name}" # Title set here
    )

    # Update layout - Ensure title isn't set again if updated later
    fig.update_layout(
        xaxis_title=feature_name,
        yaxis_title="Density",
        legend_title_text="Dataset" # Set legend title
        # Remove width/height to allow container width usage
    )

    # If you *also* want KDE overlay (more complex in plotly histogram)
    # You would calculate KDE manually and add traces:
    #fig.add_trace(go.Scatter(x=x_kde1, y=y_kde1, mode='lines', name=f'{name1} KDE', line=dict(color=color_map[name1])))
    #fig.add_trace(go.Scatter(x=x_kde2, y=y_kde2, mode='lines', name=f'{name2} KDE', line=dict(color=color_map[name2])))


    return fig