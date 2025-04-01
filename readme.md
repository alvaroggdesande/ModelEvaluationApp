# Binary Classification Model Monitoring Dashboard

A Streamlit application designed to monitor the performance, stability, fairness, and explainability of binary classification models.

## Features

*   **Overall Performance (Page 1):**
    *   View aggregate metrics (ROC AUC, PR AUC) for prediction or reference datasets.
    *   Analyze Precision, Recall, F1-Score at adjustable probability thresholds.
    *   Visualize performance using interactive ROC, Precision-Recall, and Calibration curves.
    *   Evaluate rank-based metrics with Lift and Gain charts.
    *   Compare prediction score distributions.
*   **Model Comparison (Page 2):**
    *   Directly compare performance curves (ROC, PR, Calibration, Lift) between the prediction/holdout dataset and a reference (e.g., training/test) dataset.
    *   Side-by-side comparison of key metrics.
    *   Requires both datasets to be loaded.
*   **Subgroup Analysis (Page 3):**
    *   Analyze performance metrics (Rank Metrics, Mean Prediction vs. Target) broken down by subgroups based on selected features.
    *   Optionally filter features using **SHAP importance** (if loaded) to focus analysis.
    *   Supports automatic binning for numeric features and grouping for categorical features.
*   **Data Drift (Page 4):**
    *   Calculate Population Stability Index (PSI) and Kolmogorov-Smirnov (KS) tests to detect distribution changes between reference and prediction datasets.
    *   Visualize drift vs. feature importance (**SHAP-enhanced**) to prioritize investigation.
    *   Compare feature distributions side-by-side.
    *   Optionally filter analysis to top N important features (SHAP).
*   **Explainability (Page 5 - SHAP):**
    *   Visualize global feature importance using bar plots (including aggregation of less important features).
    *   Explore feature distributions and interactions with Beeswarm and Dependence plots (supports downsampling for large datasets).
    *   Analyze individual predictions using Waterfall plots, selectable by index or prediction score rank (Lowest/Median/Highest).
    *   Displays feature values on their original scale where possible (using inverse transformation during offline SHAP data preparation).

## Setup & Usage

1.  **Environment:** Create a Python environment (e.g., using `conda` or `venv`) and install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
2.  **Prepare Data (Offline):**
    *   **Prediction/Reference Data:** Have CSV or Parquet files ready containing your features, true labels (column named `y_true`), and prediction probabilities (column named `y_pred_prob`).
    *   **(Optional) SHAP Data:** Pre-calculate SHAP values **offline** using your model and preprocessor. Generate the following files (see `pre_calculate_shap.py` example in development discussion):
        *   `shap_values.npy`: SHAP values for the positive class.
        *   `shap_features.json`: List of processed feature names corresponding to SHAP values.
        *   `shap_base_value.json`: Base value (`E[f(X)]`) on the logit/margin scale.
        *   `shap_display_data.parquet`: Data corresponding to SHAP values, *after* applying `preprocessor.inverse_transform` where possible (for display purposes).
3.  **Run the App:**
    ```bash
    streamlit run src/ModelEvaluationApp/app.py
    ```
4.  **Upload Data:** Use the sidebar in the app to upload your Prediction, Reference (optional), and SHAP data files (optional).
5.  **Navigate:** Explore the different pages using the sidebar navigation.

## Deployment

This app can be deployed to Streamlit Community Cloud. Ensure your repository is **public** and contains the necessary code files and `requirements.txt`. Do **not** commit data or model files. Use the following deployment settings:
*   **Repository:** `<your-username>/<your-repo-name>`
*   **Branch:** `main` (or your deployment branch)
*   **Main file path:** `ModelEvaluationApp/app.py`