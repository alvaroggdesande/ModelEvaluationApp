# Binary Classification Model Monitoring Dashboard

This Streamlit dashboard provides tools to monitor the performance, fairness, and stability of binary classification models.

## Features

*   **Overall Performance:** View ROC AUC, PR AUC, Calibration, and metrics at adjustable thresholds.
*   **Rank Metrics:** Analyze Gains, Lift, and Precision/Recall across prediction score deciles/quantiles.
*   **Subgroup Analysis:** Compare rank-based performance metrics across different subgroups defined by features (e.g., age, payment history).
*   **Data Drift:** Visualize distribution shifts between training and prediction datasets and view drift summaries (PSI vs. Importance).

## Project Structure

*   `app.py`: Main dashboard entry point.
*   `pages/`: Contains individual dashboard pages for different analyses.
*   `utils/`: Helper functions for data loading, calculations, and plotting.
*   `modules/`: Your custom Python modules for core analysis logic (performance, drift).
*   `data/`: Optional directory for sample data.
*   `requirements.txt`: Required Python packages.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd binary_clf_dashboard
    ```
2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
3.  **Install requirements:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Place analysis modules:** Ensure your `performance_analysis.py` and `datadrift_analysis.py` files are inside the `modules/` directory.
5.  **Run the Streamlit app:**
    ```bash
    streamlit run app.py
    ```
6.  Open your web browser to the local URL provided by Streamlit (usually `http://localhost:8501`).
7.  Upload your prediction data (and optionally training data/drift metrics) via the sidebar.

## Customization

*   **Analysis Logic:** Modify functions within `modules/` or `utils/calculations.py`.
*   **Plots:** Update plotting functions in `utils/plotting.py` (uses Plotly).
*   **Layout/UI:** Edit `app.py` and files within the `pages/` directory using Streamlit components.
*   **Styling:** Customize Streamlit themes via `.streamlit/config.toml`.