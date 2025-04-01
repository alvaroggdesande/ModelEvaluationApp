# utils/data_loader.py
import streamlit as st
import pandas as pd
# Import necessary classes/functions for loading
#from ResultService.result import Result # Adjust import path if needed
# Import your BigQueryClient which handles GCS
#from IOService.gbq_client import BigQueryClient # Adjust import path
import io
import os # Import os

# Default path within the project for sample data
DEFAULT_DATA_PATH = os.path.join("ModelEvaluationApp/data", "sample_holdout_predictions.csv")

@st.cache_data
def load_prediction_data(uploaded_file=None, default_path=DEFAULT_DATA_PATH):
    """
    Loads prediction data (CSV or Parquet) from an uploaded file or a default path.
    Includes column name cleaning and debugging output.
    """
    data_source = None
    source_name = "Unknown"

    if uploaded_file is not None:
        data_source = uploaded_file
        source_name = uploaded_file.name
        st.info(f"Loading data from uploaded file: {source_name}")
    elif default_path and os.path.exists(default_path):
        data_source = default_path
        source_name = default_path
        st.info(f"No file uploaded. Loading default data from: {source_name}")
    else:
        st.warning("No uploaded file and default data path not found or not specified.")
        return None

    try:
        if source_name.endswith('.csv'):
            df = pd.read_csv(data_source)
        elif source_name.endswith('.parquet'):
            df = pd.read_parquet(data_source)
        else:
            st.error(f"Unsupported file type: {source_name}. Please use CSV or Parquet.")
            return None

        st.write("--- Debug: Raw Columns Loaded ---") # DEBUG PRINT
        st.write(list(df.columns)) # DEBUG PRINT

        # --- Clean Column Names ---
        original_columns = list(df.columns)
        df.columns = df.columns.str.strip() # Remove leading/trailing whitespace
        # Optional: Convert to lowercase for easier matching (if needed)
        # df.columns = df.columns.str.lower()
        cleaned_columns = list(df.columns)
        if original_columns != cleaned_columns:
            st.write("--- Debug: Cleaned Columns ---") # DEBUG PRINT
            st.write(cleaned_columns) # DEBUG PRINT

        # --- Validation - ADJUST REQUIRED COLUMN NAMES IF NEEDED ---
        # Define required columns EXACTLY as they should appear AFTER cleaning
        required_cols = ['y_true', 'y_pred_prob']
        optional_feature_cols = ['Age', 'total_count_payments'] # Add more if needed later

        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
             # Show the columns that WERE found to help diagnose
             st.error(f"Error: Required columns missing from data after cleaning: {missing_cols}.")
             st.error(f"Columns found in the DataFrame: {list(df.columns)}")
             return None

        # Report found columns
        all_expected = required_cols + optional_feature_cols
        found_cols = [col for col in all_expected if col in df.columns]
        st.success(f"Data loaded successfully. Required columns found. Other expected columns found: {[c for c in found_cols if c not in required_cols]}")
        return df

    except Exception as e:
        st.error(f"Error loading prediction data from {source_name}: {e}")
        return None

"""# --- Initialize io_service using Streamlit Secrets ---
# Use st.secrets for secure configuration
PROJECT_ID = 'matka-ziemia-1485115388'

@st.cache_resource # Cache the client instance for efficiency
def get_io_service():
    #Initializes and returns the BigQueryClient for GCS access.
    if not PROJECT_ID or PROJECT_ID == "your-default-project-id":
         st.error("GCP Project ID not configured in Streamlit secrets (gcp_project_id).")
         return None
    try:
        # Note: datasetid seems irrelevant for the GCS 'open' method, only projectid matters for auth/client
        client = BigQueryClient(projectid=PROJECT_ID)
        st.info("BigQueryClient (for GCS access) initialized.")
        return client
    except Exception as e:
        st.error(f"Failed to initialize BigQueryClient: {e}")
        return None"""

"""# --- Loading Function using Result.load ---
@st.cache_data(ttl=3600) # Cache the loaded data for an hour
def load_result_object(result_gcs_path: str):
    #Loads the Result object from GCS using the Result.load static method.
    io_service = get_io_service() # Get the cached client instance
    if io_service is None:
        st.error("IO Service (BigQueryClient) is not available for loading.")
        return None

    if not result_gcs_path or "/" not in result_gcs_path:
        st.error("Invalid GCS path format provided. Expected 'bucket_name/path/to/file.pkl'.")
        return None

    st.info(f"Attempting to load Result object from GCS: {result_gcs_path}")
    try:
        # Use the static load method from your Result class
        result_obj = Result.load(result_gcs_path, io_service) # CALL YOUR METHOD

        # Optional but recommended: Verify the loaded object type
        if not isinstance(result_obj, Result):
            st.error(f"Loaded object from {result_gcs_path} is not of the expected Result type.")
            return None

        st.success(f"Successfully loaded Result object: {getattr(result_obj, 'id', 'N/A')}") # Use 'id' if defined
        return result_obj
    except FileNotFoundError:
        # Your io_service might raise this or a GCS-specific error
        st.error(f"Result file not found at GCS path: {result_gcs_path}")
        return None
    except Exception as e:
        # Catch other potential errors (pickle, GCS access, etc.)
        st.error(f"An error occurred loading Result object from GCS: {e}")
        return None"""

"""# --- Keep extract_evaluation_data as before ---
@st.cache_data
def extract_evaluation_data(_result_obj: Result, dataset_key: str = 'pred'):
    # ... (Keep the implementation from the previous answer) ...
    # Make sure it correctly accesses attributes like _result_obj.pipeline_data.y_pred etc.
    if not _result_obj or not isinstance(_result_obj, Result):
        st.error("Invalid Result object passed to extract_evaluation_data.")
        return None, None, None

    y_true = None
    y_scores = None
    metrics = None

    try:
        # Adjust these accesses based on EXACTLY how data is stored in Result/PipelineData
        pipeline_data = getattr(_result_obj, 'pipeline_data', None)
        model_evaluation = getattr(_result_obj, 'model_evaluation', {})

        if pipeline_data is None:
             raise AttributeError("pipeline_data attribute missing from Result object.")

        if dataset_key == 'train':
            y_true = getattr(pipeline_data, 'y_train', None)
            y_scores = getattr(pipeline_data, 'y_scores_train', None)
            metrics = model_evaluation.get('train', {})
        elif dataset_key == 'test':
            y_true = getattr(pipeline_data, 'y_test', None)
            y_scores = getattr(pipeline_data, 'y_scores_test', None)
            metrics = model_evaluation.get('test', {})
        elif dataset_key == 'pred':
            y_true = getattr(pipeline_data, 'y_pred', None) # Uses 'y_pred' as per your class
            y_scores = getattr(pipeline_data, 'y_scores_pred', None)
            metrics = model_evaluation.get('pred', {})
        else:
            st.error(f"Invalid dataset_key: {dataset_key}. Use 'train', 'test', or 'pred'.")
            return None, None, None

        if y_true is None or y_scores is None:
             st.warning(f"y_true or y_scores missing for dataset '{dataset_key}' in Result object.")
             return None, None, metrics

        if len(y_true) != len(y_scores):
            st.error(f"Length mismatch for dataset '{dataset_key}'.")
            return None, None, metrics

        if not isinstance(y_true, pd.Series): y_true = pd.Series(y_true, name='y_true')
        if not isinstance(y_scores, pd.Series): y_scores = pd.Series(y_scores, name='y_scores')

        return y_true, y_scores, metrics

    except AttributeError as e:
         st.error(f"Error accessing data within Result object for key '{dataset_key}'. Structure might differ. Error: {e}")
         return None, None, None
    except Exception as e:
         st.error(f"Unexpected error extracting data for key '{dataset_key}': {e}")
         return None, None, None"""