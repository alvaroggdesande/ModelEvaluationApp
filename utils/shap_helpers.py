# utils/shap_helpers.py
import streamlit as st
import pandas as pd
import numpy as np
import json
import re
import warnings
from typing import Optional, List, Tuple, Dict

@st.cache_data(ttl=3600) # Cache loaded SHAP data
def load_shap_data(
    shap_values_file,
    feature_names_file,
    base_value_file,
    processed_data_file=None # Optional but recommended
) -> Optional[Dict]:
    """Loads SHAP components from uploaded files."""
    try:
        # Load SHAP Values (.npy or .csv)
        if shap_values_file.name.endswith('.npy'):
            shap_values = np.load(shap_values_file)
        elif shap_values_file.name.endswith('.csv'):
            shap_values = pd.read_csv(shap_values_file).values # Assume CSV has no header/index for values
        else:
            st.error("Unsupported SHAP values file format. Use .npy or .csv.")
            return None

        # Load Feature Names (.json list or .txt one per line)
        if feature_names_file.name.endswith('.json'):
            feature_names = json.load(feature_names_file)
            if not isinstance(feature_names, list):
                st.error("SHAP features JSON should contain a list of strings.")
                return None
        elif feature_names_file.name.endswith('.txt'):
            feature_names = [line.strip() for line in feature_names_file.read().decode().splitlines() if line.strip()]
        else:
            st.error("Unsupported SHAP features file format. Use .json or .txt.")
            return None

        # Load Base Value (.json dict or .txt single number)
        if base_value_file.name.endswith('.json'):
            base_value_data = json.load(base_value_file)
            if 'base_value' not in base_value_data:
                st.error("SHAP base value JSON must contain a 'base_value' key.")
                return None
            base_value = float(base_value_data['base_value'])
        elif base_value_file.name.endswith('.txt'):
            try:
                base_value = float(base_value_file.read().decode().strip())
            except ValueError:
                st.error("SHAP base value TXT file should contain a single number.")
                return None
        else:
            st.error("Unsupported SHAP base value file format. Use .json or .txt.")
            return None

        # Load Processed Data (Optional, .parquet or .csv)
        processed_data = None
        if processed_data_file:
            if processed_data_file.name.endswith('.parquet'):
                processed_data = pd.read_parquet(processed_data_file)
            elif processed_data_file.name.endswith('.csv'):
                processed_data = pd.read_csv(processed_data_file) # Assume header exists
            else:
                st.warning("Unsupported processed data format. Use .parquet or .csv. Proceeding without it.")

        # --- Validation ---
        n_samples_shap, n_features_shap = shap_values.shape

        if len(feature_names) != n_features_shap:
            st.error(f"Mismatch: SHAP values have {n_features_shap} features, but found {len(feature_names)} feature names.")
            return None

        if processed_data is not None:
            if processed_data.shape[0] != n_samples_shap:
                 st.error(f"Mismatch: SHAP values have {n_samples_shap} samples, but processed data has {processed_data.shape[0]}.")
                 return None
            if processed_data.shape[1] != n_features_shap:
                 st.error(f"Mismatch: SHAP values have {n_features_shap} features, but processed data has {processed_data.shape[1]}.")
                 # Try to assign columns anyway? Or fail? Let's fail for safety.
                 return None
            # Assign column names for consistency if loaded from numpy/basic csv
            processed_data.columns = feature_names

        st.success(f"SHAP data loaded: {n_samples_shap} samples, {n_features_shap} features.")
        return {
            "shap_values": shap_values,
            "feature_names": feature_names,
            "base_value": base_value,
            "processed_data": processed_data # Will be None if not loaded/valid
        }

    except Exception as e:
        st.error(f"Error loading SHAP data: {e}")
        return None

@st.cache_data
def calculate_global_shap_importance(shap_values: np.ndarray, feature_names: List[str]) -> pd.DataFrame:
    """Calculates mean absolute SHAP value for each feature."""
    if shap_values is None or not feature_names:
        return pd.DataFrame(columns=['feature', 'feature_importance'])
    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'feature_importance': mean_abs_shap
    })
    importance_df = importance_df.sort_values('feature_importance', ascending=False).reset_index(drop=True)
    return importance_df

# --- OHE Mapping Helpers (Adapted from user's code) ---

def map_feature_to_original(processed_feature_name: str, original_cols: List[str]) -> str:
    """
    Maps a processed feature name (potentially OHE) back to its original column name.
    Relies on naming convention like 'OriginalCol_Category'.
    """
    # Simple prefix check first (covers many cases)
    for col in original_cols:
        if processed_feature_name.startswith(col + "_"): # Check common delimiter
             return col
        # Handle cases where OHE doesn't add a delimiter (e.g., category becomes part of name)
        # This is trickier and depends heavily on the preprocessor.
        # Example: If 'Type' becomes 'TypeA', 'TypeB'
        # if processed_feature_name.startswith(col) and processed_feature_name != col:
             # Potential match, but needs careful validation based on YOUR specific OHE
             # return col # Uncomment and adapt cautiously

    # If no prefix matches, assume it's an original feature (numeric or non-OHE)
    # or a feature generated differently (like missing indicators if not handled below)
    if processed_feature_name in original_cols:
        return processed_feature_name

    # Fallback: return the processed name if no mapping found
    # warnings.warn(f"Could not map '{processed_feature_name}' to an original column.")
    return processed_feature_name

def deduplicate_features(feature_list: List[str], exclude_prefixes: List[str] = ["missingindicator_"]) -> List[str]:
    """Removes duplicates and features starting with excluded prefixes."""
    seen = set()
    result = []
    for feat in feature_list:
        if any(feat.startswith(prefix) for prefix in exclude_prefixes):
            continue
        if feat not in seen:
            seen.add(feat)
            result.append(feat)
    return result

@st.cache_data
def aggregate_shap_importance(
    raw_importance_df: pd.DataFrame,
    original_feature_names: List[str]
) -> pd.DataFrame:
    """
    Aggregates SHAP importance from processed features back to original features.
    """
    if raw_importance_df.empty or not original_feature_names:
        return pd.DataFrame(columns=['original_feature', 'aggregated_importance'])

    temp_df = raw_importance_df.copy()
    # Map each processed feature back to its original source
    temp_df['original_feature'] = temp_df['feature'].apply(
        lambda x: map_feature_to_original(x, original_feature_names)
    )

    # Group by the original feature name and sum the importance
    aggregated_df = temp_df.groupby('original_feature')['feature_importance'].sum().reset_index()
    aggregated_df = aggregated_df.rename(columns={'feature_importance': 'aggregated_importance'})

    # Sort by aggregated importance
    aggregated_df = aggregated_df.sort_values('aggregated_importance', ascending=False).reset_index(drop=True)
    return aggregated_df