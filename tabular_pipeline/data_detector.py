import pandas as pd
import numpy as np
from typing import Dict, List, Any

class DataDetector:
    """
    Intelligently inspects a DataFrame to categorize columns by their data roles:
    numerical, categorical (low/high cardinality), datetime, text, and single-value.
    """
    def __init__(self, high_cardinality_threshold: int = 50, text_len_threshold: int = 100):
        self.high_cardinality_threshold = high_cardinality_threshold
        self.text_len_threshold = text_len_threshold

    def inspect(self, df: pd.DataFrame, exclude_cols: List[str] = None) -> Dict[str, Any]:
        exclude_cols = set(exclude_cols or [])
        candidate_cols = [c for c in df.columns if c not in exclude_cols]

        numeric_cols = []
        low_cat_cols = []
        high_cat_cols = []
        datetime_cols = []
        text_cols = []
        constant_cols = []
        missing_stats = {}

        for col in candidate_cols:
            series = df[col]
            missing_rate = series.isnull().mean()
            missing_stats[col] = float(missing_rate)
            unique_cnt = series.nunique(dropna=True)

            if unique_cnt <= 1:
                constant_cols.append(col)
                continue

            # Datetime check
            if pd.api.types.is_datetime64_any_dtype(series):
                datetime_cols.append(col)
                continue

            # Check if boolean
            if pd.api.types.is_bool_dtype(series):
                low_cat_cols.append(col)
                continue

            # Numeric check
            if pd.api.types.is_numeric_dtype(series):
                # If numeric but only 2 unique values and looks like boolean/binary
                if unique_cnt == 2 and set(series.dropna().unique()).issubset({0, 1, 0.0, 1.0}):
                    low_cat_cols.append(col)
                else:
                    numeric_cols.append(col)
                continue

            # Check for strings/objects/categoricals
            non_null_samples = series.dropna().astype(str).head(100)
            avg_str_len = non_null_samples.str.len().mean() if len(non_null_samples) > 0 else 0

            # If it's a long string, treat as potential free text
            if avg_str_len > self.text_len_threshold:
                text_cols.append(col)
            elif unique_cnt <= self.high_cardinality_threshold:
                low_cat_cols.append(col)
            else:
                high_cat_cols.append(col)

        return {
            "numeric": numeric_cols,
            "low_cardinality_cat": low_cat_cols,
            "high_cardinality_cat": high_cat_cols,
            "all_cat": low_cat_cols + high_cat_cols,
            "datetime": datetime_cols,
            "text": text_cols,
            "constant": constant_cols,
            "missing_stats": missing_stats,
        }
