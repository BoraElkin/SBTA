import pandas as pd
import numpy as np
import logging

def preprocess_data(df):
    """
    Clean, normalize, and structure the collected solar activity data.
    
    Parameters:
    df (pandas.DataFrame): Raw data DataFrame

    Returns:
    pandas.DataFrame: Processed data ready for training.
    """
    if df.empty:
        logging.error("Input DataFrame is empty in preprocess_data.")
        return df

    # Drop duplicates and reset index
    df = df.drop_duplicates().reset_index(drop=True)
    
    # Handle missing values: drop rows with missing essential values
    df = df.dropna()
    logging.info("Dropped missing values. Data now has %d records.", len(df))
    
    # Assume 'event' is our target variable.
    # If not present, simulate it based on an 'intensity' threshold if 'intensity' exists.
    if "event" not in df.columns:
        if "intensity" in df.columns:
            threshold = df["intensity"].mean()
            df["event"] = (df["intensity"] > threshold).astype(int)
            logging.info("Simulated 'event' column based on intensity with threshold %f.", threshold)
        else:
            logging.error("No 'event' or 'intensity' column to determine label.")
            return df

    # Normalize numeric columns (excluding target column 'event')
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if "event" in numeric_cols:
        numeric_cols.remove("event")

    for col in numeric_cols:
        min_val = df[col].min()
        max_val = df[col].max()
        if min_val != max_val:
            df[col] = (df[col] - min_val) / (max_val - min_val)
            logging.info("Normalized column '%s'.", col)
        else:
            logging.warning("Column '%s' has constant value. Skipping normalization.", col)
    
    return df 