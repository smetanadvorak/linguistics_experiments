import pandas as pd
from typing import List, Dict, Any
import math
import numpy as np

pt_columns = None # ["Video_name", "Stimuli_type"]

def from_csv(file_path: str, columns: List[str] = None) -> List[Dict[str, Any]]:
    """
    Load data from an Excel table into a list of dictionaries with specified columns.

    Args:
        file_path (str): Path to the Excel file
        columns (List[str], optional): List of column names to include. If None, all columns are included.

    Returns:
        List[Dict[str, Any]]: List of dictionaries where each dictionary represents a row
    """
    # Read the Excel file
    df = pd.read_csv(file_path, dtype=str)
    df = df.head(59)

    # If specific columns are requested, filter to only those columns
    if columns is not None:
        # Verify all requested columns exist
        missing_cols = set(columns) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Requested columns not found in file: {missing_cols}")
        df = df[columns]

    # Convert DataFrame to list of dictionaries
    dict_list = df.to_dict(orient='records')
    for d in dict_list:
        try:
            d['item_type'] = d['Video_name'].split('/')[1].split('.')[0]
        except:
            pass
    return dict_list

def get_item_order(filepath):
    data = from_csv(filepath, pt_columns)
    item_list = []
    for d in data:
        if 'item_type' in d and d['item_type'] is not None:
            item_list.append(d['item_type'])
    return item_list

# Example usage:
if __name__ == "__main__":
    import sys
    filepath = sys.argv[1]
    data = from_csv(filepath)

    item_order = get_item_order(filepath)
    print(' '.join(item_order))