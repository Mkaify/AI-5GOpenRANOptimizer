import pandas as pd
import numpy as np
import sys
import os
from sklearn.preprocessing import MinMaxScaler

def transform_data(input_path: str, output_path: str) -> None:
    print(f"Reading data from: {input_path}")
    # Load data from input file
    df = pd.read_csv(input_path)

    # Automatically identify numeric columns (float or integer types)
    # This automatically bypasses text columns like 'Band' or 'timestamp'
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # We shouldn't scale the tracking 'id' column if it exists
    if 'id' in numeric_cols:
        numeric_cols.remove('id')

    print(f"Automatically detected {len(numeric_cols)} numeric columns for scaling.")
    print("Applying MinMaxScaler transformations...")
    
    # Scale only the valid numeric columns safely
    scaler = MinMaxScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save transformed data to output file
    df.to_csv(output_path, index=False)
    print(f"Successfully saved transformed dataset to: {output_path}")

if __name__ == "__main__":
    input_p = "data/5g_network_data.csv"
    output_p = "data/processed_5g_data.csv"
    
    if len(sys.argv) > 2:
        input_p = sys.argv[1]
        output_p = sys.argv[2]
        
    transform_data(input_p, output_p)