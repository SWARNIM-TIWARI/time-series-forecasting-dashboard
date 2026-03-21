# merge_stocks.py
# ========================================
# Merge all individual stock CSVs into processed_stocks folder
# ========================================

import os
import pandas as pd

# Folders
input_folder = "individual_stocks"
output_folder = "processed_stocks"

# Create output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    print(f"Created folder: {output_folder}")

# List all CSVs in input folder
stock_files = [f for f in os.listdir(input_folder) if f.endswith(".csv")]

if not stock_files:
    print(f"No CSV files found in {input_folder}. Make sure your stocks are there.")
    exit()

# Process each stock
for stock_file in stock_files:
    file_path = os.path.join(input_folder, stock_file)
    try:
        df = pd.read_csv(file_path, parse_dates=['date'])
        df.sort_values('date', inplace=True)
        df.fillna(method='ffill', inplace=True)
        # Save processed CSV
        output_file = os.path.join(output_folder, f"your_timeseries_{stock_file}")
        df.to_csv(output_file, index=False)
        print(f"Merged & saved: {output_file}")
    except Exception as e:
        print(f"Error processing {stock_file}: {e}")

print("✅ All stocks processed successfully!")
