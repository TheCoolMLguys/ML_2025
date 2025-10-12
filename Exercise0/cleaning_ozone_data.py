import pandas as pd
import numpy as np


def missing_value_strategy(csv_file_path='Exercise0/data/ozone_level_data.csv'):
    """
    Strategy: Drop rows with >15 missing values, median impute the rest
    """
    print("SIMPLE MISSING VALUE HANDLING")
    print("=" * 40)

    df = pd.read_csv(csv_file_path, na_values='?')
    print(f"Original data: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Total missing values: {df.isnull().sum().sum()}")

    # Count missing values per row
    missing_per_row = df.isnull().sum(axis=1)

    # MISSING VALUE DISTRIBUTION
    print(f"\nMISSING VALUE DISTRIBUTION:")
    print("=" * 30)

    drop_threshold = 15
    rows_to_drop = missing_per_row[missing_per_row > drop_threshold]
    rows_to_keep = missing_per_row[missing_per_row <= drop_threshold]

    print(f"Rows with >{drop_threshold} missing values: {len(rows_to_drop)}")
    print(f"Rows with ≤{drop_threshold} missing values: {len(rows_to_keep)}")

    missing_counts = missing_per_row.value_counts().sort_index()
    for count, freq in missing_counts.items():
        if freq > 0:
            percentage = (freq / len(df)) * 100
            action = "KEEP" if count <= drop_threshold else "DROP"
            print(f"  {count:2d} missing: {freq:4d} rows ({percentage:5.1f}%) - {action}")

    # Drop rows with more than 15 missing values
    df_clean = df[missing_per_row <= 15].copy()
    print(f"\nDropped {len(df) - len(df_clean)} rows with >15 missing values")

    data_preservation = len(df_clean) / len(df) * 100
    print(f"Data Preservation: {data_preservation:.1f}% of original data")
    print(f"Kept {len(df_clean)} rows")

    # Fill remaining missing values with median
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns

    missing_before = df_clean[numeric_cols].isnull().sum().sum()
    print(f"\nFilling {missing_before} missing values with median...")

    for col in numeric_cols:
        if df_clean[col].isnull().any():
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())

    output_file = 'Exercise0/data/ozone_data_cleaned.csv'
    df_clean.to_csv(output_file, index=False)
    print(f"Saved cleaned data to: {output_file}")
    print(f"Final data: {df_clean.shape[0]} rows, {df_clean.shape[1]} columns")
    print(f"Remaining missing values: {df_clean.isnull().sum().sum()}")

    return df_clean


if __name__ == "__main__":
    cleaned_data = missing_value_strategy()
    print("\nFirst 3 rows of cleaned data:")
    print(cleaned_data.head(3))

    print("\n" + "=" * 60)
    print("STRATEGY JUSTIFICATION")
    print("=" * 60)
    print("   1. Drops only severely compromised rows (>15 missing values)")
    print("   2. Preserves 85.4% of your original data")
    print("   3. Median imputation is robust to outliers")
    print("   4. Maintains dataset integrity for ozone prediction tasks")