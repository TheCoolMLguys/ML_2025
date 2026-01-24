import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def split_range_column(series):
    min_vals = []
    max_vals = []

    for val in series:
        if pd.isna(val):
            min_vals.append(np.nan)
            max_vals.append(np.nan)
        elif '~' in str(val):
            try:
                parts = str(val).split('~')
                if len(parts) == 2:
                    min_vals.append(float(parts[0]))
                    max_vals.append(float(parts[1]))
                else:
                    min_vals.append(float(parts[0]))
                    max_vals.append(float(parts[-1]))
            except:
                min_vals.append(np.nan)
                max_vals.append(np.nan)
        else:
            try:
                num_val = float(val)
                min_vals.append(num_val)
                max_vals.append(num_val)
            except:
                min_vals.append(np.nan)
                max_vals.append(np.nan)

    return pd.DataFrame({
        f'{series.name}_min': min_vals,
        f'{series.name}_max': max_vals
    })


def process_categorical_column(series):
    processed = []
    for val in series:
        if pd.isna(val):
            processed.append([])
        elif '~' in str(val):
            categories = [cat.strip() for cat in str(val).split('~')]
            processed.append(categories)
        else:
            processed.append([str(val).strip()])
    return processed


def preprocess_student_placement(df):
    range_cols = ['Age', 'CGPA', 'Internships', 'Projects', 'Coding_Skills',
                  'Communication_Skills', 'Aptitude_Test_Score',
                  'Soft_Skills_Rating', 'Certifications', 'Backlogs']
    cat_cols = ['Gender', 'Degree', 'Branch']

    X = df.drop(columns=['Placement_Status'])
    processed_dfs = []
    num_cols = []

    for col in X.columns:
        if col in range_cols:
            split_df = split_range_column(X[col])
            processed_dfs.append(split_df)
            num_cols.extend(split_df.columns.tolist())
        elif col in cat_cols:
            processed_series = pd.Series(process_categorical_column(X[col]),
                                         index=X.index, name=col)
            processed_dfs.append(pd.DataFrame({col: processed_series}))

    X_processed = pd.concat(processed_dfs, axis=1)

    if cat_cols:
        categorical_data = []
        for col in cat_cols:
            col_data = X_processed[col].apply(lambda x: '|'.join(x) if x else 'missing')
            categorical_data.append(col_data)

        categorical_df = pd.concat(categorical_data, axis=1)
        onehot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

        onehot_encoded = onehot_encoder.fit_transform(categorical_df)
        onehot_df = pd.DataFrame(onehot_encoded,
                                 columns=onehot_encoder.get_feature_names_out(cat_cols),
                                 index=X_processed.index)

        X_processed = X_processed.drop(columns=cat_cols)
        X_processed = pd.concat([X_processed, onehot_df], axis=1)

    scaler = StandardScaler()
    X_processed[num_cols] = scaler.fit_transform(X_processed[num_cols])

    return X_processed


def preprocess_breast_cancer(df):
    X = df.drop(columns=['class'])
    processed_dfs = []
    num_cols = []

    for col in X.columns:
        split_df = split_range_column(X[col])
        processed_dfs.append(split_df)
        num_cols.extend(split_df.columns.tolist())

    X_processed = pd.concat(processed_dfs, axis=1)

    scaler = StandardScaler()
    X_processed[num_cols] = scaler.fit_transform(X_processed[num_cols])

    return X_processed


def preprocess_personality(df):
    range_cols = ['Age', 'Introversion Score', 'Sensing Score',
                  'Thinking Score', 'Judging Score']
    cat_cols = ['Gender', 'Education', 'Interest']

    X = df.drop(columns=['Personality'])
    processed_dfs = []
    num_cols = []

    for col in X.columns:
        if col in range_cols:
            split_df = split_range_column(X[col])
            processed_dfs.append(split_df)
            num_cols.extend(split_df.columns.tolist())
        elif col in cat_cols:
            processed_series = pd.Series(process_categorical_column(X[col]),
                                         index=X.index, name=col)
            processed_dfs.append(pd.DataFrame({col: processed_series}))

    X_processed = pd.concat(processed_dfs, axis=1)

    if cat_cols:
        categorical_data = []
        for col in cat_cols:
            col_data = X_processed[col].apply(lambda x: '|'.join(x) if x else 'missing')
            categorical_data.append(col_data)

        categorical_df = pd.concat(categorical_data, axis=1)
        onehot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

        onehot_encoded = onehot_encoder.fit_transform(categorical_df)
        onehot_df = pd.DataFrame(onehot_encoded,
                                 columns=onehot_encoder.get_feature_names_out(cat_cols),
                                 index=X_processed.index)

        X_processed = X_processed.drop(columns=cat_cols)
        X_processed = pd.concat([X_processed, onehot_df], axis=1)

    scaler = StandardScaler()
    X_processed[num_cols] = scaler.fit_transform(X_processed[num_cols])

    return X_processed


def main():
    input_dir = 'Exercise3/anonymized_results_global'
    output_dir = 'Exercise3/preprocessed_global_anonymized_datasets'

    os.makedirs(output_dir, exist_ok=True)

    datasets = [
        ('student_placement', 'student_placement_global_k5.csv', preprocess_student_placement),
        ('breast_cancer', 'breast_cancer_global_k5.csv', preprocess_breast_cancer),
        ('personality', 'personality_global_k5.csv', preprocess_personality)
    ]

    for dataset_name, filename, preprocess_func in datasets:
        file_path = os.path.join(input_dir, filename)

        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue

        print(f"Processing {dataset_name}...")

        df = pd.read_csv(file_path, delimiter=';')
        X_processed = preprocess_func(df)

        output_path = os.path.join(output_dir, f"{dataset_name}_processed_features.csv")
        X_processed.to_csv(output_path, index=False)

        print(f"  Saved to: {output_path}")
        print(f"  Shape: {X_processed.shape}")

    print("\nDone!")


if __name__ == '__main__':
    main()