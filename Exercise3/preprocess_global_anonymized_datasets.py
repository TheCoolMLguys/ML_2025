import os
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")


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
            processed.append(['missing'])
        elif '~' in str(val):
            categories = [cat.strip() for cat in str(val).split('~')]
            processed.append(categories)
        else:
            processed.append([str(val).strip()])
    return processed


def preprocess_dataset(df, dataset_name):

    if dataset_name == 'student_placement':
        range_cols = ['Age', 'CGPA', 'Internships', 'Projects', 'Coding_Skills',
                      'Communication_Skills', 'Aptitude_Test_Score',
                      'Soft_Skills_Rating', 'Certifications', 'Backlogs']
        cat_cols = ['Gender', 'Degree', 'Branch']
        target_col = 'Placement_Status'

    elif dataset_name == 'breast_cancer':
        range_cols = [col for col in df.columns if col != 'class']
        cat_cols = []
        target_col = 'class'

    elif dataset_name == 'personality':
        range_cols = ['Age', 'Introversion Score', 'Sensing Score',
                      'Thinking Score', 'Judging Score']
        cat_cols = ['Gender', 'Education', 'Interest']
        target_col = 'Personality'
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    X = df.drop(columns=[target_col])
    y = df[target_col]

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

    if processed_dfs:
        X_processed = pd.concat(processed_dfs, axis=1)
    else:
        X_processed = X.copy()

    preprocessor_info = {
        'range_cols': range_cols,
        'cat_cols': cat_cols,
        'num_cols': num_cols if num_cols else X_processed.select_dtypes(include=[np.number]).columns.tolist(),
        'processed_columns': X_processed.columns.tolist(),
        'target_col': target_col,
        'dataset_name': dataset_name
    }

    return X_processed, y, preprocessor_info


def save_preprocessed_data(X_processed, y, dataset_name, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    features_path = os.path.join(output_dir, f"{dataset_name}_processed_features.csv")
    X_processed.to_csv(features_path, index=False)

    target_path = os.path.join(output_dir, f"{dataset_name}_processed_target.csv")
    pd.DataFrame(y).to_csv(target_path, index=False)

    return features_path, target_path


def main():
    input_dir = 'Exercise3/anonymized_results_global'
    output_dir = 'Exercise3/preprocessed_global_anonymized_datasets'

    os.makedirs(output_dir, exist_ok=True)

    datasets = [
        ('student_placement', 'student_placement_global_k5.csv'),
        ('breast_cancer', 'breast_cancer_global_k5.csv'),
        ('personality', 'personality_global_k5.csv')
    ]

    for dataset_name, filename in datasets:
        file_path = os.path.join(input_dir, filename)

        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue

        print(f"Processing {dataset_name}...")

        df = pd.read_csv(file_path, delimiter=';')

        X_processed, y, preprocessor_info = preprocess_dataset(df, dataset_name)

        features_path, target_path = save_preprocessed_data(
            X_processed, y, dataset_name, output_dir
        )

        info_path = os.path.join(output_dir, f"{dataset_name}_preprocessor_info.npy")
        np.save(info_path, preprocessor_info, allow_pickle=True)

        print(f"  Features saved to: {features_path}")
        print(f"  Target saved to: {target_path}")
        print(f"  Shape: {X_processed.shape}")
        print(f"  Preprocessor info saved to: {info_path}")


if __name__ == '__main__':
    main()
