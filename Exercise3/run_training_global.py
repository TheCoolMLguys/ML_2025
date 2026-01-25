import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import time
import os
import warnings

warnings.filterwarnings("ignore")


def load_preprocessed_data(dataset_name):
    features_path = f'Exercise3/preprocessed_global_anonymized_datasets/{dataset_name}_processed_features.csv'
    target_path = f'Exercise3/preprocessed_global_anonymized_datasets/{dataset_name}_processed_target.csv'

    X = pd.read_csv(features_path)
    y = pd.read_csv(target_path).iloc[:, 0]

    info_path = f'Exercise3/preprocessed_global_anonymized_datasets/{dataset_name}_preprocessor_info.npy'
    if os.path.exists(info_path):
        preprocessor_info = np.load(info_path, allow_pickle=True).item()
    else:
        preprocessor_info = {
            'range_cols': [],
            'cat_cols': [],
            'num_cols': X.select_dtypes(include=[np.number]).columns.tolist(),
            'processed_columns': X.columns.tolist(),
            'target_col': 'target'
        }

    return X, y, preprocessor_info


class GlobalPreprocessor:

    def __init__(self, preprocessor_info):
        self.info = preprocessor_info
        self.scaler = StandardScaler()
        self.encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        self._is_fitted = False
        self.cat_cols_to_encode = []
        self.num_cols_to_scale = []

    def fit(self, X_train):

        self.cat_cols_to_encode = []
        for col in self.info.get('cat_cols', []):
            if col in X_train.columns:
                self.cat_cols_to_encode.append(col)

        if self.cat_cols_to_encode:
            cat_data = []
            for col in self.cat_cols_to_encode:
                col_data = X_train[col].apply(
                    lambda x: '|'.join(x) if isinstance(x, list) and x
                    else ('missing' if pd.isna(x) else str(x))
                )
                cat_data.append(col_data)

            if cat_data:
                cat_df = pd.concat(cat_data, axis=1)
                self.encoder.fit(cat_df)

        self.num_cols_to_scale = []
        for col in self.info.get('num_cols', []):
            if col in X_train.columns and pd.api.types.is_numeric_dtype(X_train[col]):
                self.num_cols_to_scale.append(col)

        for col in X_train.columns:
            if ('_min' in col or '_max' in col) and col not in self.num_cols_to_scale:
                if pd.api.types.is_numeric_dtype(X_train[col]):
                    self.num_cols_to_scale.append(col)

        if self.num_cols_to_scale:
            self.scaler.fit(X_train[self.num_cols_to_scale])

        self._is_fitted = True
        return self

    def transform(self, X):
        if not self._is_fitted:
            raise ValueError("Preprocessor must be fitted before transform")

        X_transformed = X.copy()

        if self.cat_cols_to_encode and hasattr(self.encoder, 'get_feature_names_out'):
            cat_data = []
            for col in self.cat_cols_to_encode:
                if col in X_transformed.columns:
                    col_data = X_transformed[col].apply(
                        lambda x: '|'.join(x) if isinstance(x, list) and x
                        else ('missing' if pd.isna(x) else str(x))
                    )
                    cat_data.append(col_data)

            if cat_data:
                cat_df = pd.concat(cat_data, axis=1)
                cat_encoded = self.encoder.transform(cat_df)
                cat_df_encoded = pd.DataFrame(
                    cat_encoded,
                    columns=self.encoder.get_feature_names_out(self.cat_cols_to_encode),
                    index=X_transformed.index
                )

                X_transformed = X_transformed.drop(columns=self.cat_cols_to_encode)
                X_transformed = pd.concat([X_transformed, cat_df_encoded], axis=1)

        if self.num_cols_to_scale:
            existing_num_cols = [col for col in self.num_cols_to_scale if col in X_transformed.columns]
            if existing_num_cols:
                X_transformed[existing_num_cols] = self.scaler.transform(X_transformed[existing_num_cols])

        return X_transformed


def evaluate_with_cv(X, y, model, preprocessor_info, n_splits=5, random_state=42):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    all_scores = []
    all_fit_times = []

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        print(f"  Fold {fold_idx + 1}/{n_splits}", end=" ")

        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        preprocessor = GlobalPreprocessor(preprocessor_info)
        preprocessor.fit(X_train)

        X_train_processed = preprocessor.transform(X_train)
        X_val_processed = preprocessor.transform(X_val)

        start_time = time.time()
        model.fit(X_train_processed, y_train)
        fit_time = time.time() - start_time

        y_pred = model.predict(X_val_processed)

        accuracy = accuracy_score(y_val, y_pred)
        precision_macro = precision_score(y_val, y_pred, average='macro', zero_division=1)
        recall_macro = recall_score(y_val, y_pred, average='macro', zero_division=1)
        f1_macro = f1_score(y_val, y_pred, average='macro')
        precision_weighted = precision_score(y_val, y_pred, average='weighted', zero_division=1)
        recall_weighted = recall_score(y_val, y_pred, average='weighted', zero_division=1)
        f1_weighted = f1_score(y_val, y_pred, average='weighted')

        scores = {
            "accuracy": accuracy,
            "precision_macro": precision_macro,
            "recall_macro": recall_macro,
            "f1_macro": f1_macro,
            "precision_weighted": precision_weighted,
            "recall_weighted": recall_weighted,
            "f1_weighted": f1_weighted,
            "fit_time": fit_time
        }

        all_scores.append(scores)
        all_fit_times.append(fit_time)
        print(f"- Accuracy: {accuracy:.4f}")

    return all_scores, all_fit_times


def print_results(model_name, scores_list, fit_times):
    metrics = ["accuracy", "precision_macro", "recall_macro", "f1_macro",
               "precision_weighted", "recall_weighted", "f1_weighted"]

    print(f"\n  {model_name} Results:")
    for metric in metrics:
        values = [score[metric] for score in scores_list]
        mean_val = np.mean(values)
        std_val = np.std(values, ddof=1)
        print(f"    {metric:20s}: {mean_val:.4f} (±{std_val:.4f})")

    fit_time_mean = np.mean(fit_times)
    fit_time_std = np.std(fit_times, ddof=1)
    print(f"    fit_time: {fit_time_mean:.4f}s (±{fit_time_std:.4f}s)")


def run_global_evaluation(datasets, n_splits=5, random_state=42):
    for dataset_name in datasets:
        print("\n" + "=" * 70)
        print(f"Dataset: {dataset_name}")
        print("=" * 70)

        X, y, preprocessor_info = load_preprocessed_data(dataset_name)
        print(f"  Features shape: {X.shape}")
        print(f"  Target shape: {y.shape}")
        print(f"  Preprocessor info keys: {list(preprocessor_info.keys())}")

        models = {
            "RandomForest": RandomForestClassifier(n_estimators=100, random_state=random_state),
            "LogisticRegression": LogisticRegression(max_iter=1000, random_state=random_state),
            "KNN": KNeighborsClassifier(n_neighbors=5)
        }

        for model_name, model in models.items():
            print(f"\n  Training {model_name}...")
            scores, fit_times = evaluate_with_cv(X, y, model, preprocessor_info, n_splits, random_state)
            print_results(model_name, scores, fit_times)


def run_original_data_evaluation(datasets, n_splits=5, random_state=42):
    print("\n" + "=" * 80)
    print("ORIGINAL DATA - BASELINE EVALUATION")
    print("=" * 80)

    for dataset_name in datasets:
        print("\n" + "=" * 70)
        print(f"Dataset: {dataset_name} (Original)")
        print("=" * 70)

        if dataset_name == 'student_placement':
            df = pd.read_csv("Exercise3/data/student_placement.csv")
            if 'Student_ID' in df.columns:
                df = df.drop(columns=['Student_ID'])
            X = df.drop(columns=['Placement_Status'])
            y = df['Placement_Status']

        elif dataset_name == 'breast_cancer':
            df = pd.read_csv("Exercise3/data/breast-cancer-diagnostic.shuf.lrn.csv")
            if 'ID' in df.columns:
                df = df.drop(columns=['ID'])
            X = df.drop(columns=['class'])
            y = df['class']

        elif dataset_name == 'personality':
            df = pd.read_csv("Exercise3/data/personality_types_data_v2.csv")
            X = df.drop(columns=['Personality'])
            y = df['Personality']

        print(f"  Features shape: {X.shape}")
        print(f"  Target shape: {y.shape}")

        preprocessor_info = {
            'range_cols': [],
            'cat_cols': X.select_dtypes(include=['object']).columns.tolist(),
            'num_cols': X.select_dtypes(include=[np.number]).columns.tolist(),
            'processed_columns': X.columns.tolist(),
            'target_col': 'target'
        }

        models = {
            "RandomForest": RandomForestClassifier(n_estimators=100, random_state=random_state),
            "LogisticRegression": LogisticRegression(max_iter=1000, random_state=random_state),
            "KNN": KNeighborsClassifier(n_neighbors=5)
        }

        for model_name, model in models.items():
            print(f"\n  Training {model_name}...")
            scores, fit_times = evaluate_with_cv(X, y, model, preprocessor_info, n_splits, random_state)
            print_results(model_name, scores, fit_times)


def main():
    datasets = ['student_placement', 'breast_cancer', 'personality']

    print("=" * 80)
    print("GLOBAL ANONYMIZATION - PROPER EVALUATION")
    print("=" * 80)
    print("Method: Dataset anonymized once globally, then 5-fold CV with")
    print("        preprocessing fitted on training folds only.")
    print("=" * 80)

    run_original_data_evaluation(datasets, n_splits=5, random_state=42)

    print("\n" + "=" * 80)
    print("GLOBALLY ANONYMIZED DATA - UTILITY COMPARISON")
    print("=" * 80)
    run_global_evaluation(datasets, n_splits=5, random_state=42)



if __name__ == '__main__':
    main()