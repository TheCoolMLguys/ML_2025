import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_validate, KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import time
import os
import warnings

warnings.filterwarnings("ignore")


def load_preprocessed_data(dataset_name):
    #Loading preprocessed features and target for a dataset
    features_path = f'Exercise3/preprocessed_global_anonymized_datasets/{dataset_name}_processed_features.csv'

    X = pd.read_csv(features_path)

    anonymized_path = f'Exercise3/anonymized_results_global/{dataset_name}_global_k5.csv'

    if not os.path.exists(anonymized_path):
        raise FileNotFoundError(f"Anonymized file not found: {anonymized_path}")

    anonymized_df = pd.read_csv(anonymized_path, delimiter=';')

    if dataset_name == 'student_placement':
        y = anonymized_df['Placement_Status']
    elif dataset_name == 'breast_cancer':
        y = anonymized_df['class']
    elif dataset_name == 'personality':
        y = anonymized_df['Personality']
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return X, y


def train_dataset_holdout(model, X_train, y_train, X_valid):
    start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start

    y_pred = model.predict(X_valid)

    return y_pred, train_time


def evaluate_dataset(y_valid, y_pred, train_time):
    return {
        "accuracy": accuracy_score(y_valid, y_pred),
        "precision_macro": precision_score(y_valid, y_pred, average='macro', zero_division=1),
        "recall_macro": recall_score(y_valid, y_pred, average='macro', zero_division=1),
        "f1_macro": f1_score(y_valid, y_pred, average='macro'),
        "precision_weighted": precision_score(y_valid, y_pred, average='weighted', zero_division=1),
        "recall_weighted": recall_score(y_valid, y_pred, average='weighted', zero_division=1),
        "f1_weighted": f1_score(y_valid, y_pred, average='weighted'),
        "training_time": train_time
    }


def find_mean_values(metrics_list):
    metrics = ["accuracy", "precision_macro", "recall_macro", "f1_macro",
               "precision_weighted", "recall_weighted", "f1_weighted", "training_time"]

    mean_results = {}
    collected = {m: [] for m in metrics}

    for entry in metrics_list:
        for m in metrics:
            collected[m].append(entry[m])

    for m in metrics:
        values = np.array(collected[m])
        mean_results[m] = {
            "mean": np.mean(values),
            "std": np.std(values, ddof=1)
        }

    return mean_results


def run_global_training(datasets, state=42):
    for dataset_name in datasets:

        print("#" * 30)
        print(f"Training on globally anonymized dataset: {dataset_name}")

        try:
            X, y = load_preprocessed_data(dataset_name)
            print(f"  Features shape: {X.shape}")
            print(f"  Target shape: {y.shape}")
        except Exception as e:
            print(f"  Error loading data: {str(e)}")
            continue

        X_train, X_valid, y_train, y_valid = train_test_split(
            X, y, test_size=0.2, random_state=state, shuffle=True, stratify=y
        )

        print(f"  Train shape: {X_train.shape}")
        print(f"  Valid shape: {X_valid.shape}")

        models = {
            "RandomForest": RandomForestClassifier(n_estimators=100, max_depth=None, random_state=state),
            "LogisticRegression": LogisticRegression(max_iter=100, random_state=state),
            "KNN": KNeighborsClassifier(n_neighbors=5)
        }


        for name, classifier in models.items():
            print(f"\n  Training {name} on {dataset_name}")

            kf = KFold(n_splits=5, shuffle=True, random_state=state)
            score_list = []

            for train_idx, val_idx in kf.split(X_train):
                X_train_cv, X_val_cv = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_train_cv, y_val_cv = y_train.iloc[train_idx], y_train.iloc[val_idx]

                model = classifier

                y_pred, train_time = train_dataset_holdout(model, X_train_cv, y_train_cv, X_val_cv)
                scores = evaluate_dataset(y_val_cv, y_pred, train_time)
                score_list.append(scores)

            mean_scores = find_mean_values(score_list)
            print(f"  Results:")
            for metric, values in mean_scores.items():
                if metric != "training_time":
                    print(f"    {metric}: {values['mean']:.4f} (±{values['std']:.4f})")
                else:
                    print(f"    {metric}: {values['mean']:.2f}s (±{values['std']:.2f}s)")

        print(f"\n  Final evaluation on holdout validation set:")
        for name, classifier in models.items():
            model = classifier
            model.fit(X_train, y_train)
            y_pred = model.predict(X_valid)

            accuracy = accuracy_score(y_valid, y_pred)
            f1 = f1_score(y_valid, y_pred, average='weighted')
            print(f"    {name}: Accuracy={accuracy:.4f}, F1={f1:.4f}")


def main():
    datasets = ['student_placement', 'breast_cancer', 'personality']

    print("=" * 60)
    print("TRAINING ON GLOBALLY ANONYMIZED DATA")
    print("=" * 60)

    run_global_training(datasets, state=42)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)


if __name__ == '__main__':
    main()