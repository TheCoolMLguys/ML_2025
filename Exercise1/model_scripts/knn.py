import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.model_selection import StratifiedKFold

from Preprocessing_all import Ozone_preprocessing, Personality_type_preprocessing, breast_cancer_preprocessing, \
    loan_preprocessing
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate, KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer
from sklearn.neighbors import KNeighborsClassifier
import time
import pickle
import os
import warnings

warnings.filterwarnings("ignore")


def data_exploration(df, target):
    df_copy = df.copy()
    if 'Date' in df_copy.columns:
        df_copy['Date'] = pd.to_datetime(df['Date'])
        df_copy = df_copy.set_index('Date')

    for col in df_copy.columns:
        df_copy[col] = pd.to_numeric(df_copy[col], errors="coerce")

    corr_features = df_copy.drop(columns=target).corr().abs()
    corr_with_target = df_copy.corr().abs()[target]

    high_corr_with_target = corr_with_target[(corr_with_target.abs() > 0.8)].index.tolist()
    low_corr_with_target = corr_with_target[(corr_with_target.abs() < 0.2)].index.tolist()
    print('High correlation with Target variable: {}'.format(high_corr_with_target))
    print('Low correlation with Target variable: {}'.format(low_corr_with_target))

    upper = corr_features.where(np.triu(np.ones(corr_features.shape), k=1).astype(bool))
    print(upper)


def split_dataset(df, target_column, test_set_size):
    X = df.drop(columns=[target_column])
    y = df[target_column]

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=test_set_size, random_state=42)

    return X_train, y_train, X_valid, y_valid


def train_dataset(model, X_train, y_train, X_valid):
    start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start
    y_pred = model.predict(X_valid)
    return y_pred, train_time


def evaluate_dataset(y_valid, y_pred, train_time, average='macro'):
    if average == 'macro':
        return {
            "accuracy": accuracy_score(y_valid, y_pred),
            "precision": precision_score(y_valid, y_pred, average='macro', zero_division=1),
            "recall": recall_score(y_valid, y_pred, average='macro', zero_division=1),
            "f1": f1_score(y_valid, y_pred, average='macro'),
            "training_time": train_time
        }
    elif average == 'weighted':
        return {
            "accuracy": accuracy_score(y_valid, y_pred),
            "precision": precision_score(y_valid, y_pred, average='weighted', zero_division=1),
            "recall": recall_score(y_valid, y_pred, average='weighted', zero_division=1),
            "f1": f1_score(y_valid, y_pred, average='weighted'),
            "training_time": train_time
        }

def cross_val_metrics(average='macro'):
    if average == 'macro':
        return {
            "accuracy": 'accuracy',
            "precision": make_scorer(precision_score, average='macro', zero_division=1),
            "recall": make_scorer(recall_score, average='macro', zero_division=1),
            "f1": make_scorer(f1_score, average='macro')
        }
    elif average == 'weighted':
        return {
            "accuracy": 'accuracy',
            "precision": make_scorer(precision_score, average='weighted', zero_division=1),
            "recall": make_scorer(recall_score, average='weighted', zero_division=1),
            "f1": make_scorer(f1_score, average='weighted')
        }


def run(datasets):
    for keys, data in datasets.items():

        print("#" * 30)
        print('Data Exploration')
        data_exploration(data[0], data[1])

        print("#" * 30)
        print(f"Starting modeling of dataset {keys}")

        preprocessing_class_instance = data[2]()

        # Split data into 80-20%
        X_train, y_train, X_valid, y_valid = split_dataset(data[0], data[1], 0.2)

        # Different pipeline for loan dataset
        if keys == 'Loan':
            pipeline = make_pipeline(
                preprocessing_class_instance,
                SelectKBest(score_func=f_classif, k=30),  # Reduced features
                KNeighborsClassifier()
            )
        else:
            pipeline = make_pipeline(preprocessing_class_instance, KNeighborsClassifier())

        # Train model by using pipeline object with holdout method
        y_pred, train_time = train_dataset(pipeline, X_train, y_train, X_valid)

        # Evaluate performance with both macro and weighted averaging
        metrics_macro = evaluate_dataset(y_valid, y_pred, train_time, 'macro')
        metrics_weighted = evaluate_dataset(y_valid, y_pred, train_time, 'weighted')

        print('Metrics with macro averaging')
        print(metrics_macro)
        print('Metrics with weighted averaging')
        print(metrics_weighted)

        #### Cross-validation #####
        print('#' * 30)
        print('Results for 5-fold Cross-validation')
        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        scorer_macro = cross_val_metrics('macro')
        scorer_weighted = cross_val_metrics('weighted')

        # Run cross-validation with both scoring methods
        cv_results_macro = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=kf,
            scoring=scorer_macro,
            return_train_score=True,
            n_jobs=-1)

        cv_results_weighted = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=kf,
            scoring=scorer_weighted,
            return_train_score=True,
            n_jobs=-1)

        print('Metrics acquired with macro averaging')
        for metric in scorer_macro.keys():
            print(
                f"Train {metric}: {np.mean(cv_results_macro[f'train_{metric}']):.4f} ± {np.std(cv_results_macro[f'train_{metric}']):.4f}")
            print(
                f"Validation {metric}: {np.mean(cv_results_macro[f'test_{metric}']):.4f} ± {np.std(cv_results_macro[f'test_{metric}']):.4f}")

        print(
            f"Mean fit time: {np.mean(cv_results_macro['fit_time']):.3f} s ± {np.std(cv_results_macro['fit_time']):.3f}")

        print('#' * 30)
        print('Metrics acquired with weighted averaging')
        for metric in scorer_weighted.keys():
            print(
                f"Train {metric}: {np.mean(cv_results_weighted[f'train_{metric}']):.4f} ± {np.std(cv_results_weighted[f'train_{metric}']):.4f}")
            print(
                f"Validation {metric}: {np.mean(cv_results_weighted[f'test_{metric}']):.4f} ± {np.std(cv_results_weighted[f'test_{metric}']):.4f}")

        print(
            f"Mean fit time: {np.mean(cv_results_weighted['fit_time']):.3f} s ± {np.std(cv_results_weighted['fit_time']):.3f}")

        ### Perform hyper-parameter tuning ####
        print('#' * 30)
        print('Results from Grid Search')

        scoring = {'acc': 'accuracy',
                    'f1_macro': 'f1_macro',
                    'precision_macro': 'precision_macro',
                    'recall_macro': 'recall_macro',
                    'f1_w': 'f1_weighted',
                    'precision_w': 'precision_weighted',
                    'recall_w': 'recall_weighted'}

        params_grid = [{
            'kneighborsclassifier__n_neighbors': [3, 5, 7, 9, 11, 15, 21],
            'kneighborsclassifier__weights': ['uniform', 'distance'],
            'kneighborsclassifier__metric': ['euclidean', 'manhattan', 'minkowski'],
            'kneighborsclassifier__p': [1, 2]
        }]

        grid_search = GridSearchCV(
            pipeline,
            params_grid,
            cv=5,
            scoring=scoring,
            refit='f1_macro',
            error_score='raise',
            n_jobs=-1
        )
        grid_search.fit(X_train, y_train)


        # Show all results from Grid Search

        print('Best parameters for model based on grid search are:')

        print(grid_search.best_params_)
        print('Best score for model based on grid search is:')
        print(grid_search.best_score_)

        results_df = pd.DataFrame(grid_search.cv_results_)

        metric_cols = [col for col in results_df.columns if 'mean_' in col or 'std_' in col]
        summary_df = results_df[['params'] + metric_cols]

        summary_df = summary_df.sort_values('mean_test_f1_macro', ascending=False)

        summary_df.to_csv(f'overview_{keys}_KNN.csv', index=False)

        # Use best parameters to your model - hold out method
        print('#' * 30)
        print('Feed best parameters in model')

        # Different model creation for different datasets
        if keys == 'Loan':
            model_best_params = make_pipeline(
                preprocessing_class_instance,
                SelectKBest(score_func=f_classif, k=30),
                KNeighborsClassifier(**{k.split("__")[1]: v for k, v in grid_search.best_params_.items()})
            )
        else:
            model_best_params = make_pipeline(
                preprocessing_class_instance,
                KNeighborsClassifier(**{k.split("__")[1]: v for k, v in grid_search.best_params_.items()})
            )

        y_pred_best, train_time_best = train_dataset(model_best_params, X_train, y_train, X_valid)

        # Evaluate performance with both averaging methods
        metrics_best_macro = evaluate_dataset(y_valid, y_pred_best, train_time_best, 'macro')
        metrics_best_weighted = evaluate_dataset(y_valid, y_pred_best, train_time_best, 'weighted')
        print('Metrics with hold-out method after using best parameters (macro)')
        print(metrics_best_macro)
        print('Metrics with hold-out method after using best parameters (weighted)')
        print(metrics_best_weighted)

        #####################################################################
        ###### Test impact of balancing (SMOTE) - FOR ALL DATASETS
        print('#' * 30)
        print('Test impact of balancing')

        # Create pipelines with SMOTE for ALL datasets
        pip_balanced = ImbPipeline([
            ('preprocessing', preprocessing_class_instance),
            ('smote', SMOTE(random_state=42)),
            ('model', KNeighborsClassifier())
        ])

        # Create balanced pipeline with best parameters (with same structure as grid search)
        if keys == 'Loan':
            pip_balanced_best = ImbPipeline([
                ('preprocessing', preprocessing_class_instance),
                ('smote', SMOTE(random_state=42)),
                ('feature_selection', SelectKBest(score_func=f_classif, k=30)),
                ('model', KNeighborsClassifier(**{k.split("__")[1]: v for k, v in grid_search.best_params_.items()}))
            ])
        else:
            pip_balanced_best = ImbPipeline([
                ('preprocessing', preprocessing_class_instance),
                ('smote', SMOTE(random_state=42)),
                ('model', KNeighborsClassifier(**{k.split("__")[1]: v for k, v in grid_search.best_params_.items()}))
            ])

        # Use StratifiedKFold for ALL SMOTE experiments
        kf_balanced = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        # Cross-validation with balanced pipelines (using macro scoring)
        cv_results_balanced_scorer_macro = cross_validate(
            pip_balanced,
            X_train,
            y_train,
            cv=kf_balanced,
            scoring=scorer_macro,
            return_train_score=True,
            n_jobs=-1)

        cv_results_balanced_best_scorer_macro = cross_validate(
            pip_balanced_best,
            X_train,
            y_train,
            cv=kf_balanced,
            scoring=scorer_macro,
            return_train_score=True,
            n_jobs=-1)

        cv_results_balanced_scorer_weighted = cross_validate(
            pip_balanced,
            X_train,
            y_train,
            cv=kf_balanced,
            scoring=scorer_weighted,
            return_train_score=True,
            n_jobs=-1)

        cv_results_balanced_best_scorer_weighted = cross_validate(
            pip_balanced_best,
            X_train,
            y_train,
            cv=kf_balanced,
            scoring=scorer_weighted,
            return_train_score=True,
            n_jobs=-1)

        for metric in scorer_macro.keys():
            print('Default model parameters after balancing (macro)')
            print(
                f"Train (macro) {metric}: {np.mean(cv_results_balanced_scorer_macro[f'train_{metric}']):.4f} ± {np.std(cv_results_balanced_scorer_macro[f'train_{metric}']):.4f}")
            print(
                f"Validation (macro) {metric}: {np.mean(cv_results_balanced_scorer_macro[f'test_{metric}']):.4f} ± {np.std(cv_results_balanced_scorer_macro[f'test_{metric}']):.4f}")
            print('Best model parameters after balancing (macro)')
            print(
                f"Train {metric}: (macro) {np.mean(cv_results_balanced_best_scorer_macro[f'train_{metric}']):.4f} ± {np.std(cv_results_balanced_best_scorer_macro[f'train_{metric}']):.4f}")
            print(
                f"Validation (macro) {metric}: {np.mean(cv_results_balanced_best_scorer_macro[f'test_{metric}']):.4f} ± {np.std(cv_results_balanced_best_scorer_macro[f'test_{metric}']):.4f}")
            print('#' * 20)

        print(
            f"Mean fit time (default) (macro) : {np.mean(cv_results_balanced_scorer_macro['fit_time']):.3f} s ± {np.std(cv_results_balanced_scorer_macro['fit_time']):.3f}")
        print(
            f"Mean fit time (best) (macro) : {np.mean(cv_results_balanced_best_scorer_macro['fit_time']):.3f} s ± {np.std(cv_results_balanced_best_scorer_macro['fit_time']):.3f}")


        for metric in scorer_weighted.keys():
            print('Default model parameters after balancing (weighted)')
            print(
                f"Train (weighted) {metric}: {np.mean(cv_results_balanced_scorer_weighted[f'train_{metric}']):.4f} ± {np.std(cv_results_balanced_scorer_weighted[f'train_{metric}']):.4f}")
            print(
                f"Validation (weighted) {metric}: {np.mean(cv_results_balanced_scorer_weighted[f'test_{metric}']):.4f} ± {np.std(cv_results_balanced_scorer_weighted[f'test_{metric}']):.4f}")
            print('Best model parameters after balancing (weighted)')
            print(
                f"Train (weighted){metric}: {np.mean(cv_results_balanced_best_scorer_weighted[f'train_{metric}']):.4f} ± {np.std(cv_results_balanced_best_scorer_weighted[f'train_{metric}']):.4f}")
            print(
                f"Validation (weighted){metric}: {np.mean(cv_results_balanced_best_scorer_weighted[f'test_{metric}']):.4f} ± {np.std(cv_results_balanced_best_scorer_weighted[f'test_{metric}']):.4f}")
            print('#' * 20)

        print(
            f"Mean fit time (default) (weighted): {np.mean(cv_results_balanced_scorer_weighted['fit_time']):.3f} s ± {np.std(cv_results_balanced_scorer_weighted['fit_time']):.3f}")
        print(
            f"Mean fit time (best) (weighted): {np.mean(cv_results_balanced_best_scorer_weighted['fit_time']):.3f} s ± {np.std(cv_results_balanced_best_scorer_weighted['fit_time']):.3f}")



        ###########################################################################
        ######## Test impact of feature reduction (for ALL datasets)
        print('#' * 30)
        print('Test impact of feature reduction')

        pip_reduced = make_pipeline(
            preprocessing_class_instance,
            SelectKBest(score_func=f_classif, k=30),
            KNeighborsClassifier()
        )

        kf_reduced = KFold(n_splits=5, shuffle=True, random_state=42)

        cv_results_reduced_macro = cross_validate(
            pip_reduced,
            X_train,
            y_train,
            cv=kf_reduced,
            scoring=scorer_macro,
            return_train_score=True,
            n_jobs=-1)

        cv_results_reduced_weighted = cross_validate(
            pip_reduced,
            X_train,
            y_train,
            cv=kf_reduced,
            scoring=scorer_weighted,
            return_train_score=True,
            n_jobs=-1)

        for metric in scorer_macro.keys():
            print(f"Train (macro) {metric}: {np.mean(cv_results_reduced_macro[f'train_{metric}']):.4f} ± {np.std(cv_results_reduced_macro[f'train_{metric}']):.4f}")
            print(f"Validation (macro) {metric}: {np.mean(cv_results_reduced_macro[f'test_{metric}']):.4f} ± {np.std(cv_results_reduced_macro[f'test_{metric}']):.4f}")

        print(f"Mean fit time (macro): {np.mean(cv_results_reduced_macro['fit_time']):.3f} s ± {np.std(cv_results_reduced_macro['fit_time']):.3f}")

        for metric in scorer_weighted.keys():
            print(f"Train (weighted) {metric}: {np.mean(cv_results_reduced_weighted[f'train_{metric}']):.4f} ± {np.std(cv_results_reduced_weighted[f'train_{metric}']):.4f}")
            print(f"Validation (weighted) {metric}: {np.mean(cv_results_reduced_weighted[f'test_{metric}']):.4f} ± {np.std(cv_results_reduced_weighted[f'test_{metric}']):.4f}")

        print(f"Mean fit time (weighted): {np.mean(cv_results_reduced_weighted['fit_time']):.3f} s ± {np.std(cv_results_reduced_weighted['fit_time']):.3f}")

        # Ozone - remove hourly data (special case - same as Logistic Regression)
        if keys == 'Ozone_level':
            cols_to_drop = []
            for i in range(24):
                cols_to_drop.append(f'WSR{i}')
                cols_to_drop.append(f'T{i}')
            df = data[0].drop(columns=cols_to_drop)
            X_train_red, y_train_red, X_valid_red, y_valid_red = split_dataset(df, data[1], 0.2)

            pipeline_red = make_pipeline(preprocessing_class_instance, KNeighborsClassifier())
            y_pred_red, train_time_red = train_dataset(pipeline_red, X_train_red, y_train_red, X_valid_red)
            metrics_red_macro = evaluate_dataset(y_valid_red, y_pred_red, train_time_red, 'macro')
            metrics_red_weighted = evaluate_dataset(y_valid_red, y_pred_red, train_time_red, 'weighted')

            print('Special feature reduction analysis for Ozone dataset')
            print('Metrics for dataset with feature reduction (macro):')
            print(metrics_red_macro)
            print('Metrics for dataset with feature reduction (weighted):')
            print(metrics_red_weighted)


if __name__ == '__main__':
    df_ozone = pd.read_csv("Exercise1/data" + os.sep + "ozone_level_data.csv")
    df_personality = pd.read_csv("Exercise1/data" + os.sep + "personality_types_data_v2.csv")
    df_breast_cancer = pd.read_csv("Exercise1/data" + os.sep + "breast-cancer-diagnostic.shuf.lrn.csv")
    df_loan = pd.read_csv("Exercise1/data" + os.sep + "loan-10k.lrn.csv")

    datasets = {
        "Personality_type": [df_personality, "Personality", Personality_type_preprocessing],
        "Ozone_level": [df_ozone, "Ozone", Ozone_preprocessing],
        "Breast_cancer": [df_breast_cancer, "class", breast_cancer_preprocessing],
        "Loan": [df_loan, "grade", loan_preprocessing]
    }

    run(datasets)