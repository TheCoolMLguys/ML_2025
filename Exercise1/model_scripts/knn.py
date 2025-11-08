import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif

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


def evaluate_dataset(y_valid, y_pred, train_time, dataset_name='default'):
    # Used weighted averaging for Loan dataset, macro for others
    average_method = 'weighted' if dataset_name == 'Loan' else 'macro'

    return {
        "accuracy": accuracy_score(y_valid, y_pred),
        "precision": precision_score(y_valid, y_pred, average=average_method, zero_division=1),
        "recall": recall_score(y_valid, y_pred, average=average_method, zero_division=1),
        "f1_score": f1_score(y_valid, y_pred, average=average_method),
        "training_time": train_time,
        "average_method": average_method
    }



def cross_val_metrics(dataset_name='default'):
    if dataset_name == 'Loan':
        return {
            "accuracy": 'accuracy',
            "precision": make_scorer(precision_score, average='weighted', zero_division=1),
            "recall": make_scorer(recall_score, average='weighted', zero_division=1),
            "f1": make_scorer(f1_score, average='weighted')
        }
    else:
        return {
            "accuracy": 'accuracy',
            "precision": make_scorer(precision_score, average='macro', zero_division=1),
            "recall": make_scorer(recall_score, average='macro', zero_division=1),
            "f1": make_scorer(f1_score, average='macro')
        }



def find_main_metric_column(results_df, dataset_name):
    if dataset_name == 'Loan':
        for col in results_df.columns:
            if 'mean_test_f1' in col and 'weighted' in col:
                return col
        for col in results_df.columns:
            if 'mean_test_f1' in col:
                return col
    else:
        for col in results_df.columns:
            if 'mean_test_f1' in col and 'macro' in col:
                return col
        for col in results_df.columns:
            if 'mean_test_f1' in col:
                return col

    for col in results_df.columns:
        if 'mean_test' in col:
            return col

    return 'mean_test_score'


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

        # Evaluate performance
        metrics = evaluate_dataset(y_valid, y_pred, train_time, dataset_name=keys)
        print(metrics)

        #### Cross-validation #####
        print('#' * 30)
        print('Results for 5-fold Cross-validation')
        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        # dataset-specific metrics for cross-validation
        scorer = cross_val_metrics(keys)

        # Run cross-validation
        cv_results = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=kf,
            scoring=scorer,
            return_train_score=True,
            n_jobs=-1)

        for metric in scorer.keys():
            print(f"Train {metric}: {np.mean(cv_results[f'train_{metric}']):.4f} ± {np.std(cv_results[f'train_{metric}']):.4f}")
            print(f"Validation {metric}: {np.mean(cv_results[f'test_{metric}']):.4f} ± {np.std(cv_results[f'test_{metric}']):.4f}")

        print(f"Mean fit time: {np.mean(cv_results['fit_time']):.3f} s ± {np.std(cv_results['fit_time']):.3f}")

        ### Perform hyper-parameter tuning ####
        print('#' * 30)
        print('Results from Grid Search')

        # weighted metrics for Loan dataset, macro for others
        if keys == 'Loan':
            scoring = {
                'acc': 'accuracy',
                'f1': 'f1_weighted',
                'precision': 'precision_weighted',
                'recall': 'recall_weighted'
            }
            refit_metric = 'f1'
        else:
            scoring = {
                'acc': 'accuracy',
                'f1': 'f1_macro',
                'precision': 'precision_macro',
                'recall': 'recall_macro'
            }
            refit_metric = 'f1'

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
            refit=refit_metric,
            error_score='raise',
            n_jobs=-1
        )
        grid_search.fit(X_train, y_train)

        print('Best parameters for model based on grid search are:')
        print(grid_search.best_params_)
        print('Best score for model based on grid search is:')
        print(grid_search.best_score_)

        results_df = pd.DataFrame(grid_search.cv_results_)
        metric_cols = [col for col in results_df.columns if 'mean_test' in col or 'std_test' in col]
        summary_df = results_df[['params'] + metric_cols]

        main_metric_col = find_main_metric_column(results_df, keys)
        print(f"Sorting results by: {main_metric_col}")
        summary_df = summary_df.sort_values(main_metric_col, ascending=False)

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

        # Evaluate performance
        metrics_best = evaluate_dataset(y_valid, y_pred_best, train_time_best, dataset_name=keys)
        print('Metrics with hold-out method after using best parameters')
        print(metrics_best)

        # Ozone - remove hourly data
        if keys == 'Ozone_level':
            cols_to_drop = []
            for i in range(24):
                cols_to_drop.append(f'WSR{i}')
                cols_to_drop.append(f'T{i}')
            df = data[0].drop(columns=cols_to_drop)
            X_train_red, y_train_red, X_valid_red, y_valid_red = split_dataset(df, data[1], 0.2)

            pipeline_red = make_pipeline(preprocessing_class_instance, KNeighborsClassifier())
            y_pred_red, train_time_red = train_dataset(pipeline_red, X_train_red, y_train_red, X_valid_red)
            metrics_red = evaluate_dataset(y_valid_red, y_pred_red, train_time_red, dataset_name=keys)

            print('Special feature reduction analysis for Ozone dataset')
            print('Metrics for dataset with feature reduction:')
            print(metrics_red)


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