import pandas as pd 
import numpy as np 
from Preprocessing_all import Ozone_preprocessing, Personality_type_preprocessing, breast_cancer_preprocessing, loan_preprocessing
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate, KFold, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
import time
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
    
    corr_features = df_copy.drop(columns= target).corr().abs()
    corr_with_target = df_copy.corr().abs()[target]

    high_corr_with_target = corr_with_target[(corr_with_target.abs() > 0.8)].index.tolist() 
    low_corr_with_target = corr_with_target[(corr_with_target.abs() < 0.2)].index.tolist()
    print('High correlation with Target variable: {}'.format(high_corr_with_target))
    print('Low correlation with Target variable: {}'.format(low_corr_with_target))


    upper = corr_features.where(np.triu(np.ones(corr_features.shape), k=1).astype(bool))
    print(upper)




def split_dataset(df, target_column, test_set_size):

    # Takes as input a dataframe, the target variable and the split size
    # Returns the train and validation dataset 

    X = df.drop(columns=[target_column])
    y = df[target_column]


    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=test_set_size, random_state=42, shuffle=True)

    return X_train, y_train, X_valid, y_valid


def train_dataset(model, X_train, y_train, X_valid):

    #Trains and evaluates a model on validation data.

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


def cross_val_metrics():
      
        return {
            "accuracy": 'accuracy',
            "precision_macro": make_scorer(precision_score, average='macro', zero_division=1),
            "recall_macro": make_scorer(recall_score, average='macro', zero_division=1),
            "f1_macro": make_scorer(f1_score, average='macro'),
            "precision_weighted": make_scorer(precision_score, average='weighted', zero_division=1),
            "recall_weighted": make_scorer(recall_score, average='weighted', zero_division=1),
            "f1_weighted": make_scorer(f1_score, average='weighted')
            }


def run(datasets):

    for keys, data in datasets.items():
        
        print("#"*30)
        print("#"*30)
        #print('Data Exploration')

        #data_exploration(data[0], data[1])

        print("#"*30)
        print(f"Starting modeling of dataset {keys}")

        preprocessing_class_instance = data[2]()

        # Split data into 80-20%
        X_train, y_train, X_valid, y_valid = split_dataset(data[0], data[1], 0.2)

        #Set up pipeline with preprocessing class and model
        pipeline = make_pipeline(preprocessing_class_instance, RandomForestClassifier(random_state=42))
        
        # Train model by using pipeline object with holdout method
        y_pred, train_time = train_dataset(pipeline, X_train, y_train, X_valid)
        
        #Evaluate performance
        metrics = evaluate_dataset(y_valid, y_pred, train_time)

        print(metrics)
        
        #### Cross-validation #####

        #Initiate k-fold cross validation
        print('#'*30)
        print("#"*30)
        print('Results for 5-fold Cross-validation')
        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        scorer = cross_val_metrics() 

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
        print('#'*30)
        print("#"*30)
        print('Results from Grid Search')
       
        params_grid = [{
         'randomforestclassifier__n_estimators': [50, 200, 300],
         'randomforestclassifier__max_depth': [None, 15, 30],
         'randomforestclassifier__min_samples_split': [2, 10, 0.01],
         'randomforestclassifier__min_samples_leaf': [1, 5, 30],
        }]
        scoring = {'acc': 'accuracy','f1_macro': 'f1_macro', 'precision_macro': 'precision_macro', 'recall_macro': 'recall_macro', 
                   'f1_w': 'f1_weighted', 'precision_w': 'precision_weighted', 'recall_w': 'recall_weighted'}
        
        grid_search = GridSearchCV(pipeline, params_grid, cv = 5, scoring=scoring, refit='f1_macro', error_score='raise', n_jobs=-1)
        grid_search.fit(X_train, y_train)
     

        #Show all results from Grid Search

        print('Best parameters for model based on grid search are:')
        print(grid_search.best_params_)
        print('Best score for model based on grid search is:')
        print(grid_search.best_score_)

        results_df = pd.DataFrame(grid_search.cv_results_)

        metric_cols = [col for col in results_df.columns if 'mean_' in col or 'std_' in col]
        summary_df = results_df[['params'] + metric_cols]

        summary_df = summary_df.sort_values('mean_test_f1_macro', ascending=False)


        summary_df.to_csv(f'overview_{keys}.csv', index = False)


        # Use best parameters to your model - hold out method
        print('#'*30)
        print('Feed best parameters in model')

        model_best_params = make_pipeline(preprocessing_class_instance, RandomForestClassifier(**{k.split("__")[1]: v for k, v in grid_search.best_params_.items()}))
        y_pred_best, train_time_best = train_dataset(model_best_params, X_train, y_train, X_valid)
        
        #Evaluate performance
        metrics_best = evaluate_dataset(y_valid, y_pred_best, train_time_best)
        print('Metrics with hold-out method after using best parameters')
        print(metrics_best)


        #####################################################################
        ###### Test impact of balancing

        print('#'*30)
        print("#"*30)
        print('Test impact of balancing')

        pip_balanced = Pipeline([('preprocessing',preprocessing_class_instance), ('smote', SMOTE(random_state=42)), ('model', RandomForestClassifier(random_state=42))])
        pip_balanced_best = Pipeline([('preprocessing',preprocessing_class_instance), ('smote', SMOTE(random_state=42)), ('model', RandomForestClassifier(**{k.split("__")[1]: v for k, v in grid_search.best_params_.items()}, random_state=42))])
        kf_balanced = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


        cv_results_balanced = cross_validate(
            pip_balanced,
            X_train,
            y_train,
            cv=kf_balanced,
            scoring=scorer,
            return_train_score=True,
            n_jobs=-1)

        cv_results_balanced_best = cross_validate(
            pip_balanced_best,
            X_train,
            y_train,
            cv=kf_balanced,
            scoring=scorer,
            return_train_score=True,
            n_jobs=-1)
        
        for metric_balanced in scorer.keys():
           print('Default model parameters after balancing')
           print(f"Train {metric_balanced}: {np.mean(cv_results_balanced[f'train_{metric_balanced}']):.4f} ± {np.std(cv_results_balanced[f'train_{metric_balanced}']):.4f}")
           print(f"Validation {metric_balanced}: {np.mean(cv_results_balanced[f'test_{metric_balanced}']):.4f} ± {np.std(cv_results_balanced[f'test_{metric_balanced}']):.4f}")
           print('Best model parameters after balancing')
           
           print(f"Train {metric_balanced}: {np.mean(cv_results_balanced_best[f'train_{metric_balanced}']):.4f} ± {np.std(cv_results_balanced_best[f'train_{metric_balanced}']):.4f}")
           print(f"Validation {metric_balanced}: {np.mean(cv_results_balanced_best[f'test_{metric_balanced}']):.4f} ± {np.std(cv_results_balanced_best[f'test_{metric_balanced}']):.4f}")
           print('#'*20)

        print(f"Mean fit time: {np.mean(cv_results_balanced['fit_time']):.3f} s ± {np.std(cv_results_balanced['fit_time']):.3f}")
        print(f"Mean fit time: {np.mean(cv_results_balanced_best['fit_time']):.3f} s ± {np.std(cv_results_balanced_best['fit_time']):.3f}")
        

        ###########################################################################
        ########Test impact of feature reduction 


        print('#'*30)
        print("#"*30)
        print('Test impact of feature reduction')

        pip_reduced =  make_pipeline(preprocessing_class_instance, SelectKBest(score_func=f_classif, k=30) , RandomForestClassifier(random_state=42)) 
        
        kf_reduced = KFold(n_splits=5, shuffle=True, random_state=42)


        cv_results_reduced = cross_validate(
            pip_reduced,
            X_train,
            y_train,
            cv=kf_reduced,
            scoring=scorer,
            return_train_score=True,
            n_jobs=-1)

        
        for metric_reduced in scorer.keys():
           print('Default model parameters after balancing')
           print(f"Train {metric_reduced}: {np.mean(cv_results_reduced[f'train_{metric_reduced}']):.4f} ± {np.std(cv_results_reduced[f'train_{metric_reduced}']):.4f}")
           print(f"Validation {metric_reduced}: {np.mean(cv_results_reduced[f'test_{metric_reduced}']):.4f} ± {np.std(cv_results_reduced[f'test_{metric_reduced}']):.4f}")


        print(f"Mean fit time: {np.mean(cv_results_reduced['fit_time']):.3f} s ± {np.std(cv_results_reduced['fit_time']):.3f}")


        # Ozone - remove hourly data

        if keys=='Ozone_level':
            
            cols_to_drop = []
            for i in range(24):
           
               cols_to_drop.append(f'WSR{i}')
               cols_to_drop.append(f'T{i}')
            df = data[0].drop(columns = cols_to_drop)
            X_train_red, y_train_red, X_valid_red, y_valid_red = split_dataset(df, data[1], 0.2)

            pipeline_red = make_pipeline(preprocessing_class_instance, RandomForestClassifier(random_state=42))
 
            y_pred_red, train_time_red = train_dataset(pipeline_red, X_train_red, y_train_red, X_valid_red)
            metrics_red = evaluate_dataset(y_valid_red, y_pred_red, train_time_red)
            print('Special feature reduction analysis for Ozone dataset')
            print('Metrics for dataset with feature reduction:')
            print(metrics_red)






if __name__ == '__main__':


    df_ozone = pd.read_csv("Exercise1/data"+ os.sep + "ozone_level_data.csv")
    df_personality = pd.read_csv("Exercise1/data"+ os.sep + "personality_types_data_v2.csv")
    df_breast_cancer = pd.read_csv("Exercise1/data"+ os.sep + "breast-cancer-diagnostic.shuf.lrn.csv")
    df_loan = pd.read_csv("Exercise1/data"+ os.sep + "loan-10k.lrn.csv")
    

    # Define the dataset dictionary

    datasets = {"Personality_type": [df_personality, "Personality", Personality_type_preprocessing],
                "Ozone_level": [df_ozone, "Ozone", Ozone_preprocessing],
                "Breast_cancer": [df_breast_cancer, "class", breast_cancer_preprocessing],
                "Loan": [df_loan, "grade", loan_preprocessing]}

    run(datasets)
    
        