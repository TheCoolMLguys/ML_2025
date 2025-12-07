import pandas as pd
from ML_Ex2_Preprocessing_all import House_Price_preprocessing, Phone_Addiction_preprocessing, Health_preprocessing, Ford_preprocessing
from sklearn.model_selection import train_test_split, cross_validate, KFold
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import make_scorer, mean_squared_error, mean_absolute_error, r2_score
import time
import os
import numpy as np

def split_dataset(df, target_column, test_set_size):

    # Takes as input a dataframe, the target variable and the split size
    # Returns the train and validation dataset 

    X = df.drop(columns=[target_column])
    y = df[target_column]

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=test_set_size, random_state=42) 

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
        "MSE": mean_squared_error(y_valid, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_valid, y_pred)),
        "MAE": mean_absolute_error(y_valid, y_pred),
        "R2": r2_score(y_valid, y_pred),
        "training_time": train_time
    }

def cross_val_metrics():

    return {
        "MSE": make_scorer(lambda y_true, y_pred: mean_squared_error(y_true, y_pred)),
        "RMSE": make_scorer(lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": make_scorer(mean_absolute_error),
        "R2": "r2" 
    }

def run(datasets):

    for keys, data in datasets.items():

        global preprocessing_class_instance
        preprocessing_class_instance = data[2]()

        print("#"*50)
        print(f"Gradient Boosting Regressor")
        print("#"*50)

        print("#"*30)
        print(f"HOLD OUT on {keys}")
        print("#"*30)

        # Split data into 80-20%
        X_train, y_train, X_valid, y_valid = split_dataset(data[0], data[1], 0.2)

        print("Before preprocessing:", data[0].shape)
        Xt = preprocessing_class_instance.fit_transform(X_train)
        print("After preprocessing:", Xt.shape)

        #Set up pipeline with preprocessing class and model
        pipeline = make_pipeline(preprocessing_class_instance, LinearRegression()) 
 
        # Train model by using pipeline object with holdout method
        y_pred, train_time = train_dataset(pipeline, X_train, y_train, X_valid)
    
        #Evaluate performance
        metrics = evaluate_dataset(y_valid, y_pred, train_time)
    
        print({k: round(v, 3) for k, v in metrics.items()})

        print("#"*30)
        print(f"CROSS VALIDATION on {keys}")
        print("#"*30)

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


        print("#"*50)
        print(f"Sklearn RF Regression")
        print("#"*50)

        print("#"*30)
        print(f"HOLD OUT on {keys}")
        print("#"*30)

        pipeline = make_pipeline(preprocessing_class_instance, RandomForestRegressor(max_depth=10, random_state=42))
        
        # Train model by using pipeline object with holdout method
        y_pred, train_time = train_dataset(pipeline, X_train, y_train, X_valid)
        
        #Evaluate performance
        metrics = evaluate_dataset(y_valid, y_pred, train_time)

        print({k: round(v, 3) for k, v in metrics.items()})
        
        print("#"*30)
        print(f"CROSS VALIDATION on {keys}")
        print("#"*30)

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

if __name__ == '__main__':


    df_houseprice = pd.read_csv("Exercise2/data"+ os.sep + "House_Price_Prediction_Dataset.csv")
    df_phone_addiction = pd.read_csv("Exercise2/data" + os.sep + "teen_phone_addiction_dataset.csv")
    df_health = pd.read_csv("Exercise2/data" + os.sep + "health_lifestyle_dataset.csv")
    df_ford = pd.read_csv("Exercise2/data" + os.sep + "ford.csv")

    # Define the dataset dictionary

    datasets = {"House_Price": [df_houseprice, "Price", House_Price_preprocessing],
                "Phone_Addiction": [df_phone_addiction, "Addiction_Level", Phone_Addiction_preprocessing],
                #"Health": [df_health, "cholesterol", Health_preprocessing],
                "Ford": [df_ford, "price", Ford_preprocessing]}

    run(datasets)

