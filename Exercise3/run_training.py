import pandas as pd 
import numpy as np 
from preprocessing import breast_cancer_preprocessing, Phone_Addiction_preprocessing, Student_placement_preprocessing
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, cross_validate, KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from imblearn.pipeline import Pipeline

import time
import os
import warnings

warnings.filterwarnings("ignore")



def split_dataset(df, target_column, test_set_size):

    # Takes as input a dataframe, the target variable and the split size
    # Returns the train and validation dataset 

    X = df.drop(columns=[target_column])
    y = df[target_column]


    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=test_set_size, random_state=42, shuffle=True)

    return X_train, y_train, X_valid, y_valid


def train_dataset_holdout(model, X_train, y_train, X_valid):

    #Trains and evaluates a model on validation data.

    start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start
    
    y_pred = model.predict(X_valid)
        
    return y_pred, train_time



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
        print(f"Starting modeling of dataset {keys}")

        preprocessing_class_instance = data[2]()
        
        # Split data into 80-20%
        X_train, y_train, X_valid, y_valid = split_dataset(data[0], data[1], 0.2)
        
        models = {"RandomForest": RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42),
                  "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
                  "KNN": KNeighborsClassifier(n_neighbors=5)}

        for name, model in models.items():


           #Set up pipeline with preprocessing class and model
           pipeline = make_pipeline(preprocessing_class_instance, model)
        
           print(f"Model: {name}")
           #### Cross-validation #####

           #Initiate k-fold cross validation
           print('#'*30)
           print('Results for 5-fold Cross-validation')
           kf = KFold(n_splits=5, shuffle=True, random_state=42)

           scorer = cross_val_metrics() 

           # Run cross-validation
           cv_results = cross_validate(pipeline,
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


    df_breast_cancer = pd.read_csv("data"+ os.sep + "breast-cancer-diagnostic.shuf.lrn.csv")
    df_teen_addiction = pd.read_csv("data"+ os.sep + "teen_phone_addiction_dataset.csv")
    df_student_placement = pd.read_csv("data"+ os.sep + "student_placement.csv")


    df_teen_addiction['Addiction_Level'] = df_teen_addiction['Addiction_Level'].round(0)

    # Define the dataset dictionary

    datasets = {"Breast_cancer": [df_breast_cancer, "class", breast_cancer_preprocessing],
                "Teen Addiction": [df_teen_addiction, "Addiction_Level", Phone_Addiction_preprocessing],
                "Student Placement": [df_student_placement, "Placement_Status", Student_placement_preprocessing]}

    run(datasets)