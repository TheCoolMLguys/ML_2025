import pandas as pd 
import numpy as np 
from preprocessing import breast_cancer_preprocessing, Phone_Addiction_preprocessing, Student_placement_preprocessing, SaNGreeATransformer
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


    metrics = ["accuracy", "precision_macro", "recall_macro", "f1_macro", "precision_weighted", "recall_weighted", "f1_weighted", "training_time"]

    mean_results = {} 

    collected = {m: [] for m in metrics}

    for entry in metrics_list:
        for m in metrics:
            collected[m].append(entry[m])

    for m in metrics:
        values = np.array(collected[m])
        mean_results[m] = {"mean": np.mean(values), 
                           "std": np.std(values, ddof=1)}  # sample standard deviation}

    return mean_results


def run(datasets, state, anonymity = True):

    
    print(f"Anonymization: {anonymity}")

    for keys, data in datasets.items():
        
        print("#"*30)
        print(f"Starting modeling of dataset {keys}")

        preprocessing_class_instance = data[2]()  # instantiate preprocessing class
        
        # Split data into 80-20%
        X_train, y_train, X_valid, y_valid = split_dataset(data[0], data[1], 0.2)
        

        models = {"RandomForest": RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42),
                  "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
                  "KNN": KNeighborsClassifier(n_neighbors=5)}

        # Iterate over all classifiers

        for name, classifier in models.items():
       
          print(f"Training with {name}")

          kf = KFold(n_splits=5, shuffle=True, random_state=state)

          score_list = []

          for train_idx, val_idx in kf.split(X_train):

            X_train_cv, X_val_cv = X_train.iloc[train_idx].reset_index(drop=True), X_train.iloc[val_idx].reset_index(drop=True)
            y_train_cv, y_val_cv = y_train.iloc[train_idx].reset_index(drop=True), y_train.iloc[val_idx].reset_index(drop=True)

            preprocess = preprocessing_class_instance
            X_train_prep = preprocess.fit_transform(X_train_cv)
            X_val_prep   = preprocess.transform(X_val_cv)
       
            
            if anonymity:
               sangreea = SaNGreeATransformer(k=5)
               sangreea.fit(X_train_prep)
               X_train_anon = sangreea.transform(X_train_prep)
         
            else:
                X_train_anon = X_train_prep.copy()

            model = classifier

            start = time.time()
            model.fit(X_train_anon, y_train_cv)
            train_time = time.time() - start

            y_pred = model.predict(X_val_prep)
 
            scores = evaluate_dataset(y_val_cv, y_pred, train_time)
            score_list.append(scores)

          print(find_mean_values(score_list))
     

if __name__ == '__main__':


    df_breast_cancer = pd.read_csv("data"+ os.sep + "breast-cancer-diagnostic.shuf.lrn.csv")
    df_teen_addiction = pd.read_csv("data"+ os.sep + "teen_phone_addiction_dataset.csv")
    df_student_placement = pd.read_csv("data"+ os.sep + "student_placement.csv")


    df_teen_addiction['Addiction_Level'] = df_teen_addiction['Addiction_Level'].round(0)

    # Define the dataset dictionary

    datasets = {"Breast_cancer": [df_breast_cancer, "class", breast_cancer_preprocessing],
                "Teen Addiction": [df_teen_addiction, "Addiction_Level", Phone_Addiction_preprocessing],
                "Student Placement": [df_student_placement, "Placement_Status", Student_placement_preprocessing]}

    run(datasets, state=42, anonymity = False)

    run(datasets, state=42, anonymity = True)