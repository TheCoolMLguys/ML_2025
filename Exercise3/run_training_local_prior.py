import pandas as pd 
import numpy as np 
from preprocessing import breast_cancer_preprocessing, Personality_type_preprocessing, Student_placement_preprocessing, SaNGreeATransformer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, cross_validate, KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from imblearn.pipeline import Pipeline
from joblib import Parallel, delayed
import time
import os
import warnings

warnings.filterwarnings("ignore")



def split_dataset(X, y, target_column, test_set_size):

    # Takes as input a dataframe, the target variable and the split size
    # Returns the train and validation dataset 

    #X = df.drop(columns=[target_column])
    #y = df[target_column]


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


def run_cv_fold(
    train_idx, val_idx,
    X_train, y_train,
    preprocessing_class,
    classifier_class,
    preprocess_kwargs
):
    # Split
    X_train_cv = X_train.iloc[train_idx].reset_index(drop=True)
    X_val_cv   = X_train.iloc[val_idx].reset_index(drop=True)
    y_train_cv = y_train.iloc[train_idx].reset_index(drop=True)
    y_val_cv   = y_train.iloc[val_idx].reset_index(drop=True)

    # Fresh instances per fold (CRITICAL)
    preprocess = preprocessing_class(**preprocess_kwargs)
    model = classifier_class

    # Preprocess
    X_train_prep = preprocess.fit_transform(X_train_cv)
    X_val_prep   = preprocess.transform(X_val_cv)
    print(X_train_prep)
    # Train
    start = time.time()
    model.fit(X_train_prep, y_train_cv)
    train_time = time.time() - start

    # Predict
    y_pred = model.predict(X_val_prep)

    return evaluate_dataset(y_val_cv, y_pred, train_time)


def run(datasets, state):


    for keys, data in datasets.items():
        
      print("#"*30)
      print(f"Starting modeling of dataset {keys}")
      
      anonymity = True 

      df, target_column, preprocessing_cls = data

      X = df.drop(columns=[target_column])
      y = df[target_column]

        # Anonymize here first before training the models
      if (anonymity):

            # drop necessary columns:
            if (keys == "Breast_cancer"):
                X = X.drop(columns = ["ID"])
                X.columns = X.columns.str.strip()
                X = X.drop(X.filter(regex = "perimeter|area", axis = 1).columns, axis = 1)
            if (keys == "Student Placement"):
                X = X.drop(columns=["Student_ID"], errors="ignore")
                categorical_features = ['Gender', 'Degree', 'Branch']
            if (keys == "Personality"):
                categorical_features = ['Gender', 'Interest']

            anonymizer_class=SaNGreeATransformer(k=5)
            X = anonymizer_class.fit_transform(X)
 
      print(X)
        # Split data into 80-20%
      X_train, y_train, X_valid, y_valid = split_dataset(X, y, data[1], 0.2)
        

      models = {"RandomForest": RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42),
                  "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
                  "KNN": KNeighborsClassifier(n_neighbors=5)}

        # Iterate over all classifiers
      for anonymity in [False, True]:
        
       print(f"Anonymization: {anonymity}")

       preprocess_kwargs = {
        "anonymity": anonymity,
        "k": 5,
        "anonymizer_class":  None
    }

       for name, classifier in models.items():

           print(f"Training with {name}")

           kf = KFold(n_splits=5, shuffle=True, random_state=state)

           score_list = Parallel(n_jobs=-1)(
              delayed(run_cv_fold)(
                train_idx,
                val_idx,
                X_train,
                y_train,
                data[2],              # preprocessing CLASS
                classifier,           # classifier INSTANCE (copied safely)
                preprocess_kwargs
            )
            for train_idx, val_idx in kf.split(X_train)
        )

           print(find_mean_values(score_list))


     

if __name__ == '__main__':


    df_breast_cancer = pd.read_csv("data"+ os.sep + "breast-cancer-diagnostic.shuf.lrn.csv")
    df_personality = pd.read_csv("data"+ os.sep + "personality_types_data_v2.csv")
    df_student_placement = pd.read_csv("data"+ os.sep + "student_placement.csv")

    df_student_placement = df_student_placement.sample(frac=0.2)
    df_personality = df_personality.sample(frac=0.05)


    datasets = {"Breast_cancer": [df_breast_cancer, "class", breast_cancer_preprocessing],
                "Personality type": [df_personality, "Personality", Personality_type_preprocessing],
                "Student Placement": [df_student_placement, "Placement_Status", Student_placement_preprocessing]}


    run(datasets, state=42)