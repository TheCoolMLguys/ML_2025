import numpy as np
import pandas as pd
import random
from collections import Counter
from sklearn.base import BaseEstimator, TransformerMixin
from joblib import Parallel, delayed
import time
from ML_Ex2_Preprocessing_all import Ford_preprocessing, House_Price_preprocessing, Phone_Addiction_preprocessing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, root_mean_squared_error
import pickle
from sklearn.model_selection import cross_val_score, KFold, ParameterGrid
from sklearn.pipeline import make_pipeline




class DecisionTree(BaseEstimator):


    def __init__(self, max_depth=None, min_criterion=0.0, min_sample_split=2, max_features=None, loss="mse", random_state=None):
        self.max_depth = np.nan if max_depth is None else max_depth
        self.min_criterion = min_criterion
        self.min_sample_split = min_sample_split
        self.max_features = max_features
        self.loss = loss
        self.random_state = random_state

    # Static methods
    def _mse(self, y):
        #Mean Squared Error

        if len(y) == 0:
            return 0
  
        return ((y - y.mean()) ** 2).mean()


    def _mae(self, y):
        # Mean Absolute Error 

        if len(y) == 0:
            return 0

        return np.mean(np.abs(y - y.mean()))


    def _mad(self, y):
        #Mean Absolute Deviation

        if len(y) == 0:
           return 0

        return np.mean(np.abs(y - np.median(y)))    

    # Fit part 
    def fit(self, X, y, depth=0):

        if not isinstance(X, pd.DataFrame): 
            X = pd.DataFrame(X)

        if not isinstance(y, pd.Series):    
         y = pd.Series(y)

        self.rng_ = np.random.default_rng(self.random_state)
        self.depth_ = depth
        self.n_samples_ = len(y)
        self.value_ = y.mean()
        impurity_dict = {"mse": self._mse, "mae": self._mae, "mad": self._mad}
        
        try:
            self.impurity_ = impurity_dict[self.loss]
        except Exception as err:
            print(f"Invalid loss function: {self.loss}. Choose from {list(impurity_dict.keys())}.")

        # Compute split 
        current_feature, current_thresh, current_gain = self._best_split(X, y)

        self.feature_ = current_feature
        self.threshold_ = current_thresh
        self.gain_ = current_gain
   
        if (depth >= self.max_depth or self.n_samples_ < self.min_sample_split or self.feature_ is None or self.gain_ < self.min_criterion):
            # Then we have reached a leaf node
            self.feature_ = None
            return self
 

        # Split data
        left_branch = X[self.feature_] <= self.threshold_
        right_branch = X[self.feature_] > self.threshold_

        # Grow tree left and right
        self.left_ = DecisionTree(
            max_depth=self.max_depth,
            min_criterion=self.min_criterion,
            min_sample_split=self.min_sample_split,
            max_features=self.max_features,
            loss=self.loss,
            random_state=self.rng_.integers(1e9),
        ).fit(X[left_branch], y[left_branch], depth + 1)

        self.right_ = DecisionTree(
            max_depth=self.max_depth,
            min_criterion=self.min_criterion,
            min_sample_split=self.min_sample_split,
            max_features=self.max_features,
            loss=self.loss,
            random_state=self.rng_.integers(1e9),
        ).fit(X[right_branch], y[right_branch], depth + 1)

        return self

    def _best_split(self, X, y):
        impurity_node = self.impurity_(y)
        best_gain, best_feature, best_thresh = 0.0, None, None

        # Feature subset
        if self.max_features:
            features = self.rng_.choice(
                X.columns,
                size=min(self.max_features, len(X.columns)),
                replace=False,
            )
        else:
            features = X.columns

        for col in features:
            values = np.sort(X[col].unique())
            if len(values) < 2:
                continue

            # take the mid-point
            thresholds = (values[:-1] + values[1:]) / 2
            col_values = X[col].values

            for thr in thresholds:
                left = col_values <= thr
                right = ~left

                if not left.any() or not right.any():
                    continue

                y_left, y_right = y[left], y[right]
                n_left = len(y_left) / len(y)

                imp_left =self.impurity_(y_left)
                imp_right = self.impurity_(y_right)
                weighted = n_left * imp_left + (1 - n_left) * imp_right

                gain = impurity_node - weighted
                if gain > best_gain:
                    best_gain = gain
                    best_feature = col
                    best_thresh = thr

        return best_feature, best_thresh, best_gain


    # Prediction part

    def _predict_single(self, row):
        if self.feature_ is None:
            return self.value_

        if row[self.feature_] <= self.threshold_:
            return self.left_._predict_single(row)
        else:
            return self.right_._predict_single(row)

    def predict(self, X):
        X = pd.DataFrame(X)
        return np.array([self._predict_single(row) for _, row in X.iterrows()])



class RandomForest(BaseEstimator):

    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2,
                 loss='mse', random_state=None, n_jobs=-1):
        
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.loss = loss  # 'mse', 'mae', 'mad'
        self.random_state = random_state
        self.n_jobs = n_jobs

        self.rng = np.random.default_rng(random_state)
        self.trees = []
    
    def _bootstrap_sample(self, X, y, tree_seed):
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have same number of samples")
        
        n_samples = X.shape[0]
        indices = tree_seed.integers(0, n_samples, size=n_samples)
        
        if isinstance(X, pd.DataFrame):
            return X.iloc[indices], y.iloc[indices]
        else:
            return X[indices], y[indices]
    
    def fit(self, X, y):
        self.trees = []

        self.trees = Parallel(n_jobs=self.n_jobs)(
            delayed(self._build_tree)(X, y, i)
            for i in range(self.n_estimators)
        )

        return self

    def _build_tree(self, X, y, tree_idx):
        tree_seed = np.random.default_rng(self.random_state + tree_idx)#self.rng.integers(0, 1e9) + tree_idx
        X_boot, y_boot = self._bootstrap_sample(X, y, tree_seed)

        max_features_val = X_boot.shape[1]  #int(np.sqrt(X_boot.shape[1]))
        
        tree = DecisionTree(
            max_depth=self.max_depth,
            min_criterion=0.0,
            min_sample_split=self.min_samples_split,
            max_features=max_features_val,
            loss=self.loss,
            random_state=tree_seed
        )
        tree.fit(X_boot, y_boot)
        return tree
    

    
    def predict(self, X):
        if not self.trees:
            raise ValueError("The tree list is empty, train model first")
        
        predictions = []
        for tree in self.trees:
            pred = tree.predict(X)
            predictions.append(pred.reshape(-1, 1))
        
        tree_predictions_mean = np.mean(np.concatenate(predictions, axis=1), axis=1)
        return tree_predictions_mean




##########################################################################
########################## Functions Definition ##########################
##########################################################################

# Hyperparameters to tune


def hyperparameter_mapping(Xtrain, y_train):


   hyperparameters = {'n_estimators': [150, 80, 50], 
                   'max_depth': [50, 30, 20],
                   'min_samples_split': [15, 10, 5]}

   default_params = {'n_estimators': 100, 'max_depth': None, 'min_samples_split': 2}


   kf = KFold(n_splits=5, shuffle=True, random_state=42) # to delete

   parameter_combinations = []

   for key, values in hyperparameters.items():
      grid = ParameterGrid({key: values})
      for g in grid:
          temp = default_params.copy()
          temp.update(g)
          parameter_combinations.append(temp)

   print("Starting hyperparameter tuning...\n")

   for params in parameter_combinations:

     print(f"Testing parameters: {params}")

     # Create and configure the RandomForest model
     model = RandomForest(**params, random_state=42)

     start_time = time.time()
     cv_tuning = cross_val_score(model, X_train, y_train, cv=kf, scoring='neg_mean_squared_error', n_jobs=-1)
     total_time = time.time() - start_time
     print(f" CV with tuning - Mean MSE: {-cv_tuning.mean():.2f}, Std Dev: {cv_tuning.std():.2f}, Total time: {total_time:.2f}")  



def train_model(X_transformed):


   X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.2, random_state=42)

   ###############################
   # Hold-out method for reference

   rf = RandomForest(n_estimators=100, max_depth=None, min_samples_split=2, random_state=42)
   start = time.time()
   rf.fit(X_train, y_train)
   total_time = time.time() - start
   print(f"Random Forest with Holdout - Total training time: {total_time}")

   predictions = rf.predict(X_test)
   mse_holdout = mean_squared_error(y_test, predictions)
   print(f"Random Forest with Holdout - MSE: {mse_holdout}")

   mae_holdout = mean_absolute_error(y_test, predictions)
   print(f"Random Forest with Holdout - MAE: {mae_holdout}")

   r2_holdout = r2_score(y_test, predictions)
   print(f"Random Forest with Holdout - R2: {r2_holdout}")

   rmse_holdout = root_mean_squared_error(y_test, predictions)
   print(f"Random Forest with Holdout - RMSE: {rmse_holdout}")

   ##########################################################


   # Cross-Validation - Our Model ########################

   kf = KFold(n_splits=5, shuffle=True, random_state=42)

   rf_cv = RandomForest(n_estimators=100, max_depth=None, min_samples_split=2, random_state=42)

   #CV with MSE
   start = time.time()
   cv_score_mse = cross_val_score(rf_cv, X_train, y_train, cv=kf, scoring='neg_mean_squared_error', n_jobs=-1)
   total_time = time.time() - start

   print(f" CV with our RF model - Mean MSE: {-cv_score_mse.mean():.2f}, Std Dev: {cv_score_mse.std():.2f}, Total time: {total_time:.2f}")

   #CV with MAE
   start = time.time()
   cv_score_mae = cross_val_score(rf_cv, X_train, y_train, cv=kf, scoring='neg_median_absolute_error', n_jobs=-1)
   total_time = time.time() - start

   print(f"CV with our RF model - Mean MAE: {-cv_score_mae.mean():.2f}, Std Dev: {cv_score_mae.std():.2f}, Total time: {total_time:.2f}")

   #CV with R2
   start = time.time()
   cv_score_r2 = cross_val_score(rf_cv, X_train, y_train, cv=kf, scoring='r2', n_jobs=-1)
   total_time = time.time() - start

   print(f"CV with our RF model - Mean R2: {cv_score_r2.mean():.2f}, Std Dev: {cv_score_r2.std():.2f}, Total time: {total_time:.2f}")

   #CV with RMSE
   start = time.time()
   cv_score_rmse = cross_val_score(rf_cv, X_train, y_train, cv=kf, scoring='neg_root_mean_squared_error', n_jobs=-1)
   total_time = time.time() - start

   print(f"CV with our RF model - Mean RMSE: {-cv_score_rmse.mean():.2f}, Std Dev: {cv_score_rmse.std():.2f}, Total time: {total_time:.2f}")


  ###########################################################
  ###########################################################   
  # Cross-Validation - Sklearn Model ########################

   rf_sklearn = RandomForestRegressor(n_estimators=100, max_depth=None, min_samples_split=2, random_state=42)

   start = time.time()
   rf_sklearn.fit(X_train, y_train)
   total_time = time.time() - start
   print(f"Sklearn Random Forest with Holdout - Total training time: {total_time}")

   predictions_sklearn = rf_sklearn.predict(X_test)
   mse_holdout_sk = mean_squared_error(y_test, predictions_sklearn)
   print(f"Sklearn Random Forest with Holdout - MSE: {mse_holdout_sk}")

   mae_holdout_sk = mean_absolute_error(y_test, predictions_sklearn)
   print(f"Sklearn Random Forest with Holdout - MAE: {mae_holdout_sk}")

   r2_holdout_sk = r2_score(y_test, predictions_sklearn)
   print(f"Sklearn Random Forest with Holdout - R2: {r2_holdout_sk}")

   rmse_holdout_sk = root_mean_squared_error(y_test, predictions_sklearn)
   print(f"Sklearn Random Forest with Holdout - RMSE: {rmse_holdout_sk}")



   kf2 = KFold(n_splits=5, shuffle=True, random_state=42)

   rf_sklearn_cv = RandomForestRegressor(n_estimators=100, max_depth=None, min_samples_split=2, random_state=42)

   #CV with MSE
   start = time.time()
   cv_sklearn_mse = cross_val_score(rf_sklearn_cv, X_train, y_train, cv=kf2, scoring='neg_mean_squared_error', n_jobs=-1)
   total_time = time.time() - start

   print(f" CV with sklearn RF model - Mean MSE: {-cv_sklearn_mse.mean():.2f}, Std Dev: {cv_sklearn_mse.std():.2f}, Total time: {total_time:.2f}")

   #CV with MAE
   start = time.time()
   cv_sklearn_mae = cross_val_score(rf_sklearn_cv, X_train, y_train, cv=kf2, scoring='neg_median_absolute_error', n_jobs=-1)
   total_time = time.time() - start

   print(f"CV with sklearn RF model - Mean MAE: {-cv_sklearn_mae.mean():.2f}, Std Dev: {cv_sklearn_mae.std():.2f}, Total time: {total_time:.2f}")

   #CV with R2
   start = time.time()
   cv_sklearn_r2 = cross_val_score(rf_sklearn_cv, X_train, y_train, cv=kf2, scoring='r2', n_jobs=-1)
   total_time = time.time() - start

   print(f"CV with sklearn RF model - Mean R2: {cv_sklearn_r2.mean():.2f}, Std Dev: {cv_sklearn_r2.std():.2f}, Total time: {total_time:.2f}")

   #CV with RMSE
   start = time.time()
   cv_sklearn_rmse = cross_val_score(rf_sklearn_cv, X_train, y_train, cv=kf2, scoring='neg_root_mean_squared_error', n_jobs=-1)
   total_time = time.time() - start

   print(f"CV with sklearn RF model - Mean RMSE: {-cv_sklearn_rmse.mean():.2f}, Std Dev: {cv_sklearn_rmse.std():.2f}, Total time: {total_time:.2f}")

   hyperparameter_mapping(Xtrain, y_train)
  




#############################################################
########################## TESTING ##########################
#############################################################


######## Teen Phone Addiction
df = pd.read_csv('data/teen_phone_addiction_dataset.csv', sep=",")

y = df['Addiction_Level']
X = df.drop(columns=['Addiction_Level'])


preprocess_phone= Phone_Addiction_preprocessing()

X_transformed = preprocess_phone.fit(X).transform(X)

print('########################')
print('Training the Random Forest model on Phone Addiction dataset')
train_model(X_transformed)


######## House Price #########

df = pd.read_csv('data/House_Price_Prediction_Dataset.csv', sep=",")

y = df['Price']
X = df.drop(columns=['Price'])


preprocess_house= House_Price_preprocessing()

X_transformed = preprocess_house.fit(X).transform(X)

print('########################')
print('Training the Random Forest model on House price dataset')
train_model(X_transformed)



###### Ford price prediction ###########

df = pd.read_csv('data/ford.csv', sep=",")

y = df['price']
X = df.drop(columns=['price'])

preprocess_ford= Ford_preprocessing()

X_transformed = preprocess_ford.fit(X).transform(X)

print('########################')
print('Training the Random Forest model on Ford dataset')
train_model(X_transformed)

