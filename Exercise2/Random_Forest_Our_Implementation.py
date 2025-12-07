import numpy as np
import pandas as pd
import random
from collections import Counter
from sklearn.base import BaseEstimator, TransformerMixin
from joblib import Parallel, delayed
import time
from ML_Ex2_Preprocessing_all import Phone_Addiction_preprocessing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import pickle
from sklearn.model_selection import cross_val_score, KFold
from sklearn.pipeline import make_pipeline




class DecisionTree(BaseEstimator):


    def __init__(self, max_depth=4, min_criterion=0.05, min_sample_split=2, max_features=None, loss="mse", random_state=None):
        self.max_depth = max_depth
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
    
    def _bootstrap_sample(self, X, y):
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have same number of samples")
        
        n_samples = X.shape[0]
        indices = self.rng.integers(0, n_samples, size=n_samples)
        
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
        tree_seed = self.rng.integers(0, 1e9)# + tree_idx
        X_boot, y_boot = self._bootstrap_sample(X, y)

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




#############################################################
########################## TESTING ##########################
#############################################################

df = pd.read_csv('teen_phone_addiction_dataset.csv', sep=",")

y = df['Addiction_Level']
X = df.drop(columns=['Addiction_Level'])


preprocess= Phone_Addiction_preprocessing()

X_transformed = preprocess.fit(X).transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.2, random_state=42)


kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Perform cross-validation


tree = RandomForest(n_estimators=50, max_depth=30, min_samples_split=30, random_state=42)
scores = cross_val_score(tree, X_train, y_train, cv=kf, scoring='neg_mean_squared_error')
print(f"Mean MSE: {-scores.mean():.2f}, Std Dev: {scores.std():.2f}")

#
# if you want to load the tree instance from the pickle file 
# f = open(filename, 'rb')
# tree = pickle.load(f)
# f.close()


sklearn_tree = RandomForestRegressor(n_estimators=50, max_depth=30, min_samples_split=30, random_state=42)

scores_sk = cross_val_score(sklearn_tree, X_train, y_train, cv=kf, scoring='neg_mean_squared_error')
print(f"Mean MSE: {-scores_sk.mean():.2f}, Std Dev: {scores_sk.std():.2f}")
