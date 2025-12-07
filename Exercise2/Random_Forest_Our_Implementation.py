import numpy as np
import pandas as pd
import random
from collections import Counter
from sklearn.base import BaseEstimator


class DecisionTree(BaseEstimator):


    def __init__(self, max_depth=4, min_criterion=0.05, min_sample_split=2, max_features=None, loss="mse", random_state=None):
        self.max_depth = max_depth
        self.min_criterion = min_criterion
        self.min_sample_split = min_sample_split
        self.max_features = max_features
        self.loss = loss
        self.random_state = random_state

    # Static methods
    def _mse(y):
        #Mean Squared Error

        if len(y) == 0:
            return 0
  
        return ((y - y.mean()) ** 2).mean()


    def _mae(y):
        # Mean Absolute Error 

        if len(y) == 0:
            return 0

        return np.mean(np.abs(y - y.mean()))


    def _mad(y):
        #Mean Absolute Deviation

        if len(y) == 0:
           return 0

        return np.mean(np.abs(y - np.median(y)))    

    # Fit part 
    def fit(self, X, y, depth=0):

        if not isinstance(X, pd.DataFrame) 
            X = pd.DataFrame(X)

        if not isinstance(y, pd.Series):    
        y = pd.Series(y)

        self.rng_ = np.random.default_rng(self.random_state)
        self.depth_ = depth
        self.n_samples_ = len(y)
        self.value_ = y.mean()
        impurity_dict = {"mse": self._mse, "mae": self._mae, "mad": self._mad}
        
        try:
            self.impurity_ = impurity_map[self.loss]
        except Exception as err:
            print(f"Invalid loss function: {self.loss}. Choose from {list(impurity_dict.keys())}.")

        # Compute split 
        current_feature, current_thresh, current_gain = self._best_split(X, y)

        if (depth >= self.max_depth or self.n_samples_ < self.min_sample_split or feature is None or gain < self.min_criterion):
            # Then we have reached a leaf node
            self.feature_ = None
            return self
 
        self.feature_ = current_feature
        self.threshold_ = current_thresh
        self.gain_ = current_gain

        # Split data
        left_branch = X[feature] <= thresh
        right_branch = X[feature] > thresh

        # Grow tree left and right
        self.left_ = DecisionTree(
            max_depth=self.max_depth,
            min_criterion=self.min_criterion,
            min_sample_split=self.min_sample_split,
            max_features=self.max_features,
            loss=self.loss,
            random_state=self.rng_.integers(1e9),
        ).fit(X[left_branc], y[left_branc], depth + 1)

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

                imp_left = self.impurity_(y_left)
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
            return self.left_._predict_row(row)
        else:
            return self.right_._predict_row(row)

    def predict(self, X):
        X = pd.DataFrame(X)
        return np.array([self._predict_single(row) for _, row in X.iterrows()])
