import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from Exercise2.Random_Forest_Our_Implementation import DecisionTree


class RandomForest:

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
        tree_seed = self.rng.integers(0, 1e9) + tree_idx
        X_boot, y_boot = self._bootstrap_sample(X, y)

        max_features_val = int(np.sqrt(X_boot.shape[1]))
        
        tree = DecisionTree(
            max_depth=self.max_depth,
            min_criterion=0.01,
            min_sample_split=self.min_samples_split,
            max_features=max_features_val,
            loss=self.loss,
            random_state=tree_seed
        )
        tree.fit(X_boot, y_boot)
        return tree
    
    def fit(self, X, y):
        self.trees = []

        
        self.trees = Parallel(n_jobs=self.n_jobs)(
            delayed(self._build_tree)(X, y, i)
            for i in range(self.n_estimators)
        )
        
        return self

    
    def predict(self, X):
        if not self.trees:
            raise ValueError("The tree list is empty, train model first")
        
        predictions = []
        for tree in self.trees:
            pred = tree.predict(X)
            predictions.append(pred.reshape(-1, 1))
        
        tree_predictions_mean = np.mean(np.concatenate(predictions, axis=1), axis=1)
        return tree_predictions_mean
    

