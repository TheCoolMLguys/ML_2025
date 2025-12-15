import pandas as pd 
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin

class House_Price_preprocessing(BaseEstimator, TransformerMixin):
    
    def __init__(self):
          
        self.oe = None
        self.onehot = None
        self.onehot_vars = None
    
    # drop ID column, not of any information value
    # One Hot Encoding
    def simple_preprocess(self, X):
        
        X = X.drop(columns=["Id"], errors="ignore").copy()

        # Ordinal 
        X["Condition"] = self.oe.transform(X[["Condition"]])

        # One Hot encoding 
        onehot_encoded = self.onehot.transform(X[self.onehot_vars])
        onehot_cols = self.onehot.get_feature_names_out(self.onehot_vars)
        onehot_df = pd.DataFrame(onehot_encoded, columns=onehot_cols, index=X.index)

        X = X.drop(columns=self.onehot_vars)
        X = pd.concat([X, onehot_df], axis=1)

        return X 

    def fit(self, X, Y=None):

        # Ordinal encoding: Condition
        ord_categories = [["Poor", "Fair", "Good", "Excellent"]]
        self.oe = OrdinalEncoder(categories=ord_categories)
        self.oe.fit(X[["Condition"]])

        # One Hot encoding
        cat_cols = X.select_dtypes(include="object").columns.tolist()
        self.onehot_vars = [c for c in cat_cols if c != "Condition"]
        self.onehot = OneHotEncoder(drop="if_binary", sparse_output=False)
        self.onehot.fit(X[self.onehot_vars])

        # transform
        self.simple_preprocess(X)

        return self


    def transform(self, X, Y=None):

        X_transformed = self.simple_preprocess(X)

        if Y is not None:
            return X_transformed, Y

        return X_transformed

class Phone_Addiction_preprocessing(BaseEstimator, TransformerMixin):

    def __init__(self):
          
        self.oe = None
        self.onehot = None
        self.onehot_vars = None
    
    def simple_preprocess(self, X):
        
        # drop identifiers
        X = X.drop(columns=["ID", "Name", "Location"], errors="ignore").copy()
        # make school grade numeric
        X['School_Grade'] = X['School_Grade'].str.extract(r"(\d+)").astype(int)

        # One Hot encoding 
        onehot_encoded = self.onehot.transform(X[self.onehot_vars])
        onehot_cols = self.onehot.get_feature_names_out(self.onehot_vars)
        onehot_df = pd.DataFrame(onehot_encoded, columns=onehot_cols, index=X.index)

        X = X.drop(columns=self.onehot_vars)
        X = pd.concat([X, onehot_df], axis=1)

        return X 

    def fit(self, X, Y=None):

        self.onehot_vars = [c for c in X.select_dtypes(include="object").columns if c not in  ["ID", "Name", "Location", "School_Grade"]]
        self.onehot = OneHotEncoder(drop="if_binary", sparse_output=False)
        self.onehot.fit(X[self.onehot_vars])

        # transform
        self.simple_preprocess(X)

        return self


    def transform(self, X, Y=None):

        X_transformed = self.simple_preprocess(X)

        if Y is not None:
            return X_transformed, Y

        return X_transformed
    

class Ford_preprocessing(BaseEstimator, TransformerMixin):

    def __init__(self):
          
        self.onehot = None
        self.onehot_vars = None
    
    def simple_preprocess(self, X):
        
        # One Hot encoding 
        onehot_encoded = self.onehot.transform(X[self.onehot_vars])
        onehot_cols = self.onehot.get_feature_names_out(self.onehot_vars)
        onehot_df = pd.DataFrame(onehot_encoded, columns=onehot_cols, index=X.index)

        X = X.drop(columns=self.onehot_vars)
        X = pd.concat([X, onehot_df], axis=1)

        return X 

    def fit(self, X, Y=None):

        self.onehot_vars = X.select_dtypes(include="object").columns.tolist()
        self.onehot = OneHotEncoder(drop="if_binary", sparse_output=False)
        self.onehot.fit(X[self.onehot_vars])

        # transform
        self.simple_preprocess(X)

        return self


    def transform(self, X, Y=None):

        X_transformed = self.simple_preprocess(X)

        if Y is not None:
            return X_transformed, Y

        return X_transformed

