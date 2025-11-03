import pandas as pd 
import os
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.stats import skew
import numpy as np



class Personality_type_preprocessing(BaseEstimator, TransformerMixin):

 # BaseEstimator and TransformerMixin for creating a custom transformer class

  def __init__(self):

    self.scaler = StandardScaler() 

   
  def impute_data(self, X, Y=None):

     categorical_columns = ['Gender', 'Interest']
     X_encoded = pd.get_dummies(X, columns=categorical_columns, dtype='int')
    
     return X_encoded, Y


  def apply_standarization(self, X):

    scaler = StandardScaler()
  
    cols_to_scale = X.select_dtypes(include='number').columns

    df_scaled = X.copy()

    df_scaled[cols_to_scale] = scaler.fit_transform(df_scaled[cols_to_scale])

    return df_scaled

  
  
  def fit(self, X, Y=None):
        # Fit the scaler only on training X
        X_transformed, Y = self.impute_data(X, Y)

        numeric_colums = X_transformed.select_dtypes(include='number').columns
        self.scaler.fit(X_transformed[numeric_colums])
        return self


  def transform(self, X, Y=None):

        X_transformed, Y = self.impute_data(X, Y)
        numeric_colums = X_transformed.select_dtypes(include='number').columns

        X_transformed[numeric_colums] = self.scaler.transform(X_transformed[numeric_colums])

        if Y is not None:
            return X_transformed, Y

        return X_transformed




class Ozone_preprocessing(BaseEstimator, TransformerMixin):

  # BaseEstimator and TransformerMixin for creating a custom transformer class

  def __init__(self):

    self.scaler = StandardScaler()

   
  def drop_rows_with_majority_NAN(self, X, Y= None):

  
    drop_index = X.dropna(thresh=X.shape[1]/2).index 

    X = X.loc[drop_index]
    if Y is not None:
        Y = Y.loc[drop_index]

    return X, Y



  def impute_data(self, X, Y= None):
 
    X['Date'] = pd.to_datetime(X['Date'])

    for col in X.columns[1:]:
       X[col] = pd.to_numeric(X[col], errors="coerce")

    # drop rows with more than 50% of features having missing values
    #X, Y = self.drop_rows_with_majority_NAN(X, Y)

    X['year_month'] = X['Date'].dt.to_period('M')

    cols_to_impute = X.select_dtypes(include='number').columns

    for col in cols_to_impute:
       X[col] = X.groupby('year_month')[col].transform(lambda x: x.fillna(x.mean()))
  
    # Because some months have no data, instead of dropping them, impute them with the mean of the month over the years
    X['month'] = X['Date'].dt.month
    for col in cols_to_impute:
       X[col] = X.groupby('month')[col].transform(lambda x: x.fillna(x.mean()))

    X = X.drop(columns=['year_month', 'month'])
    X = X.set_index("Date")

    return X, Y


  def apply_standarization(self, X):

    scaler = StandardScaler()
  
    #remove Target output from list of columns to be scaled
    cols_to_scale = X.select_dtypes(include='number').columns

    df_scaled = X.copy()

    df_scaled[cols_to_scale] = scaler.fit_transform(df_scaled[cols_to_scale])

    return df_scaled

  
  
  def fit(self, X, Y=None):

        # Fit the scaler only on training X
        X_transformed, Y = self.impute_data(X, Y)
    
        numeric_columns = X_transformed.select_dtypes(include='number').columns

        self.scaler.fit(X_transformed[numeric_columns])

        return self


  def transform(self, X, Y=None):

        X_transformed, Y = self.impute_data(X, Y)
        numeric_columns = X_transformed.select_dtypes(include='number').columns
        X_transformed[numeric_columns] = self.scaler.transform(X_transformed[numeric_columns])

        if Y is not None:
            return X_transformed, Y

        return X_transformed
      
   


class breast_cancer_preprocessing(BaseEstimator, TransformerMixin):



  def __init__(self):

    self.scaler = StandardScaler() 

   
  # drop area and perimeter columns
  def drop_area_perimeter_cols(self, X):

    X.columns = X.columns.str.strip()
    return X.drop(X.filter(regex = "perimeter|area", axis = 1).columns, axis = 1)


  def log_transform_dueto_skew(self, X, fit=False):
   
   df_log = X.copy()

   numeric_cols = X.drop(columns=["ID"]).select_dtypes(include='number').columns
   
   if fit:
     skewness_values = df_log[numeric_cols].apply(lambda x: skew(x, bias=False))
     self.highly_skewed = skewness_values[abs(skewness_values) >= 3].index

   # make sure there are no negative values
   min_vals = df_log[self.highly_skewed].min()
   shifts = (min_vals <= 0) * (-min_vals + 1)

   log_transformed = np.log1p(df_log[self.highly_skewed] + shifts).rename(columns=lambda x: f"log_{x}")
   df_log = df_log.drop(columns=self.highly_skewed)
   df_log = pd.concat([df_log, log_transformed], axis = 1)

   return df_log



  def apply_standardization(self, X):

   scaler = StandardScaler()
  
   #remove Target output from list of columns to be scaled
   cols_to_scale = X.drop(columns=["ID"]).select_dtypes(include='number').columns

   df_scaled = X.copy()

   df_scaled[cols_to_scale] = scaler.fit_transform(df_scaled[cols_to_scale])

   return df_scaled

  
  
  def fit(self, X, Y=None):
        # Fit the scaler only on training X
        X_prime = self.drop_area_perimeter_cols(X)
        X_transformed = self.log_transform_dueto_skew(X_prime, True)

        numeric_colums = X_transformed.select_dtypes(include='number').columns
        self.scaler.fit(X_transformed[numeric_colums])
        return self


  def transform(self, X, Y=None):

        X_prime = self.drop_area_perimeter_cols(X)
        X_transformed = self.log_transform_dueto_skew(X_prime, False)
        numeric_colums = X_transformed.select_dtypes(include='number').columns

        X_transformed[numeric_colums] = self.scaler.transform(X_transformed[numeric_colums])

        if Y is not None:
            return X_transformed, Y

        return X_transformed