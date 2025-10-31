import pandas as pd 
import numpy as np
import os
from scipy.stats import skew
from sklearn.preprocessing import StandardScaler

# drop area and perimeter columns
def drop_area_perimeter_cols(df):
    return df.drop(df.filter(regex = "perimeter|area", axis = 1).columns, axis = 1)

# applying standardization
def apply_standardization(df):

  scaler = StandardScaler()
  
  #remove Target output from list of columns to be scaled
  cols_to_scale = df.drop(columns=["ID", "class"]).select_dtypes(include='number').columns

  df_scaled = df.copy()

  df_scaled[cols_to_scale] = scaler.fit_transform(df_scaled[cols_to_scale])

  return df_scaled

# log transform variables with a skewness over a certain threshold.
# variables are then replaced by log transformed variables
# do this before scaling
def log_transform_dueto_skew(df):
   
   df_log = df.copy()

   numeric_cols = df.drop(columns=["ID", "class"]).select_dtypes(include='number').columns
   
   skewness_values = df_log[numeric_cols].apply(lambda x: skew(x, bias=False))
   highly_skewed = skewness_values[abs(skewness_values) >= 3].index

   # make sure there are no negative values
   min_vals = df_log[highly_skewed].min()
   shifts = (min_vals <= 0) * (-min_vals + 1)

   log_transformed = np.log1p(df_log[highly_skewed] + shifts).rename(columns=lambda x: f"log_{x}")
   df_log = df_log.drop(columns=highly_skewed)
   df_log = pd.concat([df_log, log_transformed], axis = 1)

   return df_log

def main():
  
  df = pd.read_csv(os.path.join('Exercise1', 'data', 'breast-cancer-diagnostic.shuf.lrn.csv'))
  df.columns = df.columns.str.strip()
  df_dropped = drop_area_perimeter_cols(df)
  df_log = log_transform_dueto_skew(df_dropped)
  df_scaled = apply_standardization(df_log)

  df_scaled.to_csv(os.path.join('Exercise1', 'data', 'breast_cancer_diagnostic_preprocessed.csv'), index=False)

if __name__=='__main__':
	main()
   