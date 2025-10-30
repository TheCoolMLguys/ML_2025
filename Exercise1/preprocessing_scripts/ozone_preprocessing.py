import pandas as pd 
import os
from sklearn.preprocessing import StandardScaler


def drop_rows_with_majority_NAN(df):

 return df.dropna(thresh=df.shape[1]/2)



def impute_data(df):
 

  df['Date'] = pd.to_datetime(df['Date'])

  for col in df.columns[1:]:
      df[col] = pd.to_numeric(df[col], errors="coerce")

  # drop rows with more than 50% of features having missing values
  df = drop_rows_with_majority_NAN(df)

  df['year_month'] = df['Date'].dt.to_period('M')

  cols_to_impute = df.select_dtypes(include='number').columns

  for col in cols_to_impute:
     df[col] = df.groupby('year_month')[col].transform(lambda x: x.fillna(x.mean()))
  
  # Because some months have no data, instead of dropping them, impute them with the mean of the month over the years
  df['month'] = df['Date'].dt.month
  for col in cols_to_impute:
     df[col] = df.groupby('month')[col].transform(lambda x: x.fillna(x.mean()))

  df = df.drop(columns=['year_month', 'month'])
  

  return df


def apply_standarization(df):

  scaler = StandardScaler()
  
  #remove Target output from list of columns to be scaled
  cols_to_scale = df.drop(columns="Ozone").select_dtypes(include='number').columns

  df_scaled = df.copy()

  df_scaled[cols_to_scale] = scaler.fit_transform(df_scaled[cols_to_scale])

  return df_scaled


def main():
  
  df = pd.read_csv('data' + os.sep +'ozone_level_data.csv')
  df_encoded = impute_data(df)
  df_scaled = apply_standarization(df_encoded)

  df_scaled.to_csv("ozone_level_data_encoded_scaled.csv", index=False)

if __name__=='__main__':

	main()
