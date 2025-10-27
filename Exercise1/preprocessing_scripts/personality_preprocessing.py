import pandas as pd 
import os


def impute_data(df):
 
  categorical_columns = ['Gender', 'Interest']
  df_encoded = pd.get_dummies(df, columns=categorical_columns, dtype='int')
  df_encoded.insert(df_encoded.shape[1]-1, 'Personality', df_encoded.pop('Personality'))
  return df_encoded


def main():
  
  df = pd.read_csv('data' + os.sep +'personality_types_data_v2.csv')
  df_encoded = impute_data(df)

  df_encoded.to_csv("personality_types_data_v2_encoded.csv", index=False)

if __name__=='__main__':

	main()
