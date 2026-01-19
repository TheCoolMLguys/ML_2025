import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, RobustScaler, OrdinalEncoder, OneHotEncoder
import modules.nodeCluster as CL
import modules.catGenHierarchy as CGH
import modules.rangeGenHierarchy as RGH
from sklearn.neighbors import NearestNeighbors



''' Anonymity techniques - Class definition  '''


class SaNGreeATransformer(BaseEstimator, TransformerMixin):

    """
    Local k-anonymization using the SaNGreeA algorithm.
    """

    def __init__(self, k, gen_hierarchies=None, adj_list=None):
        self.k = k
        self.gen_hierarchies = gen_hierarchies # json file input
        self.adj_list = adj_list


    def _knn_adj_list(self, X, k=10):
   
       # to generate adjacency list based on knn 

       nbrs = NearestNeighbors(n_neighbors=k+1).fit(X)
       _, indices = nbrs.kneighbors(X)
 
       adj = {}
       for i, neigh in enumerate(indices):
          adj[i] = neigh[1:].tolist()
       return adj



    def fit(self, X, y=None, k_graph=10):

       # if gen_hierarchies not provided (default)
       if self.gen_hierarchies is None:
        
        self.gen_hierarchies = {"categorical": {}, "range": {}}

        # Categorical columns must be provided in original DataFrame
        cat_cols = X.select_dtypes(include="object").columns
        for col in cat_cols:
            unique_vals = X[col].unique()
            self.gen_hierarchies["categorical"][col] = CGH.CatGenHierarchy(col, {val: '*' for val in unique_vals})

        # Range hierarchies for numeric columns
        num_cols = X.select_dtypes(include="number").columns
        for col in num_cols:
            col_min = X[col].min()
            col_max = X[col].max()
            self.gen_hierarchies["range"][col] = RGH.RangeGenHierarchy(col, col_min, col_max)


        # If adjacency list not provided (default), compute using k-NN
        if self.adj_list is None:
           self.adj_list = self._knn_adj_list(X, k=k_graph)

        return self
    


    def transform(self, X):
        """
        Based on logic of SaNGReea algorithm
        https://github.com/tanjascats/SaNGreeA-anonymisation/blob/master/src/SaNGreeA.py

        X: features
        returns: locally anonymized features
        """

        adults = X.to_dict(orient="index")
        clusters = []
        added = {}

        for node in adults:

            if added.get(node, False):
                continue

            cluster = CL.NodeCluster(node, adults, self.adj_list, self.gen_hierarchies)

            added[node] = True

            while len(cluster.getNodes()) < self.k:
               best_cost = 1e9
               best_candidate = None

               for candidate in adults:
                 if added.get(candidate, False):
                   continue

               cost = cluster.computeNodeCost(candidate)
               if cost < best_cost:
                  best_cost = cost
                  best_candidate = candidate

               if best_candidate is None:
                  # no more candidates to add, break the loop
                  break

               cluster.addNode(best_candidate)
               added[best_candidate] = True


            clusters.append(cluster)

        return self._clusters_to_dataframe(clusters)


    def _clusters_to_dataframe(self, clusters):
        """
        Converts clusters to anonymized DataFrame
        """
        rows = []
        for cluster in clusters:
            rows.extend(cluster.getAllAnonymizedNodes())
        return pd.DataFrame(rows)



''' Data preprocessing - Class definition  '''



class breast_cancer_preprocessing(BaseEstimator, TransformerMixin):



  def __init__(self):

    self.scaler = RobustScaler() 

   
  # drop area and perimeter columns, since they are functions of radius
  # drop ID column, not of any information value
  def drop_area_perimeter_ID_cols(self, X):

    X = X.drop(columns = ["ID"])
    X.columns = X.columns.str.strip()
    return X.drop(X.filter(regex = "perimeter|area", axis = 1).columns, axis = 1)


  # idea: quantify skewness and create threshold, which if passed, variables will be log transformed (comparison with and without)
  def log_transform_dueto_skew(self, X, fit=False):

   if fit:
     df_log = X.copy()
     numeric_cols = X.select_dtypes(include='number').columns
     
     skewness_values = df_log[numeric_cols].apply(lambda x: skew(x, bias=False))
     self.highly_skewed = skewness_values[abs(skewness_values) >= 3].index
     
     # make sure there are no negative values
     min_vals = df_log[self.highly_skewed].min()
     shifts = (min_vals <= 0) * (-min_vals + 1)
     log_transformed = np.log1p(df_log[self.highly_skewed] + shifts).rename(columns=lambda x: f"log_{x}")
     
     df_log = df_log.drop(columns=self.highly_skewed)
     df_log = pd.concat([df_log, log_transformed], axis = 1)
     
     return df_log
   
   return X


  
  def fit(self, X, Y=None):
        # Fit the scaler only on training X
        X_transformed = self.drop_area_perimeter_ID_cols(X)
        #X_transformed = self.log_transform_dueto_skew(X_prime, False)

        numeric_colums = X_transformed.select_dtypes(include='number').columns
        self.scaler.fit(X_transformed[numeric_colums])
        return self


  def transform(self, X, Y=None):

        X_transformed = self.drop_area_perimeter_ID_cols(X)
        #X_transformed = self.log_transform_dueto_skew(X_prime, False)
        numeric_colums = X_transformed.select_dtypes(include='number').columns

        X_transformed[numeric_colums] = self.scaler.transform(X_transformed[numeric_colums])

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


        return X_transformed



class Student_placement_preprocessing(BaseEstimator, TransformerMixin):


    def __init__(self):

      self.scaler = StandardScaler() 

   
    def impute_data(self, X, Y=None):

      X = X.drop(columns=["Student_ID",], errors="ignore").copy()
      categorical_columns = ['Gender', 'Degree', 'Branch']

      X_encoded = pd.get_dummies(X, columns=categorical_columns, dtype='int')
    
      return X_encoded, Y

  
  
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

      return X_transformed