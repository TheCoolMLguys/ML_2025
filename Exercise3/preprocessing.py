import pandas as pd
import numpy as np
import os
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.stats import skew
from sklearn.preprocessing import StandardScaler, RobustScaler, OrdinalEncoder, OneHotEncoder, LabelEncoder
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


    def prepareGenHierarchiesObject(self, X):
        

        genh_degree = CGH.CatGenHierarchy('degree', 'data' + os.sep + 'gen_hierarchies' + os.sep + 'Degree.json')
        genh_branch = CGH.CatGenHierarchy('branch', 'data' + os.sep + 'gen_hierarchies' + os.sep + 'Branch.json')
        genh_sex = CGH.CatGenHierarchy('sex', 'data' + os.sep + 'gen_hierarchies' + os.sep + 'Gender.json')
        genh_interest = CGH.CatGenHierarchy('interest', 'data' + os.sep + 'gen_hierarchies' + os.sep + 'Interest.json')

        gen_hierarchies = {
             'categorical': {
               'degree': genh_degree,
               'branch': genh_branch,
                'sex': genh_sex,
                'interest': genh_interest
            },
              'range': {}
              }

        return gen_hierarchies

    
    def _knn_adj_list(self, X, k=5, ignore_columns=None):


      X_knn = X.copy()

    # Drop ignored columns (e.g., names)
      if ignore_columns is not None:
        X_knn = X_knn.drop(columns=ignore_columns, errors='ignore')

    # Initialize encoders dictionary
      self._knn_encoders = {}

    # Encode categorical columns
      for col in X_knn.select_dtypes(include='object').columns:
        le = LabelEncoder()
        X_knn[col] = le.fit_transform(X_knn[col].astype(str))
        self._knn_encoders[col] = le

    # Ensure all remaining columns are numeric
      X_knn = X_knn.apply(pd.to_numeric)

    # Fit NearestNeighbors
      nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto').fit(X_knn)
      distances, indices = nbrs.kneighbors(X_knn)

    # Build adjacency list (skip self, which is first)
      adj_list = {i: list(indices[i][1:]) for i in range(len(X_knn))}

      return adj_list



    def fit(self, X, y=None, k_graph=10):

   
      if self.gen_hierarchies is None:
        self.gen_hierarchies = self.prepareGenHierarchiesObject(X)

        # build numeric range hierarchies
        self.range_bounds = {}
        num_cols = X.select_dtypes(include="number").columns

        for col in num_cols:
            col_min = X[col].min()
            col_max = X[col].max()
            self.range_bounds[col] = (col_min, col_max)

            self.gen_hierarchies["range"][col] = RGH.RangeGenHierarchy(
                col, col_min, col_max
            )

    # build adjacency graph ONLY on training data
      if self.adj_list is None:
        self.adj_list = self._knn_adj_list(X, k=k_graph)

    # build clusters only for training
      self.clusters_ = self.build_clusters(X)

      return self

    
    def build_clusters(self, X):

      adults = X.to_dict(orient="index")
      clusters = []
      added = {}

      for node in adults:
        if added.get(node, False):
            continue

        cluster = CL.NodeCluster(node, adults, self.adj_list, self.gen_hierarchies)

        added[node] = True

        while len(cluster.getNodes()) < self.k:

            best_cost = 1e9 # very large number
            best_candidate = None

            for candidate in adults:
                if added.get(candidate, False):
                    continue

                cost = cluster.computeNodeCost(candidate)
                if cost < best_cost:
                    best_cost = cost
                    best_candidate = candidate

            if best_candidate is None:
                break

            cluster.addNode(best_candidate)
            added[best_candidate] = True

        clusters.append(cluster)

      return clusters



    def transform(self, X):

        """
        Based on logic of SaNGReea algorithm
        https://github.com/tanjascats/SaNGreeA-anonymisation/blob/master/src/SaNGreeA.py

        X: features
        returns: locally anonymized features
        """

        for col, (low, high) in self.range_bounds.items():
           X[col] = X[col].clip(lower=low, upper=high)

        # if fit() has been called, then just assign to existing clusters
        if hasattr(self, "clusters_"):
           return self.assign_to_clusters(X)

        clusters = self.build_clusters(X)
        return self.clusters_to_dataframe(clusters, X)


    def assign_to_clusters(self, X):

       rows = {}

       for idx, row in X.iterrows():
          row_dict = row.to_dict()
    
          best_cluster = None
          best_cost = 1e9

          for cluster in self.clusters_:
            cost = cluster.computeCost_per_instance(row_dict)
            if cost < best_cost:
                best_cost = cost
                best_cluster = cluster

          out = {}

        # numeric attributes → cluster ranges
          for col, (low, high) in best_cluster._genRangeFeatures.items():
            out[f"{col}_min"] = low
            out[f"{col}_max"] = high

        # categorical attributes → cluster generalized value
          for col, val in best_cluster._genCatFeatures.items():
            out[col] = val

          rows[idx] = out

       df = pd.DataFrame.from_dict(rows, orient="index")
       df = df.loc[X.index]  # preserve order
       return df


    def clusters_to_dataframe(self, clusters, X):
        """
        Converts clusters to anonymized DataFrame.
        Ensures every original row is present.
        """
        rows = {}
        for cluster in clusters:

          nodes = cluster.getNodes()
          anonymized_rows = cluster.getAllAnonymizedNodes()
          for idx, node in enumerate(nodes):
             rows[node] = anonymized_rows[idx]
    

        for idx in X.index:
           if idx not in rows:
              rows[idx] = X.loc[idx].to_dict()
    
        df = pd.DataFrame.from_dict(rows, orient="index")
        df = df.loc[X.index]  # preserve order
       
        return df

    
class SaNGreeATransformer_microaggregation(BaseEstimator, TransformerMixin):
       
    "Local k-anonymization using the SaNGreeA algorithm."

    def __init__(self, k, cat_features=None, gen_hierarchies=None, adj_list=None):
        self.k = k
        self.gen_hierarchies = gen_hierarchies # json file input
        self.cat_features = cat_features
        self.adj_list = adj_list

    def _knn_adj_list(self, X, k=10):
      # to generate adjacency list based on knn 

      nbrs = NearestNeighbors(n_neighbors=k+1).fit(X)
      _, indices = nbrs.kneighbors(X)
 
      adj = {}
      for i, neigh in enumerate(indices):
        adj[i] = neigh[1:].tolist()
      return adj
    
    def _microaggregate_cluster(self, cluster, X):
       # select data that needs microaggregation from clustering
       nodes = cluster.getNodes()
       temp = X.loc[nodes]

       aggregated = {}

       for col in self.numeric_features:
        aggregated[col] = temp[col].mean()

       for col in self.categorical_features:
        aggregated[col] = temp[col].mode(dropna=True).iloc[0]
             
       return aggregated, nodes
    

    def fit(self, X, y=None, k_graph=10):
      """
      Based on logic of SaNGReea algorithm
      https://github.com/tanjascats/SaNGreeA-anonymisation/blob/master/src/SaNGreeA.py

      X: features
      returns: locally anonymized features
      """

      adults = X.to_dict(orient="index")
      clusters = []
      added = {}

      if self.cat_features is None:
        self.categorical_features = set(X.select_dtypes(include=["object", "category"]).columns)
      else:
        self.categorical_features = set(self.cat_features)

      self.numeric_features = [
          col for col in X.columns if col not in self.cat_features
      ]     

      # if gen_hierarchies not provided (default)
      if self.gen_hierarchies is None:
        self.gen_hierarchies = {"categorical": {}, "range": {}}

        # Categorical columns must be provided in original DataFrame
        for col in self.categorical_features:
            unique_vals = X[col].unique()
            self.gen_hierarchies["categorical"][col] = CGH.CatGenHierarchy(col, {val: '*' for val in unique_vals})

        # Range hierarchies for numeric columns
        for col in self.numeric_features:
            col_min = X[col].min()
            col_max = X[col].max()
            self.gen_hierarchies["range"][col] = RGH.RangeGenHierarchy(col, col_min, col_max)


      # If adjacency list not provided (default), compute using k-NN
      if self.adj_list is None:
         self.adj_list = self._knn_adj_list(X, k=k_graph)

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
        
      self.clusters_ = clusters
      self.columns_ = X.columns
      self.dtypes_ = X.dtypes
      
      return self
    
    def transform(self, X):
       # use clustering from SaNGReea algorithm to aggregate
       rows = {}
       
       for cluster in self.clusters_:
          agg_row, nodes = self._microaggregate_cluster(cluster, X)
          
          for node in nodes:
             rows[node] = agg_row.copy()

       return pd.DataFrame.from_dict(rows, orient="index").loc[X.index]


''' Data preprocessing - Class definition  '''



class breast_cancer_preprocessing(BaseEstimator, TransformerMixin):



  def __init__(self, anonymity=False, k=5, anonymizer_class=None):

    self.scaler = RobustScaler()
    self.anonymity = anonymity
    self.k = k
    self.anonymizer_class = anonymizer_class
    self.label_encoders = {}     
    self.numeric_columns = None
    self.cat_columns = None
    self.enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

   
  # drop area and perimeter columns, since they are functions of radius
  # drop ID column, not of any information value
  def drop_area_perimeter_ID_cols(self, X):

    X = X.drop(columns = ["ID"])
    X.columns = X.columns.str.strip()
    return X.drop(X.filter(regex = "perimeter|area", axis = 1).columns, axis = 1)


  
  def fit(self, X, Y=None):
        X = self.drop_area_perimeter_ID_cols(X)

        self.numeric_columns = X.select_dtypes(include="number").columns.tolist()
        self.cat_columns = X.select_dtypes(include="object").columns.tolist()

        self.scaler.fit(X[self.numeric_columns])

        if self.anonymity and self.anonymizer_class is not None:
            self.anonymizer = self.anonymizer_class(k=self.k)
            self.anonymizer.fit(X)

        if self.cat_columns:
            self.enc.fit(X[self.cat_columns].astype(str))

        return self


  def transform(self, X, Y=None):

        X = self.drop_area_perimeter_ID_cols(X)

        X[self.numeric_columns] = self.scaler.transform(X[self.numeric_columns])

        if self.anonymity:
            X = self.anonymizer.transform(X)

        if self.cat_columns:
            X_cat = pd.DataFrame(self.enc.transform(X[self.cat_columns].astype(str)),
                                 columns=self.enc.get_feature_names_out(self.cat_columns),
                                 index=X.index)

            X = pd.concat([X[self.numeric_columns], X_cat], axis=1)

        return X


class Personality_type_preprocessing(BaseEstimator, TransformerMixin):


    def __init__(self, anonymity=False, k=5, anonymizer_class=None):

        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.k = k
        self.feature_columns = None
        self.numeric_columns = None
        self.anonymity = anonymity
        self.anonymizer_class = anonymizer_class
        self.categorical_columns = ['Gender', 'Interest']
        self.enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    
    def impute_data(self, X):
        return X.copy()


    def get_numeric_columns_after_anonymization(self, X):
       
       cols = []

       for col in self.numeric_columns:
          if col in X.columns:
 
             cols.append(col)
          elif f"{col}_min" in X.columns and f"{col}_max" in X.columns:
             # local transformation with ranges
             cols.extend([f"{col}_min", f"{col}_max"])

       return cols



    def fit(self, X, Y=None):

        X = self.impute_data(X)

        self.numeric_columns = (X.drop(columns=self.categorical_columns).select_dtypes(include='number').columns)

        self.scaler.fit(X[self.numeric_columns])

        X_scaled = X.copy()
        X_scaled[self.numeric_columns] = self.scaler.transform(X[self.numeric_columns])

        if self.anonymity:
            self.anonymizer = self.anonymizer_class(k=self.k)
            self.anonymizer.fit(X_scaled)

        self.enc.fit(X_scaled[self.categorical_columns].astype(str))

        return self



    def transform(self, X, Y=None):

        X = self.impute_data(X)

        X_scaled = X.copy()
        X_scaled[self.numeric_columns] = self.scaler.transform(X[self.numeric_columns])

        if self.anonymity:
            X_scaled = self.anonymizer.transform(X_scaled)

        new_numeric_cols = self.get_numeric_columns_after_anonymization(X_scaled)

        X_num = X_scaled[new_numeric_cols]

        X_cat = pd.DataFrame(self.enc.transform(X_scaled[self.categorical_columns].astype(str)),
                               columns=self.enc.get_feature_names_out(self.categorical_columns),
                                index=X.index)

        return pd.concat([X_num, X_cat], axis=1)



class Student_placement_preprocessing(BaseEstimator, TransformerMixin):

    def __init__(self, anonymity=False, k=5, anonymizer_class=None):
    

        self.scaler = StandardScaler()
        self.categorical_columns = ['Gender', 'Degree', 'Branch']
        self.numeric_columns = None
        self.k = k
        self.numeric_columns = None
        self.label_encoders = {}
        self.anonymity = anonymity
        self.anonymizer_class = anonymizer_class
        self.enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)


    def impute_data(self, X):

        X = X.drop(columns=["Student_ID"], errors="ignore").copy()
        return X


    def get_numeric_columns_after_anonymization(self, X):
       
       cols = []

       for col in self.numeric_columns:
          if col in X.columns:
 
             cols.append(col)
          elif f"{col}_min" in X.columns and f"{col}_max" in X.columns:
             # local transformation with ranges
             cols.extend([f"{col}_min", f"{col}_max"])

       return cols



    def fit(self, X, Y=None):

        X = self.impute_data(X)

        self.numeric_columns = (X.drop(columns=self.categorical_columns).select_dtypes(include='number').columns)

        self.scaler.fit(X[self.numeric_columns])

        X_scaled = X.copy()
        X_scaled[self.numeric_columns] = self.scaler.transform(X[self.numeric_columns])

        if self.anonymity:
            self.anonymizer = self.anonymizer_class(k=self.k)
            self.anonymizer.fit(X_scaled)

        self.enc.fit(X_scaled[self.categorical_columns].astype(str))

        return self




    def transform(self, X, Y=None):

        X = self.impute_data(X)

        X_scaled = X.copy()
        X_scaled[self.numeric_columns] = self.scaler.transform(X[self.numeric_columns])

        if self.anonymity:
            X_scaled = self.anonymizer.transform(X_scaled)

        new_numeric_cols = self.get_numeric_columns_after_anonymization(X_scaled)

        X_num = X_scaled[new_numeric_cols]

        X_cat = pd.DataFrame(
            self.enc.transform(X_scaled[self.categorical_columns].astype(str)),
            columns=self.enc.get_feature_names_out(self.categorical_columns),
            index=X.index
        )

        return pd.concat([X_num, X_cat], axis=1)