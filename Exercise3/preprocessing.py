import pandas as pd
import numpy as np
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

       # if gen_hierarchies not provided (default)
       if self.gen_hierarchies is None:
        
        self.gen_hierarchies = {"categorical": {}, "range": {}}

        # Categorical columns must be provided in original DataFrame
        cat_cols = X.select_dtypes(include="object").columns
        for col in cat_cols:
            unique_vals = X[col].unique()
            self.gen_hierarchies["categorical"][col] = CGH.CatGenHierarchy(col, {val: '*' for val in unique_vals})

        # Range hierarchies for numeric columns
        self.range_bounds = {}
        num_cols = X.select_dtypes(include="number").columns
        for col in num_cols:
            col_min = X[col].min()
            col_max = X[col].max()
            self.range_bounds[col] = (col_min, col_max)
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

        X = X.copy()
        for col, (low, high) in self.range_bounds.items():
              X[col] = X[col].clip(lower=low, upper=high)

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

        return self._clusters_to_dataframe(clusters, X)


    def _clusters_to_dataframe(self, clusters, X):
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

   
  # drop area and perimeter columns, since they are functions of radius
  # drop ID column, not of any information value
  def drop_area_perimeter_ID_cols(self, X):

    X = X.drop(columns = ["ID"])
    X.columns = X.columns.str.strip()
    return X.drop(X.filter(regex = "perimeter|area", axis = 1).columns, axis = 1)


  
  def fit(self, X, Y=None):

        X_transformed = self.drop_area_perimeter_ID_cols(X)

     
        self.numeric_columns = X_transformed.select_dtypes(include="number").columns.tolist()
        self.cat_columns = X_transformed.select_dtypes(include="object").columns.tolist()

        self.scaler.fit(X_transformed[self.numeric_columns])

    
        if self.anonymity and self.anonymizer_class is not None:
            self.anonymizer = self.anonymizer_class(k=self.k)
            self.anonymizer.fit(X_transformed)

        return self


  def transform(self, X, Y=None):

        X_transformed = self.drop_area_perimeter_ID_cols(X)


        X_transformed[self.numeric_columns] = self.scaler.transform(X_transformed[self.numeric_columns])


        if self.anonymity:
            X_transformed = self.anonymizer.transform(X_transformed)

  
        for col in X_transformed.select_dtypes(include="object").columns:
            if col not in self.label_encoders:
                le = LabelEncoder()
                le.fit(X_transformed[col].astype(str))
                self.label_encoders[col] = le
            else:
                le = self.label_encoders[col]

            X_transformed[col] = le.transform(X_transformed[col].astype(str))

        return X_transformed


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
        self.unknown_label = '__unknown__'

    
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

       for col in self.categorical_columns:
            le = LabelEncoder()
            le.fit(list(X_scaled[col].astype(str)) + [self.unknown_label])
           # le.fit(X_scaled[col].astype(str))
            self.label_encoders[col] = le


       return self




    def transform(self, X, Y=None):


        X = self.impute_data(X)

        X_scaled = X.copy()
        X_scaled[self.numeric_columns] = self.scaler.transform(X[self.numeric_columns])

        if self.anonymity and self.anonymizer_class is not None:
            X_scaled = self.anonymizer.transform(X_scaled)

        new_numeric_cols = self.get_numeric_columns_after_anonymization(X_scaled)

        X_cat = pd.DataFrame(index=X_scaled.index)

        for col in self.categorical_columns:
            values = X_scaled[col].astype(str).apply(
                lambda x: x if x in self.label_encoders[col].classes_ else self.unknown_label)

            X_cat[col] = self.label_encoders[col].transform(values) #X_scaled[col].astype(str))

        X_num = X_scaled[new_numeric_cols]

        X_transformed = pd.concat([X_num, X_cat], axis=1)

        return X_transformed



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
        self.unknown_label = '__unknown__'


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

       for col in self.categorical_columns:
            le = LabelEncoder()
            le.fit(list(X_scaled[col].astype(str)) + [self.unknown_label])
           # le.fit(X_scaled[col].astype(str))
            self.label_encoders[col] = le


       return self




    def transform(self, X, Y=None):


        X = self.impute_data(X)

        X_scaled = X.copy()
        X_scaled[self.numeric_columns] = self.scaler.transform(X[self.numeric_columns])

        if self.anonymity and self.anonymizer_class is not None:
            X_scaled = self.anonymizer.transform(X_scaled)

        new_numeric_cols = self.get_numeric_columns_after_anonymization(X_scaled)

        X_cat = pd.DataFrame(index=X_scaled.index)

        for col in self.categorical_columns:
            values = X_scaled[col].astype(str).apply(
                lambda x: x if x in self.label_encoders[col].classes_ else self.unknown_label)

            X_cat[col] = self.label_encoders[col].transform(values) #X_scaled[col].astype(str))

        X_num = X_scaled[new_numeric_cols]

        X_transformed = pd.concat([X_num, X_cat], axis=1)

        return X_transformed