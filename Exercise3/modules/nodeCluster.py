class NodeCluster:
    """
    Generalized cluster for local k-anonymization.
    """

    def __init__(self, node, dataset=None, adj_list=None, gen_hierarchies=None):
        self._nodes = [node]
        self._dataset = dataset
        self._adjList = adj_list or {}
        self._genHierarchies = gen_hierarchies or {"categorical": {}, "range": {}}

        # Initialize categorical features
        self._genCatFeatures = {}
        for col in self._genHierarchies["categorical"]:
            self._genCatFeatures[col] = self._dataset[node][col]

        # Initialize range (numeric) features as [min, max]
        self._genRangeFeatures = {}
        for col in self._genHierarchies["range"]:
            val = self._dataset[node][col]
            self._genRangeFeatures[col] = [val, val]

    def getNodes(self):
        return self._nodes

    def addNode(self, node):
        self._nodes.append(node)
        self._adjList[node] = self._adjList.get(node, [])

        # Update categorical features
        for col in self._genCatFeatures:
            self._genCatFeatures[col] = self.computeNewGeneralization(col, node)[1]

        # Update range features
        for col in self._genRangeFeatures:
            current_range = self._genRangeFeatures[col]
            val = self._dataset[node][col]
            self._genRangeFeatures[col] = [min(current_range[0], val), max(current_range[1], val)]

    def computeNodeCost(self, node, alpha=1.0, beta=0.0):
        """
        Compute total cost (GIL + SIL)
        SIL is optional and can be 0
        """
        gil = self.computeGIL(node)
        sil = 0  # placeholder for Structural Information Loss
        return alpha * gil + beta * sil

    def computeGIL(self, node):
        """Generalization Information Loss"""
        total_cost = 0.0
        # Use weights from globals if available, otherwise equal weights
        try:
            import globals as GLOB
            weight_vector = GLOB.GEN_WEIGHT_VECTORS[GLOB.VECTOR]
        except ImportError:
            # fallback: equal weights
            weight_vector = {"categorical": {col: 1.0 for col in self._genCatFeatures},
                             "range": {col: 1.0 for col in self._genRangeFeatures}}

        for col in self._genCatFeatures:
            weight = weight_vector["categorical"].get(col, 1.0)
            total_cost += weight * self.computeCategoricalCost(col, node)

        for col in self._genRangeFeatures:
            weight = weight_vector["range"].get(col, 1.0)
            total_cost += weight * self.computeRangeCost(col, node)

        return total_cost

    def computeCategoricalCost(self, col, node):
        hierarchy = self._genHierarchies["categorical"][col]
        cluster_level = self.computeNewGeneralization(col, node)[0]
        return float((hierarchy.nrLevels() - cluster_level) / hierarchy.nrLevels())

    def computeRangeCost(self, col, node):
        hierarchy = self._genHierarchies["range"][col]
        current_range = self._genRangeFeatures[col]
        val = self._dataset[node][col]
        return hierarchy.getCostOfRange(min(current_range[0], val), max(current_range[1], val))

    def computeNewGeneralization(self, col, node):
        """Returns new generalization level and value for a categorical feature"""
        hierarchy = self._genHierarchies["categorical"][col]
        cluster_val = self._genCatFeatures[col]
        node_val = self._dataset[node][col]
        cluster_level = hierarchy.getLevelEntry(cluster_val)
        node_level = hierarchy.getLevelEntry(node_val)

        while cluster_val != node_val:
            old_node_level = node_level
            if cluster_level <= node_level:
                node_val = hierarchy.getGeneralizationOf(node_val)
                node_level -= 1
            if old_node_level <= cluster_level:
                cluster_val = hierarchy.getGeneralizationOf(cluster_val)
                cluster_level -= 1

        return [cluster_level, cluster_val]

    def toString(self):
        """Return string representation of cluster"""
        rows = []
        for node in self._nodes:
            row = {}
            for col in self._genRangeFeatures:
                r = self._genRangeFeatures[col]
                row[col] = r[0] if r[0] == r[1] else f"[{r[0]}-{r[1]}]"
            for col in self._genCatFeatures:
                row[col] = self._genCatFeatures[col]
            rows.append(row)
        return str(rows)
    

    def getAllAnonymizedNodes(self):
        """
         Return a list of dicts representing the anonymized rows for this cluster.
         Easy to be converted into a dataframe
        """
        anonymized_rows = []

        for node in self._nodes:
           row = {}

           for col, value_range in self._genRangeFeatures.items():
              row[col] = value_range[0] if value_range[0] == value_range[1] else (value_range[0] + value_range[1])/2

           for col, val in self._genCatFeatures.items():
              row[col] = val

           anonymized_rows.append(row)

        return anonymized_rows
