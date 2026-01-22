class NodeCluster:
    """
    Generalized cluster for local k-anonymization (SaNGreeA-style).

    - Categorical attributes: dynamic flat hierarchy (value -> '*')
    - Numeric attributes: range-based generalization [min, max]
    """

    def __init__(self, node, dataset=None, adj_list=None, gen_hierarchies=None):
        self._nodes = [node]
        self._dataset = dataset
        self._adjList = adj_list or {}

        # Only range hierarchies are external now
        self._genHierarchies = gen_hierarchies or {"range": {}}

        # Initialize categorical features (store actual value)
        self._genCatFeatures = {}
        for col in self._dataset[node]:
          if col not in self._genHierarchies["range"]:
             self._genCatFeatures[col] = self._dataset[node][col]

        # Initialize numeric range features as [min, max]
        self._genRangeFeatures = {}
        for col in self._genHierarchies["range"]:
            val = self._dataset[node][col]
            self._genRangeFeatures[col] = [val, val]

    # ------------------------------------------------------------------
    # Basic getters
    # ------------------------------------------------------------------

    def getNodes(self):
        return self._nodes

    # ------------------------------------------------------------------
    # Dynamic categorical hierarchy helpers
    # ------------------------------------------------------------------

    def _cat_level(self, val):
        """Hierarchy level: 1 = specific, 0 = generalized"""
        return 0 if val == "*" else 1

    def _cat_generalize(self, val):
        """Generalize categorical value"""
        return "*" if val != "*" else "*"

    def _cat_nr_levels(self):
        """Total number of categorical hierarchy levels"""
        return 2

    # ------------------------------------------------------------------
    # Cluster update
    # ------------------------------------------------------------------

    def addNode(self, node):
        self._nodes.append(node)
        self._adjList[node] = self._adjList.get(node, [])

        # Update categorical features
        for col in self._genCatFeatures:
            self._genCatFeatures[col] = self.computeNewGeneralization(col, node)[1]

        # Update numeric range features
        for col in self._genRangeFeatures:
            current_range = self._genRangeFeatures[col]
            val = self._dataset[node][col]
            self._genRangeFeatures[col] = [
                min(current_range[0], val),
                max(current_range[1], val)
            ]

    # ------------------------------------------------------------------
    # Cost computation
    # ------------------------------------------------------------------

    def computeNodeCost(self, node, alpha=1.0, beta=0.0):
        """
        Compute total cost (GIL + SIL)
        SIL is optional (0 by default)
        """
        gil = self.computeGIL(node)
        sil = 0
        return alpha * gil + beta * sil

    def computeGIL(self, node):
        """Generalization Information Loss"""
        total_cost = 0.0

        # Weights (optional)
        try:
            import globals as GLOB
            weight_vector = GLOB.GEN_WEIGHT_VECTORS[GLOB.VECTOR]
        except Exception:
            weight_vector = {
                "categorical": {col: 1.0 for col in self._genCatFeatures},
                "range": {col: 1.0 for col in self._genRangeFeatures}
            }

        # Categorical cost
        for col in self._genCatFeatures:
            weight = weight_vector["categorical"].get(col, 1.0)
            total_cost += weight * self.computeCategoricalCost(col, node)

        # Numeric cost
        for col in self._genRangeFeatures:
            weight = weight_vector["range"].get(col, 1.0)
            total_cost += weight * self.computeRangeCost(col, node)

        return total_cost

    def computeCategoricalCost(self, col, node):
        cluster_level = self.computeNewGeneralization(col, node)[0]
        return float(
            (self._cat_nr_levels() - cluster_level) / self._cat_nr_levels()
        )

    def computeRangeCost(self, col, node):
        hierarchy = self._genHierarchies["range"][col]
        current_range = self._genRangeFeatures[col]
        val = self._dataset[node][col]
        return hierarchy.getCostOfRange(
            min(current_range[0], val),
            max(current_range[1], val)
        )

    # ------------------------------------------------------------------
    # Generalization logic
    # ------------------------------------------------------------------

    def computeNewGeneralization(self, col, node):
        """
        Returns [new_level, generalized_value] for categorical attributes
        """
        cluster_val = self._genCatFeatures[col]
        node_val = self._dataset[node][col]

        cluster_level = self._cat_level(cluster_val)
        node_level = self._cat_level(node_val)

        while cluster_val != node_val:
            old_node_level = node_level

            if cluster_level <= node_level:
                node_val = self._cat_generalize(node_val)
                node_level -= 1

            if old_node_level <= cluster_level:
                cluster_val = self._cat_generalize(cluster_val)
                cluster_level -= 1

        return [cluster_level, cluster_val]

    # ------------------------------------------------------------------
    # Output helpers
    # ------------------------------------------------------------------

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
        Return a list of dicts representing anonymized rows.
        """
        anonymized_rows = []

        for _ in self._nodes:
          row = {}

          for col, value_range in self._genRangeFeatures.items():
              row[f"{col}_min"] = value_range[0]
              row[f"{col}_max"] = value_range[1]

          for col, val in self._genCatFeatures.items():
              row[col] = val

          anonymized_rows.append(row)

        return anonymized_rows
