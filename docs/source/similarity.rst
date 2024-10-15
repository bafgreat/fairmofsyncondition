===============================
SimilarityFinder Module
===============================

Introduction
============
The `SimilarityFinder` class is designed to compute the similarity between two crystal structures represented as AtomGraphs, a graph-based representation of molecular structures. This graph conversion is useful for directly comparing crystal structures by encoding their atomic and bond features into nodes and edges.

The computed similarity index quantifies how closely two structures resemble each other. It leverages differences in graph properties such as the number of nodes (atoms), edges (bonds), atomic numbers, node embeddings, and edge properties.

Similarity Index
================
The `SimilarityFinder` class computes a similarity index using the following approach:

1. **Graph Comparison**: The two input structures are first converted into AtomGraphs. These graphs store:
    - Nodes, representing atoms in the structure.
    - Edges, representing the bonds or interactions between atoms.
    - Node features, such as atomic numbers and embeddings.
    - Edge features, such as bond vectors and relationships between the sender and receiver atoms.

2. **Padding Strategy**: In cases where the graphs have different sizes (e.g., varying number of atoms or bonds), the smaller graph is padded with zeros to equalize their lengths to ensure a fair comparison.

3. **Weighted Metrics**: Various features are compared (e.g., atomic numbers, bond vectors, node embeddings) and normalized differences are weighted by importance. Each difference is transformed into a similarity score (ranging from 0 to 1), where:
    - 1 indicates perfect similarity.
    - 0 indicates no similarity.

4. **Final Similarity Score**: The individual similarity scores are then combined to produce a single similarity index, with the default weights emphasizing the relative importance of each feature.


Example
=======

Below is an example usage of the `SimilarityFinder` class:

```python
# Import necessary libraries
import torch
from fairmofsyncondition. geometry.grapher import SimilarityFinder

# Assume graph1 and graph2 are AtomGraph objects
graph1 = ...
graph2 = ...

# Initialize the SimilarityFinder
similarity_finder = SimilarityFinder(graph1, graph2)

# Compute similarity index
similarity_index = similarity_finder.compute_similarity_index()

print(f"Similarity Index: {similarity_index}")
```