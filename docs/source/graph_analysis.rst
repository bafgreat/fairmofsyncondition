===============================
MOFNetworkAnalysis Module
===============================

Overview
========
The `MOFNetworkAnalysis` module is a tool for performing social network analysis on Metal-Organic Frameworks (MOFs)
based on their structural similarities. The module constructs a graph where MOFs are represented as nodes,
and the edges between them represent the similarity relationships derived from a similarity score. This enables the identification of key MOFs,
detection of communities (clusters), and visual representation of the relationships among MOFs.

The module uses data from an SQLite database, performs network analysis using graph-based algorithms,
and outputs both a visual network diagram and a JSON file containing the detected communities of MOFs.

Key features of the module include:
- **Graph Construction**: MOFs are represented as nodes and connected by edges based on their similarity.
- **Centrality Computation**: Degree, closeness, and betweenness centrality metrics are calculated to identify key MOFs in the network.
- **Community Detection**: The module detects and clusters MOFs into communities based on their similarity relationships.
- **Visualization**: The network is visualized with colored communities, allowing easy identification of clusters.
- **JSON Output**: Detected MOF communities are saved to a JSON file for further analysis or sharing.


Usage Example
=============
Below is an example of how to use the `MOFNetworkAnalysis` module to analyze MOF similarity data stored in an SQLite database:

```python
from mof_network_analysis import MOFNetworkAnalysis

# Initialize the analysis with a database and similarity threshold
analysis = MOFNetworkAnalysis('json_datai2.db', similarity_threshold=0.7)

# Run the complete analysis workflow
analysis.run_analysis()
