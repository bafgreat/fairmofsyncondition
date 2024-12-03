import sqlite3
import random
import json
import pandas as pd
import matplotlib.pyplot as plt
from networkx.algorithms import community
import networkx as nx


class MOFNetworkAnalysis:
    """
    A class for performing social network analysis on MOF similarity data.

    Attributes:
    ----------
    db_name : str
        The name of the SQLite database containing MOF similarity data.
    similarity_threshold : float
        The threshold for filtering similarity edges between MOFs.
    G : nx.Graph
        The graph representation of MOFs and their similarity relationships.
    unique_mofs : list
        A list of communities (clusters) of MOFs.
    filtered_data : pd.DataFrame
        The filtered similarity data used to create the graph.
    """

    def __init__(self, db_name, similarity_threshold=0.5):
        """
        Initialize the MOFNetworkAnalysis class with a database and similarity threshold.

        Parameters:
        ----------
        db_name : str
            The SQLite database file containing MOF similarity data.
        similarity_threshold : float
            The minimum similarity score to consider when creating the network.
        """
        self.db_name = db_name
        self.similarity_threshold = similarity_threshold
        self.G = nx.Graph()
        self.unique_mofs = []
        self.filtered_data = None

    def load_data(self):
        """
        Load MOF similarity data from the SQLite database and filter by similarity threshold.
        """
        conn = sqlite3.connect(self.db_name)
        query = '''
        SELECT mof_name, similar_mof, similarity FROM json_data
        '''
        data = pd.read_sql_query(query, conn)
        conn.close()

        # Filter edges based on the similarity threshold
        self.filtered_data = data[data['similarity'] > self.similarity_threshold]

    def create_graph(self):
        """
        Create a graph where MOFs are nodes, and edges represent similarity relationships.
        """
        for index, row in self.filtered_data.iterrows():
            self.G.add_edge(row['mof_name'], row['similar_mof'], weight=row['similarity'])

    def compute_centrality(self):
        """
        Compute centrality metrics: degree centrality, closeness centrality, and betweenness centrality.

        Returns:
        -------
        dict
            A dictionary containing the centrality metrics for each node in the graph.
        """
        degree_centrality = nx.degree_centrality(self.G)
        closeness_centrality = nx.closeness_centrality(self.G)
        betweenness_centrality = nx.betweenness_centrality(self.G)

        return {
            'degree_centrality': degree_centrality,
            'closeness_centrality': closeness_centrality,
            'betweenness_centrality': betweenness_centrality
        }

    def detect_communities(self):
        """
        Identify communities (clusters) of MOFs using a community detection algorithm.

        Returns:
        -------
        list
            A list of communities, where each community is a list of MOFs.
        """
        communities = community.greedy_modularity_communities(self.G, weight='weight')
        self.unique_mofs = [list(community) for community in communities]
        return self.unique_mofs

    def visualize_network(self, centrality_metrics, output_image='mof_network.png'):
        """
        Visualize the MOF network with colored communities and save the figure.

        Parameters:
        ----------
        centrality_metrics : dict
            Centrality metrics for each node in the graph.
        output_image : str
            The file name for saving the network visualization image.
        """
        pos = nx.spring_layout(self.G)

        # Generate random colors for each community
        colors = [f'#{random.randint(0, 0xFFFFFF):06x}' for _ in range(len(self.unique_mofs))]

        plt.figure(figsize=(12, 10))

        for i, community_nodes in enumerate(self.unique_mofs):
            node_color = colors[i]
            nx.draw_networkx_nodes(self.G, pos, nodelist=community_nodes, node_size=50, node_color=node_color)

        # Draw edges and labels
        nx.draw_networkx_edges(self.G, pos, alpha=0.5)
        nx.draw_networkx_labels(self.G, pos, font_size=8)

        plt.title('MOF Network with Centrality and Communities')

        # Save the figure
        plt.savefig(output_image, dpi=300, bbox_inches='tight')
        plt.show()

    def save_unique_mofs_as_json(self, output_json='unique_mofs.json'):
        """
        Save the unique MOFs and their community groupings as a JSON file.

        Parameters:
        ----------
        output_json : str
            The file name for saving the unique MOFs as a JSON file.
        """
        mof_dict = {f'Community {i+1}': community for i, community in enumerate(self.unique_mofs)}

        # Save the dictionary as a JSON file
        with open(output_json, 'w') as json_file:
            json.dump(mof_dict, json_file, indent=4)

    def run_analysis(self):
        """
        Run the complete MOF network analysis: load data, create graph,
        compute centrality,
        detect communities, visualize the network, and save the results.
        """
        self.load_data()
        self.create_graph()
        centrality_metrics = self.compute_centrality()
        self.detect_communities()
        self.visualize_network(centrality_metrics)
        self.save_unique_mofs_as_json()



if __name__ == "__main__":
    analysis = MOFNetworkAnalysis('../../data/json_datai3.db', similarity_threshold=0.5)
    analysis.run_analysis()
