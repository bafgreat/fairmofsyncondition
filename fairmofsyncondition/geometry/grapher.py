#!/usr/bin/python
from __future__ import print_function
__author__ = "Dr. Dinga Wonanke"
__status__ = "production"

##############################################################################
# fairmofsyncondition is a machine learning package for predicting the        #
# synthesis condition of the crystal structures of MOFs. It is also intended  #
# for predicting all MOFs the can be generated from a given set of conditions #
# In addition the package also predicts the stability of MOFs, compute their  #
# their PXRD and crystallite sizes. This package is part of our effort to     #
# to accelerate the discovery and optimization of the synthesises of novel    #
# high performing MOFs. This package is being developed by Dr Dinga Wonanke   #
# as part of hos MSCA post doctoral fellowship at TU Dresden.                 #
#                                                                             #
###############################################################################

import torch


class SimilarityFinder:
    """
    A class to compute the similarity between two ASE Atoms graph systems.
    These  graphs corresponds to AtomGraph from orbital-materials module
    which provides and efficient way to convert ase atoms to graphs. This is
    particularly useful for converting ase crystal structures to graphs for
    directly comparison.

    This class takes two atom graph representations (graph1 and graph2),
    and computes a similarity index between them based on node features,
    edge features, and other properties of the graph.

    The approach used to compute similarity involves comparing key features of
    the two AtomGraphs. These features include the number of nodes (representing
    atoms), edges (representing bonds or connections between atoms), atomic numbers,
    node embeddings, and edge properties such as vectors and the relationships between
    senders and receivers of these edges. The senders and receivers here correspond to
    the indices of atoms in the graph.

    - The sender is the starting atom of a bond
    - The receiver is the atom that the bond points to

    Moreover, incases where features of both graph differe in size, a padding strategy
    is applied to equalize their lengths to optimise the comparison between the graphs.
    Each feature difference is normalized, weighted based
    on its importance, and then combined to form a final similarity index that ranges
    from 0 to 1.
    - 1 indicates perfect similarity.
    - 0 indicates no similarity.
    """

    def __init__(self, graph1, graph2):
        """
        Initialize the SimilarityFinder with two graphs.

        **parameters:**
            - graph1: The first atom graph (an instance of an AtomGraph or similar structure).
            - graph2: The second atom graph (an instance of an AtomGraph or similar structure).
        """
        self.graph1 = graph1
        self.graph2 = graph2

    @staticmethod
    def normalize(value, max_value):
        """
        Normalize a value between 0 and 1.

        **parameters:**
            - value: The value to normalize.
            - max_value: The maximum possible value for normalization.

        **returns:**
            - Normalized value between 0 and 1.
        """
        return min(value / (max_value + 1e-6), 1.0)

    @staticmethod
    def pad_targent(tensor, target_length):
        """
        Pad a tensor to the target length. Neccessary to
        improve the comparison between two features of the
        graphs.

        **parameters:**
            - tensor: The input tensor to pad or truncate.
            - target_length: The target length to match.

        **returns:**
            - A tensor of length equal to target_length.
        """
        current_length = tensor.shape[0]
        if current_length == target_length:
            return tensor
        elif current_length < target_length:
            padding = torch.zeros(
                (target_length - current_length,) + tensor.shape[1:], dtype=tensor.dtype)
            return torch.cat([tensor, padding], dim=0)
        else:
            return tensor[:target_length]

    def compute_similarity_index(self, weights=None):
        """
        Compute a similarity index between two AtomGraphs using a padding strategy to equalize sizes.

        **parameters:**
            - weights: Optional dictionary to assign different importance to various similarity metrics.

        **returns:**
            - A similarity index between 0 and 1.
        """

        if weights is None:
            weights = {
                'n_nodes_diff': 0.1,
                'n_edges_diff': 0.1,
                'atomic_numbers_diff': 0.4,
                'embedding_diff': 0.2,
                'senders_diff': 0.1,
                'receivers_diff': 0.1,
                'edge_vectors_diff': 0.1,
                'positions_diff': 0.1
            }

        # - Check number of nodes and edges
        n_nodes_diff = torch.abs(
            self.graph1.n_node - self.graph2.n_node).item()
        n_edges_diff = torch.abs(
            self.graph1.n_edge - self.graph2.n_edge).item()

        # - Compare atomic numbers
        atomic_numbers_1 = self.graph1.node_features['atomic_numbers']
        atomic_numbers_2 = self.graph2.node_features['atomic_numbers']

        max_len_atomic = max(len(atomic_numbers_1), len(atomic_numbers_2))
        atomic_numbers_1_padded = self.pad_targent(
            atomic_numbers_1, max_len_atomic)
        atomic_numbers_2_padded = self.pad_targent(
            atomic_numbers_2, max_len_atomic)
        atomic_diff = torch.sum(atomic_numbers_1_padded !=
                                atomic_numbers_2_padded).item() / max_len_atomic

        # - Compare node embeddings
        embeddings_1 = self.graph1.node_features['atomic_numbers_embedding']
        embeddings_2 = self.graph2.node_features['atomic_numbers_embedding']

        max_len_embed = max(embeddings_1.shape[0], embeddings_2.shape[0])
        embeddings_1_padded = self.pad_targent(embeddings_1, max_len_embed)
        embeddings_2_padded = self.pad_targent(embeddings_2, max_len_embed)
        embedding_diff = torch.norm(
            embeddings_1_padded - embeddings_2_padded, p='fro').item()

        # - Compare edge senders and receivers
        senders_1 = self.graph1.senders
        senders_2 = self.graph2.senders
        receivers_1 = self.graph1.receivers
        receivers_2 = self.graph2.receivers

        max_len_senders = max(len(senders_1), len(senders_2))
        senders_1_padded = self.pad_targent(senders_1, max_len_senders)
        senders_2_padded = self.pad_targent(senders_2, max_len_senders)
        senders_diff = torch.sum(
            senders_1_padded != senders_2_padded).item() / max_len_senders

        max_len_receivers = max(len(receivers_1), len(receivers_2))
        receivers_1_padded = self.pad_targent(
            receivers_1, max_len_receivers)
        receivers_2_padded = self.pad_targent(
            receivers_2, max_len_receivers)
        receivers_diff = torch.sum(
            receivers_1_padded != receivers_2_padded).item() / max_len_receivers

        # - Compare edge vectors
        edge_vectors_1 = self.graph1.edge_features['vectors']
        edge_vectors_2 = self.graph2.edge_features['vectors']

        max_len_edges = max(edge_vectors_1.shape[0], edge_vectors_2.shape[0])
        edge_vectors_1_padded = self.pad_targent(
            edge_vectors_1, max_len_edges)
        edge_vectors_2_padded = self.pad_targent(
            edge_vectors_2, max_len_edges)
        edge_diff = torch.norm(edge_vectors_1_padded -
                               edge_vectors_2_padded, p='fro').item()

        # - Compare positions (only if the number of nodes match)
        positions_1 = self.graph1.node_features['positions']
        positions_2 = self.graph2.node_features['positions']

        max_len_pos = max(positions_1.shape[0], positions_2.shape[0])
        positions_1_padded = self.pad_targent(positions_1, max_len_pos)
        positions_2_padded = self.pad_targent(positions_2, max_len_pos)
        position_diff = torch.norm(
            positions_1_padded - positions_2_padded, p='fro').item()

        # - Normalize the differences
        n_nodes_sim = 1 - self.normalize(n_nodes_diff, max_value=max(
            self.graph1.n_node.item(), self.graph2.n_node.item()))
        n_edges_sim = 1 - self.normalize(n_edges_diff, max_value=max(
            self.graph1.n_edge.item(), self.graph2.n_edge.item()))
        atomic_numbers_sim = 1 - self.normalize(atomic_diff, max_value=1.0)
        embedding_sim = 1 - \
            self.normalize(embedding_diff, max_value=embedding_diff + 1e-6)
        senders_sim = 1 - senders_diff
        receivers_sim = 1 - receivers_diff
        edge_vectors_sim = 1 - \
            self.normalize(edge_diff, max_value=edge_diff + 1e-6)
        position_sim = 1 - \
            self.normalize(position_diff, max_value=position_diff + 1e-6)

        # - Combine similarities into a single index
        similarity_index = (
            weights['n_nodes_diff'] * n_nodes_sim +
            weights['n_edges_diff'] * n_edges_sim +
            weights['atomic_numbers_diff'] * atomic_numbers_sim +
            weights['embedding_diff'] * embedding_sim +
            weights['senders_diff'] * senders_sim +
            weights['receivers_diff'] * receivers_sim +
            weights['edge_vectors_diff'] * edge_vectors_sim +
            weights['positions_diff'] * position_sim
        )

        return min(similarity_index, 1.0)
