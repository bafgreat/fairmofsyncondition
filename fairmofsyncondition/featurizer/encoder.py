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
import numpy as np
from sentence_transformers import SentenceTransformer, util


def onehot_encoder(data, category):
    '''
    A function that performs one-hot encoding on a given categorical feature.

    This function takes a categorical feature and
    performs one-hot encoding to convert
    it into a numerical representation.
    The function returns a binary matrix where each
    column corresponds to a unique category in the feature.

    Parameters:
        data (list): The categorical feature to encode.
        category (list): The unique categories in the feature.

    Returns:
        numpy.ndarray: A binary matrix representing the
        one-hot encoded feature.
    '''
    onehot = np.zeros((len(data), len(category)))
    for i, cat in enumerate(data):
        onehot[i][category.index(cat)] = 1
    return onehot


def onehot_encoder_pyg(data, category):
    '''
    A function that performs one-hot encoding on a given categorical feature
    and returns a PyTorch tensor compatible with PyTorch Geometric.

    Parameters:
        data (list): The categorical feature to encode.
        category (list): The unique categories in the feature.

    Returns:
        torch.Tensor: A binary tensor representing the
        one-hot encoded feature.
    '''
    onehot = torch.zeros((len(data), len(category)), dtype=torch.float32)
    for i, cat in enumerate(data):
        onehot[i, category.index(cat)] = 1
    return onehot

def reverse_onehot_pyg(onehot_tensor, category):
    '''
    Converts a one-hot encoded PyTorch tensor back to categorical labels.

    Parameters:
        onehot_tensor (torch.Tensor): The one-hot encoded tensor.
        category (list): The original categories corresponding to indices.

    Returns:
        list: The original categorical values.
    '''
    indices = torch.argmax(onehot_tensor, dim=1)
    return [category[i] for i in indices]


def load_sentence_model(list_of_words):
    """
    Function to load a word embedding to facilitate data retrieval

    Parameters:
        list_of_words (list): A list of textual data.

    Returns:
        dict_embeddings (torch.Tensor): A tensor of
        embeddings for the dictionary.

    """
    model = SentenceTransformer('all-MiniLM-L6-v2')
    dict_embeddings = model.encode(list_of_words, convert_to_tensor=True)
    return dict_embeddings, model


def best_embedding_match(query, dictionary_keys, dict_embeddings, model):
    """
    Function to find the most similar embedding in a dictionary
    based on cosine similarity between the query embedding and the embeddings
    in the dictionary.

    Parameters:
        query (str): The text input query.
        dictionary_keys (list): A list of keys in the dictionary.
        dict_embeddings (torch.Tensor): A tensor of embeddings
        for the dictionary.
        model (SentenceTransformer): The SentenceTransformer model
        for embedding generation.

    Returns:
        str: The key from the dictionary that has the highest
        cosine similarity with the query embedding, or the original
        query if no match is found.

    """
    query_emb = model.encode(query, convert_to_tensor=True)
    scores = util.cos_sim(query_emb, dict_embeddings)[0]
    best_idx = torch.argmax(scores).item()
    best_score = scores[best_idx].item()
    print(best_score)
    if best_score >= 0.9:
        return dictionary_keys[best_idx]
    else:
        return query
