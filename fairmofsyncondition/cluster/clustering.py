import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import ListedColormap
from sklearn.neighbors import NearestNeighbors
from fairmofsyncondition.read_write import filetyper

def predict_kmeans(data, num_clusters, random_state=0):
    model = KMeans(n_clusters=num_clusters, random_state=random_state)
    model.fit(data)
    labels = model.predict(data)
    centroids = model.cluster_centers_
    return labels, centroids, model.inertia_


class Cluster:
    def __init__(self, data, random_state=0):
        self.data = data

    def find_number_of_clusters(self, path_to_save='number_of_clusters.png', show_plot=False):
        num_clusters = range(1, 30)
        inertias = []
        for k in num_clusters:
            _, _, inertia_ = self.predict_kmeans(self.data, k)
            inertias.append(inertia_)

        plt.figure(figsize=(8, 6))
        plt.plot(num_clusters, inertias, '-o')
        plt.xlabel('Number of Clusters (k)', fontsize=17)
        plt.ylabel('Inertia', fontsize=17)
        plt.tick_params(axis='both', which='major', labelsize=14)
        plt.savefig(path_to_save, dpi=600)
        if show_plot:
            plt.show()
        plt.clf()

    def plot_clustering(self, num_clusters, path_to_save='cluster_data', colors=None, marker='D'):
        if not os.path.exists(path_to_save):
            os.makedirs(path_to_save)

        scaler = StandardScaler()
        self.scaled_data = scaler.fit_transform(self.data)
        self.labels, self.centroids, _ = predict_kmeans(self.scaled_data, num_clusters)
        self.centroids_original = scaler.inverse_transform(self.centroids)
        if colors is None:
            colors = plt.cm.viridis(np.linspace(0, 1, num_clusters))

        plt.figure(figsize=(10, 6))
        plt.scatter(self.data[:, 0], self.data[:, 0], c=self.labels, cmap=ListedColormap(
            colors), alpha=0.7, marker='o')

        plt.scatter(self.centroids_original[:, 0], self.centroids_original[:, 0],
                    c='black', s=60, marker=marker, label='Centroids')

        plt.savefig(f'{path_to_save}/clusters_clusters.png', dpi=600)
        plt.show()
        plt.clf()

    def extract_neighbors(self, dict_value):
        json_data = {}
        neighbors_data = {}
        tmp = []
        tmp2 = {}
        nbrs = NearestNeighbors(n_neighbors=11).fit(self.data)
        for i, centroid in enumerate(self.centroids_original):
            distances = np.linalg.norm(self.data - centroid, axis=1)
            closest_index = np.argmin(distances)
            refcode_1 = dict_value[closest_index]
            tmp.append(
                {"refcode": refcode_1, 'energy': self.data[closest_index].tolist()[0]})
            neighbor_distances, indices = nbrs.kneighbors([centroid])
            neighbors = indices[0]
            tmp_neighbors = []
            for idx in neighbors:
                refcode = dict_value[idx]
                tmp_neighbors.append(
                    {"refcode": refcode, 'energy': self.data[idx].tolist()[0]})
            tmp2[i+1] = tmp_neighbors
        json_data.update(tmp)
        neighbors_data.update(tmp2)

energy_data = filetyper.load_data('../../data/compiled_ligand_bde_energy.json')

data = np.array(list(energy_data.values())).reshape(-1,1)
clustering = Cluster(data)
# clustering.find_number_of_clusters(show_plot=True)
clustering.plot_clustering(num_clusters=5)