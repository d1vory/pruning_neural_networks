import cv2
import numpy as np

import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from keras import applications
from sklearn.cluster import KMeans
from pylab import percentile, tile
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from tqdm.notebook import trange, tqdm

from cached_property import cached_property
from tensorflow_addons.image import gaussian_filter2d
from progress.bar import Bar
from kerassurgeon import Surgeon

from layer_activation import LayerFiltersMaxActivation

class LayerFiltersClusterization:
    def __init__(self, model, layer_max_activation: LayerFiltersMaxActivation):
        self.model = model
        self.layer = layer_max_activation.layer

        self.filters_activation_maps = np.array(layer_max_activation.filters_activation_maps)
        self.filters_num = layer_max_activation.filters_number

        self.clusters_num = 1
        self.labels = []
        self.score = None
        self.kmeans = None

    @cached_property
    def flattened_activation_maps(self):
        old_shape = self.filters_activation_maps.shape
        # print(old_shape)
        newshape = (old_shape[0], -1)

        flattened_maps = np.reshape(self.filters_activation_maps, newshape)
        # print(flattened_maps.shape)

        scaled = StandardScaler().fit_transform(flattened_maps)
        variance = 0.999
        pca = PCA(variance)
        pca.fit(scaled)

        transformed = pca.transform(scaled)
        # print(transformed.shape)
        return transformed
        # return flattened_maps

    @cached_property
    def images_indices_in_clusters(self):
        arr = [[] for _ in range(self.clusters_num)]
        for filter_index, cluster_index in enumerate(self.labels):
            arr[cluster_index].append(filter_index)
        return arr

    def clasterize_activation_maps(self, k, max_iter=300, random_state=0):
        kmeans = KMeans(k, max_iter=max_iter, random_state=random_state).fit(self.flattened_activation_maps)
        labels = kmeans.predict(self.flattened_activation_maps)
        score = kmeans.score(self.flattened_activation_maps)
        inertia = kmeans.inertia_
        return kmeans, labels, score, inertia

    def perform_k_search(self, max_iter=300, random_state=0):
        max_k = self.filters_num // 2
        sil = []
        for k in trange(2, max_k, desc='Filter clustering'):
            kmeans = KMeans(k, max_iter=max_iter, random_state=random_state).fit(self.flattened_activation_maps)
            labels = kmeans.labels_
            sil.append(silhouette_score(self.flattened_activation_maps, labels, metric='euclidean'))

        optimal_k = np.argmax(sil) + 2
        self.kmeans, self.labels, self.score, _ = self.clasterize_activation_maps(optimal_k)
        self.clusters_num = optimal_k

        return self.labels

    def show_images_in_cluster(self, cluster_index):
        imgs = np.hstack([self.filters_activation_maps[filter_index] for filter_index in
                          self.images_indices_in_clusters[cluster_index]])

        cv2.imshow(imgs)

    def show_images_in_clusters(self):
        plt.rcParams["figure.figsize"] = (40, 6)
        plt.rcParams["figure.dpi"] = 300
        mylen = 10 if self.clusters_num > 10 else self.clusters_num
        for cluster_index in range(mylen):
            imgs_in_cluster = np.hstack([self.filters_activation_maps[filter_index] for filter_index in
                                         self.images_indices_in_clusters[cluster_index]])
            # title=f'cluster index: {cluster_index}'
            # Row, column, index
            plt.subplot(self.clusters_num, 1, cluster_index + 1)

            plt.imshow(imgs_in_cluster)
            # plt.title(title,fontsize=8)
            plt.xticks([])
            plt.yticks([])
        plt.show()