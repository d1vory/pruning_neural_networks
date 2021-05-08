import numpy as np
from collections import OrderedDict

from kerassurgeon import Surgeon
from tqdm.notebook import trange, tqdm

from layer_activation import PrunningMixin
from filter_significance import FilterSignificance
from layer_activation import LayerFiltersMaxActivation
from clusterization import LayerFiltersClusterization

class FunctionalPrunning(PrunningMixin):
    def __init__(self, model, x_test, y_test, global_prunning_ratio):
        self.model = model
        self.x_test = x_test
        self.y_test = y_test
        self.global_prunning_ratio = global_prunning_ratio
        self.surgeon = Surgeon(model)
        self.pruned_model = None

        self.input_image_shape = (224, 224, 3)
        self.theta_decay = 0.01
        self.learning_iterations = 150
        self.theta_blur_radius = 0.8
        self.theta_small_norm_percent = 10

        self.layer_activations = OrderedDict()
        self.layer_clusters = OrderedDict()
        self.layer_filter_significance = None

        self.layer_filter_indeces_to_prune = OrderedDict()

    def start_prunning(self):
        print('Starting pruning of the model...')

        print('Clustering activation maps')
        self.get_layers_clusters()
        print('All clustered')

        print('Finding each filter significance')
        self.get_filter_siginificance()
        print('Siginificance found')

        print('Prunning...')
        self.decide_which_filters_to_prune()
        self.pruned_model = self.surgeon.operate()
        return self.pruned_model

    def get_layers_activation_maps(self):
        for layer in self.model.layers:
            activ = LayerFiltersMaxActivation(
                self.model,
                layer,
                input_image_shape=self.input_image_shape,
                theta_decay=self.theta_decay,
                learning_iterations=self.learning_iterations,
                theta_blur_radius=self.theta_blur_radius,
                theta_small_norm_percent=self.theta_small_norm_percent
            )
            activ.compute_layer_filters_max_activations()
            self.layer_activations[layer.name] = activ
        return self.layer_activations

    def get_layers_clusters(self):
        for layer_index in self.conv_layers_indices:
            layer = self.model.layers[layer_index]
            print(f'Computing activation map for layer {layer.name}')
            activ = LayerFiltersMaxActivation(
                self.model,
                layer,
                input_image_shape=self.input_image_shape,
                theta_decay=self.theta_decay,
                learning_iterations=self.learning_iterations,
                theta_blur_radius=self.theta_blur_radius,
                theta_small_norm_percent=self.theta_small_norm_percent
            )
            activ.compute_layer_filters_max_activations()
            clustering = LayerFiltersClusterization(self.model, activ)

            print(f'Computing cluster k search for layer {layer.name}')
            clustering.perform_k_search()
            self.layer_clusters[layer.name] = clustering.images_indices_in_clusters
        return self.layer_clusters

    def get_filter_siginificance(self):
        fs = FilterSignificance(self.model, self.x_test, self.y_test)
        self.layer_filter_significance = fs.compute_significance()
        return self.layer_filter_significance

    def decide_which_filters_to_prune(self):
        for layer_index in self.conv_layers_indices:
            layer = self.model.layers[layer_index]
            significance = self.layer_filter_significance[layer.name]
            layer_prunning_ratio = self.global_prunning_ratio
            argsorted = np.argsort(significance)
            candidates_to_prune_indeces = argsorted[:argsorted.shape[0] * layer_prunning_ratio]
            clusters = np.array(self.layer_clusters[layer.name])
            to_prune = [i for i in candidates_to_prune_indeces if i in clusters]
            self.surgeon.add_job('delete_channels', layer, to_prune)

