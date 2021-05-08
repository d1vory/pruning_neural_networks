import numpy as np
import tensorflow as tf
import keras
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from keras import layers, models, losses


class Prunning:
    def __init__(self, model, percent_to_prune = 0.1):
        self.init_model = model
        self.layers = self.init_model.layers
        self.layers_num = len(self.init_model.layers)
        self.conv_layers_indices = self.get_conv_layers_indices()
        assert ((percent_to_prune > 0) and
                    (percent_to_prune < 1)), 'the percentage of filters to prune should be greater than 0 and less than 1'

        self.percent_to_prune = percent_to_prune

        self.layers_weights = self.get_layers_info()
        self.model_config = self.init_model.get_config()

    def get_layers_info(self):
        layers_weights = dict()
        for layer in self.layers:
            layers_weights[layer.name] = layer.get_weights()
        return layers_weights

    def find_config_index_by_name(self, name):
        for i in range(len(self.model_config['layers'])):
            if self.model_config['layers'][i]['config']['name'] == name:
                return i

    def get_conv_layers_indices(self):
        """
        returns list of indices of convolutional layers in model
        """
        res = []
        for i, layer in enumerate(self.layers):
            # print(layer.name)
            if 'conv2d' in layer.name:
                res.append(i)

        return res

    def sum_of_absolute_kernel_weights(self, kernel: np.ndarray):
        """
        calculate the sum of its absolute kernel weights
        """
        return np.sum(np.abs(kernel))

    def sum_of_filter(self, _filter):
        """
        calculate the sum of the sum of absolute kernel weights
        on a filter
        """
        res = np.zeros(shape=(_filter.shape[-1]))
        for i in range(_filter.shape[-1]):
            res[i] = self.sum_of_absolute_kernel_weights(_filter[:, :, i])
        return np.sum(res)

    def prune_filters(self, m: int, axis: int, indeces_to_prune: np.ndarray, weights: np.ndarray):
        """
        deletes m filters along given axis that correspond to indices in weights matrix
        """
        pruned_weights = np.delete(weights, indeces_to_prune, axis=axis)
        return pruned_weights

    def assign_weights(self, layer_name, weights):
        """
        assigns weights and set new number of filters for layer
        """
        self.layers_weights[layer_name] = weights

    def assign_filters_num(self, layer_name, new_filters_num):
        layer_config_index = self.find_config_index_by_name(layer_name)
        self.model_config['layers'][layer_config_index]['config']['filters'] = new_filters_num
        # self.layers_configs[layer_name]['filters'] = new_filters_num

    def process_conv_layer(self, layer_index: int, next_conv_layer_index: int = None):
        layer = self.layers[layer_index]
        all_weights = self.layers_weights[layer.name]
        weights = all_weights[0]
        biases = all_weights[1]

        output_depth = weights.shape[3]

        sum_of_weights = np.zeros(shape=(output_depth))
        for j in range(output_depth):
            sum_of_weights[j] = self.sum_of_filter(weights[:, :, :, j])
        sorted_indeces = np.argsort(sum_of_weights)

        m = int(output_depth * self.percent_to_prune)
        resulted_filters_num = output_depth - m
        indeces_to_prune = sorted_indeces[:m]

        pruned_weights = self.prune_filters(m, 3, indeces_to_prune, weights)
        pruned_biases = self.prune_filters(m, 0, indeces_to_prune, biases)

        self.assign_weights(layer.name, [pruned_weights, pruned_biases])
        self.assign_filters_num(layer.name, resulted_filters_num)

        self.remove_corresponding_channels_until_next_conv(m, resulted_filters_num, indeces_to_prune, layer_index + 1,
                                                           next_conv_layer_index)

        # prune 3 channel of next conv layer
        if next_conv_layer_index is not None:
            next_conv_layer = self.layers[next_conv_layer_index]
            all_weights = self.layers_weights[next_conv_layer.name]
            weights = all_weights[0]
            biases = all_weights[1]

            weights = self.prune_filters(m, 2, indeces_to_prune, weights)
            self.assign_weights(next_conv_layer.name, [weights, biases])

    def print_model_weights_shape(self):
        for lname, lweights in self.layers_weights.items():
            for w in lweights:
                print(f"{lname}: {w.shape}")

    def remove_corresponding_channels_until_next_conv(self, m, new_filters_num, indeces_to_prune, start_index,
                                                      stop_index):
        """
        removes corresponding weights in following layers until stop index
        """
        print(f'corresponding: start:{start_index}  end:{stop_index}')
        if stop_index is None:
            # conv layers ended and we should just prune next layer
            for i in range(start_index, len(self.layers)):
                # find first dense layer after conv layers
                if 'dense' in self.layers[i].name:
                    stop_index = i + 1
                    break

        for i in range(start_index, stop_index):
            # prune non conv layers
            layer = self.layers[i]
            class_name = self.init_model.get_config()['layers'][i + 1]['class_name']
            # print(f'i: {i}, class name: {class_name}, layer name: {layer.name}')

            if class_name == 'BatchNormalization':
                new_weights = [self.prune_filters(m, 0, indeces_to_prune, old_w) for old_w in
                               self.layers_weights[layer.name]]
                self.assign_weights(layer.name, new_weights)

            elif class_name == 'Dense':
                all_weights = self.layers_weights[layer.name]
                input = all_weights[0]
                output = all_weights[1]
                new_input = self.prune_filters(m, 0, indeces_to_prune, input)
                self.assign_weights(layer.name, [new_input, output])

    def prepare_weights(self):
        res = []
        for layer in self.layers:
            for w in self.layers_weights[layer.name]:
                res.append(w)
        return res

    def return_new_pruned_model(self):
        new_model = keras.Sequential.from_config(self.model_config)
        prepared_weights = self.prepare_weights()
        new_model.set_weights(prepared_weights)
        return new_model

    def prune_model(self):
        for i in range(len(self.conv_layers_indices)):
            # print(f'iteration: {i}')
            try:
                self.process_conv_layer(self.conv_layers_indices[i], self.conv_layers_indices[i + 1])
            except IndexError:
                self.process_conv_layer(self.conv_layers_indices[i], None)

        return self.return_new_pruned_model()