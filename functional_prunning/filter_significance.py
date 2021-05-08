import numpy as np
import tensorflow as tf
from collections import OrderedDict

import keras

from tqdm.notebook import trange, tqdm


from cached_property import cached_property

from layer_activation import PrunningMixin

class FilterSignificance(PrunningMixin):

    def __init__(self, model, test_images_x, test_images_y):
        self.model = model
        self.test_images_x = test_images_x
        self.test_images_y = test_images_y
        self.layers_filters_significance = OrderedDict()

    @cached_property
    def test_images(self):
        len = self.test_images_x.shape[0]
        len = 50 if len > 50 else len
        return self.test_images_x[:len, :, :, :]

    @cached_property
    def len_images(self):
        return self.test_images.shape[0]

    @cached_property
    def test_images_labels(self):
        return self.test_images_y[:self.len_images]

    @cached_property
    def image_losses(self):
        losses = np.zeros(shape=(self.len_images))
        for image_i in range(self.len_images):
            x = np.expand_dims(self.test_images[image_i], axis=0)
            y = np.expand_dims(self.test_images_labels[image_i], axis=0)
            losses[image_i] = self.model.evaluate(x, y, verbose=0)[0]
        return losses

    def get_layer_feature_map(self, layer):
        feature_model = keras.Model(inputs=self.model.inputs, outputs=layer.output)
        maps = feature_model.predict(self.test_images)
        return maps

    def compute_significance(self):
        for layer_index in tqdm(self.conv_layers_indices):
            layer = self.model.layers[layer_index]
            layer_feature_map = self.get_layer_feature_map(layer)

            filters_num = layer.get_config()['filters']
            filters_average_gradients = self.get_filters_average_gradients(filters_num, layer_feature_map)

            self.layers_filters_significance[layer.name] = filters_average_gradients

        return self.layers_filters_significance

    def get_filters_average_gradients(self, filters_num, layer_feature_map):
        filters_average_gradients = np.zeros(filters_num)
        for filter_index in trange(filters_num):
            filters_average_gradients[filter_index] = self.get_av_grad(filter_index, layer_feature_map)
        return filters_average_gradients

    def get_av_grad(self, filter_index, layer_feature_map):
        grad_norms = np.zeros(shape=(self.len_images))
        for i in range(self.len_images):
            loss = self.image_losses[i]
            filter_feature_map = layer_feature_map[i, :, :, filter_index]
            grad_norms[i] = self.compute_gradient_norm(loss, filter_feature_map)
        return np.average(grad_norms)

    def compute_gradient_norm(self, loss, filter_feature_map):
        fmap_tensor = tf.convert_to_tensor(filter_feature_map)
        with tf.GradientTape() as tape:
            tape.watch(fmap_tensor)
            loss = loss / tf.reduce_mean(fmap_tensor) if loss != 0 and tf.math.count_nonzero(
                fmap_tensor) != 0 else tf.reduce_mean(fmap_tensor)
        grads = tape.gradient(loss, fmap_tensor)
        norm = np.linalg.norm(grads.numpy())
        return norm