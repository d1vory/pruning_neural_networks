import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import math

import keras
from tqdm.notebook import trange, tqdm

from tensorflow_addons.image import gaussian_filter2d



class PrunningMixin:

    @property
    def conv_layers_indices(self):
        return [i for i, layer in enumerate(self.model.layers) if 'conv2d' in layer.name]


class LayerFiltersMaxActivation:
    def __init__(
            self,
            model,
            layer,
            input_image_shape=(180, 180, 3),
            learning_rate=10.0,
            learning_iterations=30,
            filters=None,
            px_benefit_percentile=0,
            theta_decay=0.0001,
            blur_every_n_step=4,
            theta_blur_radius=1.0,
            theta_small_norm_percent=0
    ):
        self.model = model
        self.layer = layer
        self.filters_activation_maps = []
        self.filters_activation_losses = []

        self.feature_extractor = keras.Model(inputs=model.input, outputs=self.layer.output)
        self.input_image_shape = input_image_shape
        self.filters_number = self.layer.get_config()['filters'] if filters is None else filters
        self.learning_rate = learning_rate
        self.learning_iterations = learning_iterations

        self.px_benefit_percentile = px_benefit_percentile
        self.theta_decay = theta_decay
        self.regularizer = keras.regularizers.L2(self.theta_decay)
        self.blur_every_n_step = blur_every_n_step
        self.theta_blur_radius = theta_blur_radius
        self.theta_small_norm_percent = theta_small_norm_percent
        # self.L2Regulizer = keras.Model()

    def compute_layer_filters_max_activations(self):

        for filter_index in trange(self.filters_number, desc='Filter computing'):
            loss, img = self.filter_synthesis(filter_index)
            self.filters_activation_maps.append(img)
            self.filters_activation_losses.append(loss)

    def filter_synthesis(self, filter_index, grads=None):
        img = self.initialize_image()
        for iteration in range(self.learning_iterations):
            # img = tf.clip_by_value(img, clip_value_min=0, clip_value_max=255)
            loss, img = self.gradient_ascent_step(img, filter_index, self.learning_rate)

            img *= (1 - self.theta_decay)
            if self.px_benefit_percentile > 0:
                img = self.clip_small_contrib_pixels(grads, img)
            img = self.apply_gaussisan_filter(img, iteration)
            if self.theta_small_norm_percent > 0:
                img = self.clip_small_norm_pixels(img)
        img = self.denormalize_image(img[0].numpy())
        return loss, img

    def initialize_image(self):
        img = tf.random.uniform((1, *self.input_image_shape))
        # img = tf.random.uniform(shape=(1, *self.input_image_shape), minval=0, maxval=255)
        return img

    @tf.function
    def gradient_ascent_step(self, img, filter_index, learning_rate):
        with tf.GradientTape() as tape:
            tape.watch(img)
            loss = self.compute_loss(img, filter_index)
        grads = tape.gradient(loss, img)
        grads = tf.math.l2_normalize(grads)
        img += learning_rate * grads
        return loss, img

    def compute_loss(self, input_image, filter_index):
        activation = self.feature_extractor(input_image)
        filter_activation = activation[:, 2:-2, 2:-2, filter_index]
        return tf.reduce_mean(filter_activation)  # + self.regularizer(filter_activation)

    def clip_small_contrib_pixels(self, grads, xx):
        pred_0_benefit = grads * (-xx)
        px_benefit = tf.math.reduce_sum(pred_0_benefit, axis=3)
        smallben = px_benefit < tfp.stats.percentile(px_benefit, self.px_benefit_percentile)
        smallben3 = tf.tile(tf.expand_dims(smallben, axis=3), tf.constant([1, 1, 1, 3], tf.int32))
        kek = tf.cast(smallben3, tf.float32)
        return xx * kek

    def apply_gaussisan_filter(self, img, step):
        if self.blur_every_n_step is not 0 and self.theta_blur_radius > 0 and step % self.blur_every_n_step == 0:
            return gaussian_filter2d(img, sigma=self.theta_blur_radius)
        return img

    def clip_small_norm_pixels(self, img):
        norms = tf.norm(img, axis=3)
        smallpx = norms < tfp.stats.percentile(norms, self.theta_small_norm_percent)
        smallpx32 = tf.tile(tf.expand_dims(smallpx, axis=3), tf.constant([1, 1, 1, 3], tf.int32))
        smallpx32 = tf.cast(smallpx32, tf.float32)
        return img - img * smallpx32

    def denormalize_image(self, img):
        img -= img.mean()
        img /= img.std() + 1e-5
        img *= 0.15
        img = img[25:-25, 25:-25, :]
        img += 0.5
        img = np.clip(img, 0, 1)
        img *= 255
        img = np.clip(img, 0, 255).astype("uint8")

        return img

    def show_images(self):
        margin = 5
        n = int(math.sqrt(self.filters_number))
        n = 16 if n > 17 else n
        cropped_width = self.input_image_shape[0] - 25 * 2
        cropped_height = self.input_image_shape[1] - 25 * 2
        width = n * cropped_width + (n - 1) * margin
        height = n * cropped_height + (n - 1) * margin
        stitched_filters = np.zeros((width, height, 3))
        for i in range(n):
            for j in range(n):
                img = self.filters_activation_maps[i * n + j]
                stitched_filters[
                (cropped_width + margin) * i: (cropped_width + margin) * i + cropped_width,
                (cropped_height + margin) * j: (cropped_height + margin) * j
                                               + cropped_height,
                :,
                ] = img
        keras.preprocessing.image.save_img("filters.png", stitched_filters)

        from IPython.display import Image, display

        display(Image("filters.png"))



