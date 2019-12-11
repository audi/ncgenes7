# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

"""
Utils for keras library
"""
from nucleus7.utils.model_utils import KerasLayersMixin
import tensorflow as tf


class KerasCompositionalLayer(tf.keras.layers.Layer, KerasLayersMixin):
    """
    The same class as the base keras layer, but it allows to reset the
    added using self.add_keras_layer interface layers to this layer
    inside of self.build
    """

    def build(self, input_shape):
        self.reset_keras_layers()
        super().build(input_shape)
