# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

"""
Plugins as dense layers
"""
from typing import Dict
from typing import List
from typing import Union

import nucleus7 as nc7
import tensorflow as tf


class FullyConnectedPlugin(nc7.model.ModelPlugin):
    """
    Plugin with fully connected layers

    Parameters
    ----------
    layers_units
        list of units for dense layers; dense layers will be applied
        sequentially
    last_layer_without_activation
        if last layer should be without activation, e.g. to be used as logits
        in loss further

    Attributes
    ----------
    incoming_keys
        * features : input features

    generated_keys
        * features : result features after applying fully connected layers
    """
    incoming_keys = [
        "features",
    ]
    generated_keys = [
        "features",
    ]

    def __init__(self, *,
                 layers_units: Union[List[int], int],
                 last_layer_without_activation: bool = False,
                 **plugin_kwargs):
        super(FullyConnectedPlugin, self).__init__(**plugin_kwargs)
        if isinstance(layers_units, int):
            layers_units = [layers_units]
        self.layers_units = layers_units
        self.last_layer_without_activation = last_layer_without_activation
        self._dense_layers = []

    def create_keras_layers(self):
        super().create_keras_layers()
        for i_layer, each_layer_units in enumerate(self.layers_units):
            activation = self.activation
            is_last_layer = i_layer == len(self.layers_units) - 1
            if self.last_layer_without_activation and is_last_layer:
                activation = None
            dense_layer = self.add_keras_layer(
                tf.keras.layers.Dense(each_layer_units,
                                      activation=activation,
                                      kernel_initializer=self.initializer))
            self._dense_layers.append(dense_layer)

    def predict(self, features) -> Dict[str, tf.Tensor]:
        # pylint: disable=arguments-differ
        # parent save method has more generic signature
        x = features
        for i_layer, each_layer in enumerate(self._dense_layers):
            x = each_layer(x)
            if self.dropout and i_layer < len(self._dense_layers) - 1:
                x = self.dropout(x, training=self.is_training)
        result = {"features": x}
        return result
