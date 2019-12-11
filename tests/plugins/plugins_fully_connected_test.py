# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

from absl.testing import parameterized
import tensorflow as tf

from ncgenes7.plugins.fully_connected import FullyConnectedPlugin


class TestFullyConnectedPlugin(parameterized.TestCase, tf.test.TestCase):

    def test_empty_graph_after_build(self):
        tf.reset_default_graph()
        plugin = FullyConnectedPlugin(layers_units=[10, 20]).build()
        self.assertListEqual([],
                             tf.get_default_graph().get_operations())

    @parameterized.parameters(

        {"multi_processes": True,
         "layers_units": 10, "last_layer_without_activation": True,
         "shape_must": [None, 10]},
        {"multi_processes": True,
         "layers_units": [10, 20], "last_layer_without_activation": False,
         "shape_must": [None, 20]},
        {"layers_units": [10, 5], "last_layer_without_activation": True,
         "shape_must": [None, 5]},
        {"layers_units": [10, 5], "last_layer_without_activation": True,
         "shape_must": [None, 5], "dropout": {"rate": 0.2}},
    )
    def test_predict_tf(self,
                        shape_must,
                        layers_units,
                        last_layer_without_activation,
                        dropout=None,
                        multi_processes=False):
        tf.reset_default_graph()
        plugin = FullyConnectedPlugin(
            layers_units=layers_units,
            last_layer_without_activation=last_layer_without_activation,
            dropout=dropout
        ).build()
        features = self._get_inputs()

        result = plugin.predict(features=features)
        self.assertSetEqual(set(plugin.generated_keys_all),
                            set(result.keys()))
        self.assertSetEqual(set(shape_must),
                            set(result["features"].shape.as_list()))

        if multi_processes:
            trainable_vars = tf.trainable_variables()
            result = plugin.predict(features=features)
            self.assertSetEqual(set(trainable_vars),
                                set(tf.trainable_variables()))
            plugin.reset_keras_layers()
            result = plugin.predict(features=features)
            self.assertEqual(len(trainable_vars) * 2,
                             len(tf.trainable_variables()))

    def _get_inputs(self):
        return tf.placeholder(tf.float32, [None, 16])
