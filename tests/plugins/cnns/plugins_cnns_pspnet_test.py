# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

from absl.testing import parameterized
import tensorflow as tf

from ncgenes7.plugins.cnns.pspnet import PSPNetPlugin


class TestPSPNetPlugin(parameterized.TestCase, tf.test.TestCase):

    @parameterized.parameters({"pool_sizes": (-1, -2, -3, -6)},
                              {"pool_sizes": (10, 20, 30, 40)})
    def test_empty_graph_after_build(self, pool_sizes):
        tf.reset_default_graph()
        plugin = PSPNetPlugin(filters=10,
                              pool_sizes=pool_sizes,
                              output_height=100,
                              output_width=150).build()
        self.assertListEqual([],
                             tf.get_default_graph().get_operations())

    @parameterized.parameters(
        {"filters": 16, "pool_sizes": (-1, -2, -3, -6),
         "output_height": 150, "output_width": 110,
         "shape_must": [None, 150, 110, 16]},
        {"multi_processes": True,
         "filters": 16, "pool_sizes": (-1, -2, -3, -6),
         "output_height": 150, "output_width": 110,
         "num_classes": 4,
         "shape_must": [None, 150, 110, 4]},
        {"multi_processes": True,
         "filters": 16, "pool_sizes": (10, 20, 30, 40),
         "output_height": 150, "output_width": 110,
         "num_classes": 4,
         "shape_must": [None, 150, 110, 4],
         "inputs_with_nan_shapes": True},
        {"filters": 16, "pool_sizes": (10, 20, 30, 40),
         "output_height": -4, "output_width": -2,
         "shape_must": [None, None, None, 16],
         "inputs_with_nan_shapes": True},
        {"filters": 16, "pool_sizes": (-1, -2, -3, -4),
         "output_height": -4, "output_width": -2,
         "shape_must": [None, 400, 300, 16],
         "inputs_with_nan_shapes": False},
    )
    def test_predict_tf(self,
                        shape_must,
                        filters=16,
                        pool_sizes=(-1, -2, -3, -6),
                        output_height=150,
                        output_width=110,
                        num_classes=None,
                        multi_processes=False,
                        inputs_with_nan_shapes=False,
                        ):
        tf.reset_default_graph()
        plugin = PSPNetPlugin(filters=filters,
                              pool_sizes=pool_sizes,
                              output_width=output_width,
                              output_height=output_height,
                              num_classes=num_classes).build()
        feature_maps = self._get_inputs(inputs_with_nan_shapes)

        result = plugin.predict(feature_maps=feature_maps)
        self.assertSetEqual(set(plugin.generated_keys_all),
                            set(result.keys()))
        self.assertSetEqual(set(shape_must),
                            set(result["feature_maps"].shape.as_list()))

        if multi_processes:
            trainable_vars = tf.trainable_variables()
            result = plugin.predict(feature_maps=feature_maps)
            self.assertSetEqual(set(trainable_vars),
                                set(tf.trainable_variables()))
            plugin.reset_keras_layers()
            result = plugin.predict(feature_maps=feature_maps)
            self.assertEqual(len(trainable_vars) * 2,
                             len(tf.trainable_variables()))

    def _get_inputs(self, inputs_with_nan_shapes):
        if inputs_with_nan_shapes:
            return tf.placeholder(tf.float32, [None, None, None, 16])
        return tf.placeholder(tf.float32, [None, 100, 150, 16])
