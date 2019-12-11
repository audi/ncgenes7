# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

from absl.testing import parameterized
import tensorflow as tf

from ncgenes7.plugins.cnns.densenet import DensenetPlugin


class TestDensenetPlugin(parameterized.TestCase, tf.test.TestCase):

    def test_empty_graph_after_build(self):
        tf.reset_default_graph()
        plugin = DensenetPlugin(num_dense_blocks_per_layer=[2, 4]).build()
        self.assertListEqual([],
                             tf.get_default_graph().get_operations())

    @parameterized.parameters(
        {},
        {"multi_processes": True,
         "num_dense_blocks_per_layer": (2, 4),
         "kernel_size": [3, 5], "dilation_rate": [1, 3],
         "batch_normalization_params": {"momentum": 0.997}},
        {"multi_processes": True,
         "num_dense_blocks_per_layer": (3, 4, 5),
         "kernel_size": [3, 5, 7], "dilation_rate": [1, 3, 3],
         "batch_normalization_params": {"momentum": 0.997},
         "use_inception_module": True},
        {"kernel_size": 1, "dilation_rate": 1},
        {"kernel_size": 1, "bottleneck_factor": 2},
        {"num_dense_blocks_per_layer": (1, 2),
         "kernel_size": [1, 3, 5], "bottleneck_factor": -16,
         "use_inception_module": True},
    )
    def test_predict_tf(self,
                        num_dense_blocks_per_layer=(2, 3),
                        multi_processes=False,
                        kernel_size=3,
                        dilation_rate=1,
                        bottleneck_factor=4,
                        batch_normalization_params=None,
                        inputs_with_nan_dimensions=False,
                        use_inception_module=False):
        tf.reset_default_graph()
        if inputs_with_nan_dimensions:
            feature_maps = tf.placeholder(tf.float32, [None, None, None, 3])
        else:
            feature_maps = tf.placeholder(tf.float32, [None, 206, 289, 3])

        plugin = DensenetPlugin(
            num_dense_blocks_per_layer=num_dense_blocks_per_layer,
            kernel_size=kernel_size,
            dilation_rate=dilation_rate,
            bottleneck_factor=bottleneck_factor,
            batch_normalization_params=batch_normalization_params,
            use_inception_module=use_inception_module,
        ).build()

        result = plugin.predict(feature_maps=feature_maps)

        self.assertSetEqual(set(plugin.generated_keys_all),
                            set(result.keys()))

        if multi_processes:
            trainable_vars = tf.trainable_variables()
            result = plugin.predict(feature_maps=feature_maps)
            self.assertSetEqual(set(trainable_vars),
                                set(tf.trainable_variables()))
            plugin.reset_keras_layers()
            result = plugin.predict(feature_maps=feature_maps)
            self.assertEqual(len(trainable_vars) * 2,
                             len(tf.trainable_variables()))
