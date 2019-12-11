# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from ncgenes7.plugins.cnns.duc import DUCPlugin


class TestDUCPlugin(parameterized.TestCase, tf.test.TestCase):

    def test_empty_graph_after_build(self):
        tf.reset_default_graph()
        plugin = DUCPlugin(kernel_size=3,
                           num_classes=10,
                           stride_upsample=2).build()
        self.assertListEqual([],
                             tf.get_default_graph().get_operations())

    @parameterized.parameters(

        {"multi_processes": True,
         "dilation_rates": (2, 4, 8, 16), "stride_upsample": 2,
         "num_classes": 8,
         "shape_must": [3, 200, 300, 8]},
        {"dilation_rates": (2, 6), "stride_upsample": 4,
         "num_classes": 2,
         "shape_must": [3, 400, 600, 2]},
        {"dilation_rates": (2, 6), "stride_upsample": 4,
         "num_classes": 2,
         "shape_must": [3, 400, 600, 2],
         "inputs_with_nan_shapes": True},
    )
    def test_predict_tf(self,
                        shape_must,
                        dilation_rates,
                        stride_upsample,
                        multi_processes=False,
                        inputs_with_nan_shapes=False,
                        num_classes=4,
                        image_height=None,
                        image_width=None,
                        ):
        tf.reset_default_graph()
        plugin = DUCPlugin(kernel_size=3,
                           num_classes=num_classes,
                           stride_upsample=stride_upsample,
                           dilation_rates=dilation_rates,
                           image_height=image_height,
                           image_width=image_width,
                           ).build()
        feature_maps = self._get_inputs(inputs_with_nan_shapes)

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

        feature_maps_np = np.random.rand(3, 100, 150, 16)
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            result_eval = sess.run(result,
                                   feed_dict={feature_maps: feature_maps_np})
        self.assertSetEqual(set(shape_must),
                            set(result_eval["feature_maps"].shape))

    def _get_inputs(self, inputs_with_nan_shapes):
        if inputs_with_nan_shapes:
            return tf.placeholder(tf.float32, [None, None, None, 16])
        return tf.placeholder(tf.float32, [None, 100, 150, 16])
