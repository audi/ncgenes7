# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

from absl.testing import parameterized
import tensorflow as tf

from ncgenes7.plugins.cnns.aev_nips2017 import AEVNips2017Plugin


class TestAEVNips2017Plugin(parameterized.TestCase, tf.test.TestCase):

    def test_empty_graph_after_build(self):
        tf.reset_default_graph()
        plugin = AEVNips2017Plugin(filters=[10, 20, 30]).build()
        self.assertListEqual([],
                             tf.get_default_graph().get_operations())

    @parameterized.parameters(
        {"multi_processes": True,
         "filters": [10, 20],
         "inception_kernel_size": [1, 3],
         "dilated_inception_kernel_size": [1, 5],
         "dilated_inception_dilation_rate": [1, 3]},
        {"multi_processes": True,
         "sampling_type": "decoder",
         "filters": [10, 20],
         "inception_kernel_size": [1, 3],
         "dilated_inception_kernel_size": [1, 5],
         "dilated_inception_dilation_rate": None},
        {"multi_processes": True,
         "filters": [10, 20],
         "inception_kernel_size": 1,
         "dilated_inception_kernel_size": 1,
         "dilated_inception_dilation_rate": 3},
    )
    def test_predict_tf(self,
                        filters,
                        inception_kernel_size,
                        dilated_inception_kernel_size,
                        dilated_inception_dilation_rate,
                        sampling_type="encoder",
                        multi_processes=False,
                        inputs_with_nan_shapes=False,
                        ):
        tf.reset_default_graph()
        plugin = AEVNips2017Plugin(
            sampling_type=sampling_type,
            filters=filters,
            inception_kernel_size=inception_kernel_size,
            dilated_inception_kernel_size=dilated_inception_kernel_size,
            dilated_inception_dilation_rate=dilated_inception_dilation_rate
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

    def _get_inputs(self, inputs_with_nan_shapes):
        if inputs_with_nan_shapes:
            return tf.placeholder(tf.float32, [None, None, None, 16])
        return tf.placeholder(tf.float32, [None, 100, 150, 16])
