# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

"""
Learning rate manipulators
"""
from absl.testing import parameterized
import tensorflow as tf

from ncgenes7.learning_rate_manipulators.with_linear_warmup import (
    TFLearningRateDecayWithWarmup)


class TestTFLearningRateDecayWithWarmup(
    parameterized.TestCase, tf.test.TestCase):

    def test_output_shape_and_type(self):
        tf.reset_default_graph()

        lr_manipulator = TFLearningRateDecayWithWarmup(
            100, 'exponential', decay_steps=1000, decay_rate=0.9)

        lr_in = 1.0
        step = tf.placeholder(tf.int64, [])
        lr_out = lr_manipulator.get_current_learning_rate(lr_in, step)

        self.assertListEqual(lr_out.get_shape().as_list(), [])

        self.assertEqual(lr_out.dtype, tf.float32)

    @parameterized.parameters([
        {'decay_name': 'piecewise_constant',
         'boundaries': [1000, 2000], 'values': [1.0, 0.5]},
        {'decay_name': 'polynomial', 'decay_steps': 1000}
    ])
    def test_all_types(self, decay_name, **decay_params):
        tf.reset_default_graph()

        lr_manipulator = TFLearningRateDecayWithWarmup(
            100, decay_name, **decay_params)

        lr_in = 1.0
        step = tf.placeholder(tf.int64, [])
        lr_out = lr_manipulator.get_current_learning_rate(lr_in, step)

        with self.test_session() as sess:
            lr_evaluated = sess.run(
                lr_out,
                feed_dict={step: 0})
            self.assertAllClose(lr_evaluated, 0.0)

            # Just check if it doesn't crash
            _ = sess.run(
                lr_out,
                feed_dict={step: 1000})
