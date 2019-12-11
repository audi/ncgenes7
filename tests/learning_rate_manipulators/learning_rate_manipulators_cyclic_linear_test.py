# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

"""
Tests for cyclic learning rate manipulators
"""
from absl.testing import parameterized
import tensorflow as tf

from ncgenes7.learning_rate_manipulators.cyclic_linear import (
    CyclicLinearLearningRate)


class TestCyclicLinearLearningRate(parameterized.TestCase, tf.test.TestCase):

    def setUp(self):
        tf.reset_default_graph()

        lr_manipulator = CyclicLinearLearningRate(200)

        lr_in = 1.0
        self.step = step = tf.placeholder(tf.int64, [])
        self.lr_out = lr_manipulator.get_current_learning_rate(lr_in, step)

    def test_output_shape_and_type(self):
        self.assertListEqual(self.lr_out.get_shape().as_list(), [])
        self.assertEqual(self.lr_out.dtype, tf.float32)

    def test_computation(self):
        input_to_output = {
            0: 0.0,
            100: 1.0,
            200: 0.1,
            300: 1.0,
            400: 0.1,
            500: 1.0
        }

        with self.test_session() as sess:
            for current_step, target_output in input_to_output.items():
                lr_evaluated = sess.run(
                    self.lr_out,
                    feed_dict={self.step: current_step})
                self.assertAllClose(lr_evaluated, target_output)
