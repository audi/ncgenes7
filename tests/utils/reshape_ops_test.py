# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

"""
Tests for reshape ops
"""

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from ncgenes7.utils.reshape_ops import expand_batch_dimension
from ncgenes7.utils.reshape_ops import get_conversion_function
from ncgenes7.utils.reshape_ops import maybe_reshape
from ncgenes7.utils.reshape_ops import maybe_restore
from ncgenes7.utils.reshape_ops import squeeze_batch_dimension


class TestSqueezeExpandDims(parameterized.TestCase, tf.test.TestCase):

    def test_squeeze_batch_dimension(self):
        class _Dummy(object):
            @squeeze_batch_dimension("input1")
            def process(self, input1, input2):
                return {
                    "input1": input1,
                    "input2": input2
                }

        dummy_object = _Dummy()
        output = dummy_object.process(input1=tf.constant([[1.0]]),
                                      input2=None)
        output_shape = np.asarray(output['input1'].get_shape().as_list(),
                                  dtype=np.int32)
        self.assertAllEqual([1], output_shape)

    def test_expand_batch_dimension(self):
        class _Dummy(object):
            @expand_batch_dimension("input1")
            def process(self, input1, input2):
                return {
                    "input1": input1,
                    "input2": input2
                }

        dummy_object = _Dummy()
        output = dummy_object.process(input1=tf.constant([1.0]),
                                      input2=None)
        output_shape = np.asarray(output['input1'].get_shape().as_list(),
                                  dtype=np.int32)
        self.assertAllEqual([1, 1], output_shape)

    def test_both(self):
        class _Dummy(object):
            @squeeze_batch_dimension("input1")
            @expand_batch_dimension("input1")
            def process(self, input1, input2):
                return {
                    "input1": input1,
                    "input2": input2
                }

        dummy_object = _Dummy()
        output = dummy_object.process(input1=tf.constant([[1.0]]),
                                      input2=None)
        self.assertAllEqual(tf.constant([[1.0]]), output['input1'])

    @parameterized.parameters(
        {"shape_in": [2, 3, 4], 'wanted_rank': 2},
        {"shape_in": [2, 3], 'wanted_rank': 2},
        {"shape_in": [2, 3, 4, 5, 6], 'wanted_rank': 2},
        {"shape_in": [2, 3, 4], 'wanted_rank': 1},
        {"shape_in": [2, 3], 'wanted_rank': 1},
        {"shape_in": [2, 3, 4, 5, 6], 'wanted_rank': 1},
        {"shape_in": [2, 3, 4], 'wanted_rank': 3},
        {"shape_in": [2, 3, 4, 5, 6], 'wanted_rank': 3})
    def test_helpers(self, shape_in, wanted_rank):
        np.random.seed(4564)
        tf.reset_default_graph()

        data = tf.random_uniform(shape_in)

        data_target_rank, shape_in_batch = maybe_reshape(data, wanted_rank)

        actual_rank = tf.rank(data_target_rank)

        data_res = maybe_restore(data_target_rank, shape_in_batch)

        for _ in range(2):
            d_in, rank_inter, d_out, d_target = self.evaluate(
                [data, actual_rank, data_res, data_target_rank])

            self.assertEqual(
                wanted_rank,
                rank_inter,
                'Rank conversion failed: Expected rank {}, given rank {}, '
                'numpy rank {}'.format(
                    wanted_rank, rank_inter, len(d_target.shape)
                ))

            self.assertAllClose(d_in, d_out,
                                msg='Reverting does not work as expected')
            self.assertListEqual(
                list(d_in.shape),
                list(d_out.shape),
                msg='Shape does not match'
            )

    @parameterized.parameters(
        {'in_format': 'NTC', 'out_format': 'NHWC',
         'in_shape': [2, 3, 4], 'out_shape': [2, 1, 3, 4]},
        {'in_format': 'NTC', 'out_format': 'NHWC',
         'in_shape': [1, 1, 1], 'out_shape': [1, 1, 1, 1]},
        {'in_format': 'NHWC', 'out_format': 'NCHW',
         'in_shape': [1, 2, 3, 4], 'out_shape': [1, 4, 2, 3]},
        {'in_format': 'NHWC', 'out_format': 'NCHW',
         'in_shape': [4, 3, 2, 1], 'out_shape': [4, 1, 3, 2]},
        {'in_format': 'NCHW', 'out_format': 'NHWC',
         'in_shape': [1, 2, 3, 4], 'out_shape': [1, 3, 4, 2]},
        {'in_format': 'NCHW', 'out_format': 'NHWC',
         'in_shape': [4, 3, 2, 1], 'out_shape': [4, 2, 1, 3]}
    )
    def test_converters_single(
            self, in_format, out_format, in_shape, out_shape):
        tf.reset_default_graph()
        in_dat = tf.zeros(in_shape)
        converter = get_conversion_function(in_format, out_format)
        out_dat = converter(in_dat)
        out_shape_eval = np.asarray(
            out_dat.get_shape().as_list(), dtype=np.int32)
        out_shape = np.asarray(out_shape, dtype=np.int32)
        self.assertAllEqual(out_shape, out_shape_eval)

    @parameterized.parameters([
        {'format_1': 'NCHW', 'format_2': 'NHWC',
         'shape_1': [1, 2, 3, 4], 'shape_2': [1, 2, 3, 4]},
        {'format_1': 'NCHW', 'format_2': 'NHWC',
         'shape_1': [4, 2, 4, 1], 'shape_2': [9, 2, 1, 3]}])
    def test_circle_consistency(
            self, format_1, format_2, shape_1, shape_2, n_runs=3):
        tf.reset_default_graph()
        ph_1 = tf.placeholder(tf.float32, shape_1)
        ph_2 = tf.placeholder(tf.float32, shape_2)

        f1_2_f2 = get_conversion_function(format_1, format_2)
        f2_2_f1 = get_conversion_function(format_2, format_1)

        input_1_restored = f2_2_f1(f1_2_f2(ph_1))
        input_2_restored = f1_2_f2(f2_2_f1(ph_2))

        fail_msg = ('Cycle consistency not given for data formats: '
                    '{}, {}'.format(format_1, format_2))

        with self.test_session() as sess:
            for _ in range(n_runs):
                input_1 = np.random.uniform(size=shape_1).astype(np.float32)
                input_2 = np.random.uniform(size=shape_2).astype(np.float32)
                input_1_calculated, input_2_calculated = sess.run(
                    [input_1_restored, input_2_restored],
                    feed_dict={ph_1: input_1, ph_2: input_2})
                self.assertAllEqual(input_1, input_1_calculated, msg=fail_msg)
                self.assertAllEqual(input_2, input_2_calculated, msg=fail_msg)
