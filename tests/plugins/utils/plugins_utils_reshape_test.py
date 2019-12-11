# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

"""
Contains tests for reshape plugins
"""
from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from ncgenes7.plugins.utils.reshape import ConcatPlugin
from ncgenes7.plugins.utils.reshape import DataFormatConverter
from ncgenes7.plugins.utils.reshape import ExtractDimensionsFromBatch
from ncgenes7.plugins.utils.reshape import HideDimensionsInBatch


class TestHidingAndRestoring(tf.test.TestCase):

    def test_predict(self):
        np.random.seed(4564)
        tf.reset_default_graph()

        data = tf.random_uniform([2, 3, 4, 5])

        hider = HideDimensionsInBatch(inbound_nodes=[], target_rank=2)
        restorer = ExtractDimensionsFromBatch(inbound_nodes=[])

        out_hider = hider.predict(features=data)

        self.assertSetEqual({"features", "shape_in_batch"},
                            set(out_hider))
        out_restorer = restorer.predict(
            features=out_hider['features'],
            shape_in_batch=out_hider['shape_in_batch']
        )

        self.assertSetEqual({'features'},
                            set(out_restorer))


class TestDataFormatConversion(parameterized.TestCase, tf.test.TestCase):

    @parameterized.parameters(
        {'in_format': 'NTC', 'out_format': 'NHWC',
         'in_shape': [2, 3, 4], 'out_shape': [2, 1, 3, 4]},
        {'in_format': 'NTC', 'out_format': 'NHWC', 'mode': 'train',
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
    def test_predict(
            self, in_format, out_format,
            in_shape, out_shape, mode='eval'):
        tf.reset_default_graph()
        converter = DataFormatConverter(
            input_format=in_format, output_format=out_format)
        converter.mode = mode
        in_dat = tf.zeros(in_shape)
        out_dat = converter.predict(features=in_dat)['features']
        out_shape_eval = np.asarray(
            out_dat.get_shape().as_list(), dtype=np.int32)
        out_shape = np.asarray(out_shape, dtype=np.int32)
        self.assertAllEqual(out_shape, out_shape_eval)


class TestConcatPlugin(tf.test.TestCase, parameterized.TestCase):

    def test_predict_nested(self):
        inputs = {
            "key1": [np.random.rand(2, 3), np.random.rand(5, 3)],
            "key2": {"sub1": [np.random.rand(1), np.random.rand(2)],
                     "sub2": [np.random.rand(1, 5, 1), np.random.rand(1, 2, 1)]}
        }
        axis = {"default": 0,
                "key2//sub2": 1}
        plugin = ConcatPlugin(inbound_nodes=[], axis=axis).build()
        result = plugin.predict(**inputs)
        result_eval = self.evaluate(result)
        result_must = {
            "key1": np.concatenate(inputs["key1"], 0),
            "key2": {"sub1": np.concatenate(inputs["key2"]["sub1"], 0),
                     "sub2": np.concatenate(inputs["key2"]["sub2"], 1)},
        }
        self.assertAllClose(result_must,
                            result_eval)

    def test_predict(self):
        inputs = {"key1": [np.random.rand(2, 3), np.random.rand(5, 3)],
                  "key2": [np.random.rand(3, 2, 1), np.random.rand(1, 2, 1)]}
        axis = 0
        plugin = ConcatPlugin(inbound_nodes=[], axis=axis).build()
        result = plugin.predict(**inputs)
        result_eval = self.evaluate(result)
        result_must = {k: np.concatenate(v, axis)
                       for k, v in inputs.items()}
        self.assertAllClose(result_must,
                            result_eval)

    def test_raises_on_predict(self):
        inputs1 = {"key1": [np.random.rand(2, 3), np.random.rand(5, 3)],
                   "key2": [np.random.rand(3, 2, 1), np.random.rand(1, 2, 1)]}
        axis = {"key1": 0}
        plugin = ConcatPlugin(inbound_nodes=[], axis=axis).build()
        with self.assertRaises(ValueError):
            _ = plugin.predict(**inputs1)

        with self.assertRaises(ValueError):
            plugin.predict(key1=0)

        plugin = ConcatPlugin(inbound_nodes=[], axis=2).build()

        with self.assertRaises(ValueError):
            plugin.predict(key1=0)
