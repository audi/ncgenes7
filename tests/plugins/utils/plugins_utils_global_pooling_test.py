# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

"""
Contains tests for global pooling plugin
"""
from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from ncgenes7.plugins.utils.global_pooling import GlobalAveragePooling


class TestGlobalAveragePooling(parameterized.TestCase, tf.test.TestCase):

    @parameterized.parameters(
        {'in_shape': [2, 3, 4, 5],
         "pool_dimensions": [-1, -2], 'keepdims': False},
        {'in_shape': [2, 3, 4, 5],
         "pool_dimensions": [-1, -2], 'keepdims': True})
    def test_predict(
            self, in_shape, keepdims, pool_dimensions=None):
        np.random.seed(4564 + np.prod(in_shape))
        tf.reset_default_graph()

        inputs_np = np.random.rand(*in_shape)
        inputs_tf = tf.constant(inputs_np, tf.float32)

        pooler = GlobalAveragePooling(
            inbound_nodes=[],
            keepdims=keepdims,
            pool_dimensions=pool_dimensions)
        result = pooler.predict(features=inputs_tf)
        result_eval = self.evaluate(result)
        result_must = {
            "features": inputs_np.mean(axis=tuple(pool_dimensions),
                                       keepdims=keepdims)}

        self.assertAllClose(result_must,
                            result_eval)
