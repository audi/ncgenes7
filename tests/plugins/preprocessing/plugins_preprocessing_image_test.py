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

from ncgenes7.plugins.preprocessing.image import ImageLocalMeanSubtraction
from ncgenes7.plugins.preprocessing.image import ImageStandardization


class TestImageStandardization(parameterized.TestCase, tf.test.TestCase):

    @parameterized.parameters({"num_channels": 3},
                              {"num_channels": 1})
    def test_predict(self, num_channels):
        inputs_np = np.random.randn(3, 10, 20, num_channels).astype(np.float32)
        inputs_tf = tf.constant(inputs_np)

        preprocessor = ImageStandardization().build()
        result = preprocessor.predict(images=inputs_tf)
        result_eval = self.evaluate(result)
        self.assertSetEqual(set(preprocessor.generated_keys_all),
                            set(result))
        images_must = self.evaluate(
            [tf.image.per_image_standardization(each_image)
             for each_image in inputs_np])
        self.assertAllClose(images_must,
                            result_eval["images"])


class TestImageLocalMeanSubtraction(parameterized.TestCase, tf.test.TestCase):

    @parameterized.parameters(
        [{'shape_in': [10, 11, 3, 1], 'kernel_size': 3},
         {'shape_in': [2, 15, 12, 3], 'kernel_size': [7, 3]}])
    def test_predict(self, shape_in, kernel_size):
        np.random.seed(4564)
        tf.reset_default_graph()
        images_np = np.random.rand(*shape_in) + 2.0
        images_tf = tf.constant(images_np, tf.float32)
        preprocessor = ImageLocalMeanSubtraction(
            kernel_size=kernel_size).build()
        result = preprocessor.predict(images=images_tf)
        self.assertSetEqual(set(preprocessor.generated_keys_all),
                            set(result))
        if isinstance(kernel_size, int):
            ksize = [1, kernel_size, kernel_size, 1]
        else:
            ksize = [1, kernel_size[0], kernel_size[1], 1]
        images_local_mean = self.evaluate(
            tf.nn.avg_pool(images_tf, ksize, [1] * 4, "SAME"))
        result_must = images_np - images_local_mean
        self.assertAllClose(result_must,
                            result["images"])
