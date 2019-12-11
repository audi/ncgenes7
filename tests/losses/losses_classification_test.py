# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

import numpy as np
import tensorflow as tf
from absl.testing import parameterized

from ncgenes7.losses.classification import ClassificationFocalLoss


class TestClassificationFocalLoss(parameterized.TestCase, tf.test.TestCase):
    @parameterized.parameters({"predictions": [[1, 0], [0, 1]],
                               "labels": [[0], [1]],
                               "result_must": 0.0},
                              {"predictions": [[1, 0, 0]],
                               "labels": [[0]],
                               "result_must": 0.0},
                              {"predictions": [[0, 1], [1, 0], [0, 1]],
                               "labels": [[0], [1], [0]],
                               "result_must": 18.420681})
    def test_process(self, predictions, labels, result_must):
        predictions = tf.constant(np.asarray(predictions), tf.float32)
        labels = tf.constant(np.asarray(labels), tf.int32)
        loss = ClassificationFocalLoss()

        result = loss.process(classification_softmax=predictions,
                              groundtruth_classes=labels)
        self.assertSetEqual(set(loss.generated_keys), set(result.keys()))
        self.assertEmpty(result["loss"].shape)

        with self.test_session() as sess:
            result_value = sess.run(result)['loss']
            self.assertAllClose(result_must, result_value)
