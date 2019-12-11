# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

"""
Tests for generic metrics
"""
import numpy as np
import tensorflow as tf
from absl.testing import parameterized

from ncgenes7.metrics.generic import AccuracyMetric
from ncgenes7.metrics.generic import PerClassAccuracyMetric


class TestAccuracyMetric(parameterized.TestCase, tf.test.TestCase):
    def test_interfaces(self):
        np.random.seed(4564)
        tf.reset_default_graph()

        data_in = tf.zeros([2, 3])
        data_in = data_in
        metric = AccuracyMetric(
            inbound_nodes=[])
        output = metric.process(data_in, data_in)

        self.assertTrue('scalar_accuracy' in output)
        self.assertTrue(isinstance(output['scalar_accuracy'], tf.Tensor))

    def test_computation(self):
        labels = tf.ones([100], dtype=tf.int32)
        ph_predictions = tf.placeholder(tf.int32)

        targets = np.zeros(shape=[100], dtype=np.int32)

        metric = AccuracyMetric(
            inbound_nodes=[])
        output = metric.process(labels=labels, predictions=ph_predictions)

        output_scalar = output['scalar_accuracy']

        initer_locals = tf.local_variables_initializer()

        with self.test_session() as sess:
            for ii in range(100):
                targets[ii] = 1
                sess.run(initer_locals)
                output_eval = sess.run(
                    output_scalar,
                    feed_dict={ph_predictions: targets})
                err = abs(float(output_eval) - float(ii + 1) / 100)
                self.assertTrue(err < 1e-5)


class TestPerClassAccuracyMetric(parameterized.TestCase,
                                 tf.test.TestCase):

    @parameterized.parameters({"labels": [[1], [0], [2], [2]],
                               "predictions": [1, 0, 2, 2],
                               "num_classes": 3},
                              {"labels": [[1], [0], [1], [0]],
                               "predictions": [0, 1, 0, 1],
                               "num_classes": 2}
                              )
    def test_process(self, labels, predictions, num_classes):
        tf.reset_default_graph()
        summary = PerClassAccuracyMetric(num_classes=num_classes,
                                         inbound_nodes=[]).build()
        labels = tf.constant(np.asarray(labels), tf.int32)
        predictions = tf.constant(np.asarray(predictions), tf.int32)
        result = summary.process(labels=labels, predictions=predictions)
        result_must = self._get_keys_based_on_classes(num_classes)

        self.assertSetEqual(set(result_must), set(result))
        initer_locals = tf.local_variables_initializer()
        self.evaluate(initer_locals)
        self.evaluate(result)

    def _get_keys_based_on_classes(self, num_classes):
        result_must = []
        for class_id in range(num_classes):
            result_must.append(
                'scalar_accuracy_for_class_class_' + str(class_id))
        result_must.append('scalar_mean_accuracy')
        return result_must
