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

from ncgenes7.metrics.image import SemanticSegmentationMetrics


class TestSemanticSegmentationMetric(parameterized.TestCase, tf.test.TestCase):

    def test_mean_iou_for_each_class(self):
        tf.reset_default_graph()
        predictions_np = np.array([[0, 2, 1, 3, 4], [0, 2, 1, 3, 4]])
        labels_np = np.array([[0, 2, 1, 3, 4], [0, 2, 1, 3, 4]])
        num_classes = np.max(labels_np) + 1
        predictions = tf.constant(predictions_np, dtype=tf.int64)
        labels = tf.constant(labels_np, dtype=tf.int64)
        result_must = {}
        for c in range(num_classes):
            result_must.update({'scalar_iou_for_class_{:02d}'.format(c): 1.0})

        iou_process = SemanticSegmentationMetrics(
            inbound_nodes=[],
            num_classes=num_classes,
            resize_labels_to_predictions=False)
        iou_process.mode = 'train'
        result_iou = iou_process.process(
            prediction_segmentation_classes=predictions,
            groundtruth_segmentation_classes=labels)

        with self.test_session() as sess:
            tf.local_variables_initializer().run()
            result = sess.run(result_iou)
            for i in range(num_classes):
                self.assertEqual(
                    result['scalar_iou_for_class_{:02d}'.format(i)],
                    result_must['scalar_iou_for_class_{:02d}'.format(i)]
                )
