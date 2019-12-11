# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

import numpy as np
import tensorflow as tf

from ncgenes7.plugins.object_detection.general import \
    DetectionsClassSelectorPlugin


class TestDetectionsClassSelectorPlugin(tf.test.TestCase):
    def setUp(self):
        self.classes_to_select = [1, 5, 7]

        self.object_classes_np = np.array([
            [2, 1, 2],
            [5, 3, 1],
            [3, 2, 0],
        ], np.int32)
        self.object_boxes_np = np.array([
            [[1, 2, 3, 4],
             [5, 6, 7, 8],
             [9, 10, 11, 12]],
            [[13, 14, 15, 16],
             [17, 18, 19, 20],
             [21, 22, 23, 24]],
            [[25, 26, 27, 28],
             [29, 30, 31, 32],
             [0, 0, 0, 0]],
        ], np.float32)
        self.instance_ids_np = np.array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 0],
        ], np.int32)
        self.num_objects_np = np.array([3, 3, 2], np.int32)

        self.object_boxes = tf.placeholder(tf.float32, [None, None, 4])
        self.object_classes = tf.placeholder(tf.int32, [None, None])
        self.instance_ids = tf.placeholder(tf.int32, [None, None])
        self.num_objects = tf.placeholder(tf.int32, [None])

        self.result_must = {
            "detection_object_classes": np.array([
                [1, 0, 0],
                [5, 1, 0],
                [0, 0, 0],
            ], np.int32),
            "detection_object_boxes": np.array([
                [[5, 6, 7, 8],
                 [0, 0, 0, 0],
                 [0, 0, 0, 0]],
                [[13, 14, 15, 16],
                 [21, 22, 23, 24],
                 [0, 0, 0, 0]],
                [[0, 0, 0, 0],
                 [0, 0, 0, 0],
                 [0, 0, 0, 0]],
            ], np.float32),
            "detection_instance_ids": np.array([
                [2, 0, 0],
                [4, 6, 0],
                [0, 0, 0],
            ], np.int32),
            "num_object_detections": np.array([1, 2, 0]),
        }

    def test_predict(self):
        plugin = DetectionsClassSelectorPlugin(
            classes_to_select=self.classes_to_select,
        ).build()
        result = plugin.predict(
            detection_object_classes=self.object_classes,
            detection_object_boxes=self.object_boxes,
            detection_instance_ids=self.instance_ids,
            num_object_detections=self.num_objects,
        )
        with self.test_session() as sess:
            feed_dict = {
                self.object_classes: self.object_classes_np,
                self.object_boxes: self.object_boxes_np,
                self.instance_ids: self.instance_ids_np,
                self.num_objects: self.num_objects_np,
            }
            result_eval = sess.run(result, feed_dict=feed_dict)

        self.assertAllClose(self.result_must,
                            result_eval)
