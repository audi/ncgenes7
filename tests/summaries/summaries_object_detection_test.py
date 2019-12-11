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

from ncgenes7.summaries.object_detection import ObjectDrawerSummary


class TestObjectDrawerSummary(parameterized.TestCase, tf.test.TestCase):

    def setUp(self):
        np.random.seed(65475)
        self.num_classes = 5
        self.batch_size = 2
        self.images = np.random.rand(self.batch_size, 30, 20, 3)
        self.object_boxes = np.array([[[0.1, 0.1, 0.3, 0.4],
                                       [0., 0., 0., 0.]],
                                      [[0.2, 0.3, 0.4, 0.5],
                                       [0.5, 0.6, 0.7, 0.7]]])
        self.object_classes = np.random.randint(
            1, self.num_classes, [self.batch_size, 2])
        self.object_scores = np.random.rand(self.batch_size, 2)
        self.object_instance_ids = np.array([[1, 10], [5, 4]])
        self.keypoints = np.array([
            [[[0.5, 0.5],
              [0.5, 1.0],
              [0, 0],
              [0, 0],
              [0, 0.25]],
             [[0, 0.8],
              [0, 0],
              [0.1, 0.3],
              [1, 1],
              [0, 0]]],
            [[[8 / 9, 4 / 5],
              [4 / 9, 1 / 5],
              [0, 0],
              [4 / 9, 4 / 5],
              [0, 1]],
             [[0, 0],
              [0, 0],
              [0, 0],
              [0, 0],
              [0, 0]]]
        ], np.float32)
        self.instance_masks = np.zeros([self.batch_size, 2, 30, 20], np.uint8)
        self.instance_masks[0, 0, 5:10, 2:5] = 1
        self.instance_masks[0, 1, 1:5, 1:10] = 1
        self.instance_masks[1, 0, 20:30, 10:15] = 1

    @parameterized.parameters(
        {"provide_ids": True, "provide_masks": True, "provide_keypoints": True},
        {"provide_ids": False, "provide_masks": False,
         "provide_keypoints": False},
        {"provide_ids": True, "provide_masks": False,
         "provide_keypoints": False},
        {"provide_ids": False, "provide_masks": True,
         "provide_keypoints": True},
    )
    def test_process(self, provide_ids, provide_masks, provide_keypoints):
        summary = ObjectDrawerSummary(num_classes=5,
                                      inbound_nodes=[]).build()
        object_instance_ids = self.object_instance_ids if provide_ids else None
        keypoints = self.keypoints if provide_keypoints else None
        instance_masks = self.instance_masks if provide_masks else None

        result = summary.process(
            images=self.images,
            object_boxes=self.object_boxes,
            object_classes=self.object_classes,
            object_scores=self.object_scores,
            object_instance_ids=object_instance_ids,
            object_keypoints=keypoints,
            object_instance_masks_on_image=instance_masks)

        self.assertSetEqual(set(summary.generated_keys),
                            set(result))
        self.evaluate(result)
