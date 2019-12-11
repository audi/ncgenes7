# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

from unittest.mock import patch

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from ncgenes7.utils import object_detection_io_utils


class TestObjectDetectionIOUtils(parameterized.TestCase, tf.test.TestCase):

    def _get_json_fname(self, with_scores):
        if with_scores:
            fname = [{'id': 1, 'class_label': 2,
                      'bbox': {'xmin': 10, 'ymin': 20, 'w': 30, 'h': 40},
                      'score': 0.1},
                     {'id': 2, 'class_label': 3,
                      'bbox': {'xmin': 100, 'ymin': 200,
                               'xmax': 400, 'ymax': 600},
                      'score': 0.2}]
        else:
            fname = [{'id': 1, 'class_label': 2,
                      'bbox': {'xmin': 10, 'ymin': 20, 'w': 30, 'h': 40}},
                     {'id': 2, 'class_label': 3,
                      'bbox': {'xmin': 100, 'ymin': 200,
                               'xmax': 400, 'ymax': 600}}]
        return fname

    def get_data_must(self, with_scores=False):
        instance_ids_must = np.array([1, 2])
        class_labels_must = np.array([2, 3])
        bboxes_must = np.array([[20, 10, 60, 40],
                                [200, 100, 600, 400]]).astype(np.float32)
        if with_scores:
            scores = np.array([0.1, 0.2])
            return instance_ids_must, class_labels_must, bboxes_must, scores
        else:
            return instance_ids_must, class_labels_must, bboxes_must

    @parameterized.parameters({'ext': '.json', 'normalize_bbox': True},
                              {'ext': '.json', 'normalize_bbox': False})
    @patch("nucleus7.utils.io_utils.load_json", side_effect=lambda x: x)
    @patch("os.path.splitext",
           side_effect=lambda x: (None, '.json'))
    def test_read_labels(self, f, jf, ext, normalize_bbox):
        image_size = [50, 150]
        fname = self._get_json_fname(False)
        (class_labels, instance_ids, bboxes
         ) = object_detection_io_utils.read_objects(fname, image_size,
                                                    normalize_bbox)
        instance_ids_must, class_labels_must, bboxes_must = self.get_data_must()
        if normalize_bbox:
            bboxes_must[:, 0] = bboxes_must[:, 0] / image_size[0]
            bboxes_must[:, 1] = bboxes_must[:, 1] / image_size[1]
            bboxes_must[:, 2] = bboxes_must[:, 2] / image_size[0]
            bboxes_must[:, 3] = bboxes_must[:, 3] / image_size[1]
        self.assertTrue(np.allclose(instance_ids, instance_ids_must))
        self.assertTrue(np.allclose(class_labels, class_labels_must))
        self.assertTrue(np.allclose(bboxes, bboxes_must))

    @parameterized.parameters({'ext': '.json', 'normalize_bbox': True},
                              {'ext': '.json', 'normalize_bbox': False})
    @patch("nucleus7.utils.io_utils.load_json", side_effect=lambda x: x)
    @patch("os.path.splitext",
           side_effect=lambda x: (None, '.json'))
    def test_read_labels_with_scores(self, f, jf, ext, normalize_bbox):
        image_size = [50, 150]
        fname = self._get_json_fname(True)
        class_labels, instance_ids, bboxes, scores = (
            object_detection_io_utils.read_objects(
                fname, image_size, normalize_bbox, with_scores=True))
        (instance_ids_must, class_labels_must,
         bboxes_must, scores_must) = self.get_data_must(True)
        if normalize_bbox:
            bboxes_must[:, 0] = bboxes_must[:, 0] / image_size[0]
            bboxes_must[:, 1] = bboxes_must[:, 1] / image_size[1]
            bboxes_must[:, 2] = bboxes_must[:, 2] / image_size[0]
            bboxes_must[:, 3] = bboxes_must[:, 3] / image_size[1]
        self.assertTrue(np.allclose(instance_ids, instance_ids_must))
        self.assertTrue(np.allclose(class_labels, class_labels_must))
        self.assertTrue(np.allclose(bboxes, bboxes_must))
        self.assertTrue(np.allclose(scores, scores_must))
