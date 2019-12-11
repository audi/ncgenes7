# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

import json
import os
from unittest.mock import patch

from absl.testing import parameterized
import matplotlib

matplotlib.use('Agg')

import numpy as np
import tensorflow as tf

from ncgenes7.callbacks.object_detection import ConverterToImageFrameCallback
from ncgenes7.callbacks.object_detection import NonMaxSuppressionCallback
from ncgenes7.callbacks.object_detection import NormalizeCoordinatesCallback
from ncgenes7.callbacks.object_detection import ObjectDrawerCallback
from ncgenes7.callbacks.object_detection import ObjectsSaver


class TestObjectDrawerCallback(parameterized.TestCase,
                               tf.test.TestCase):

    def setUp(self):
        np.random.seed(65475)
        self.num_classes = 5
        self.batch_size = 2
        self.images = np.random.rand(self.batch_size, 30, 20, 3)
        self.object_boxes = np.array([[[0.1, 0.1, 0.3, 0.4],
                                       [0., 0., 0., 0.]],
                                      [[0.2, 0.3, 0.4, 0.5],
                                       [0.5, 0.6, 0.7, 0.7]]])
        self.keypoints = np.random.rand(self.batch_size, 2, 5, 2)
        self.instance_masks = np.random.rand(self.batch_size, 2, 30, 20) * 10
        self.object_classes = np.random.randint(
            1, self.num_classes, [self.batch_size, 2])
        self.object_scores = np.random.rand(self.batch_size, 2)
        self.object_instance_ids = np.array([[1, 10], [5, 4]])

    @parameterized.parameters(
        {"provide_ids": True, "provide_keypoints": True,
         "provide_instance_masks": True},
        {"provide_ids": False, "provide_keypoints": False,
         "provide_instance_masks": False},
        {"provide_ids": False, "provide_keypoints": True,
         "provide_instance_masks": True}
    )
    @patch("object_detection.utils.visualization_utils."
           "visualize_boxes_and_labels_on_image_array")
    def test_on_iteration_end(self, visualize_fn, provide_ids,
                              provide_keypoints, provide_instance_masks):
        resulted_image = np.random.randint(
            0, 255, size=(10, 10, 3)).astype(np.uint8)
        visualize_fn.return_value = resulted_image
        callback = ObjectDrawerCallback(num_classes=self.num_classes,
                                        max_boxes_to_draw=100,
                                        line_thickness=3,
                                        score_threshold=0.1,
                                        inbound_nodes=[]).build()
        category_index_must = {i: {'id': i, 'name': 'class_%d' % i}
                               for i in range(1, self.num_classes + 1)}
        self.assertDictEqual(category_index_must,
                             callback.category_index)
        object_instance_ids = self.object_instance_ids if provide_ids else None
        keypoints = self.keypoints if provide_keypoints else None
        instance_masks = self.instance_masks if provide_instance_masks else None
        result = callback.on_iteration_end(
            images=self.images, object_boxes=self.object_boxes,
            object_classes=self.object_classes,
            object_scores=self.object_scores,
            object_instance_ids=object_instance_ids,
            object_keypoints=keypoints,
            object_instance_masks_on_image=instance_masks,
        )

        for sample_index in range(self.batch_size):
            call = visualize_fn.call_args_list[sample_index]
            *_, call_kwargs = call
            image_uint = (self.images[sample_index] * 255.0).astype(np.uint)
            self.assertAllClose(image_uint,
                                call_kwargs["image"])
            self.assertAllClose(self.object_boxes[sample_index],
                                call_kwargs["boxes"])
            self.assertAllClose(self.object_classes[sample_index],
                                call_kwargs["classes"])
            self.assertAllClose(self.object_scores[sample_index],
                                call_kwargs["scores"])
            if provide_ids:
                self.assertAllClose(self.object_instance_ids[sample_index],
                                    call_kwargs["track_ids"])
            if provide_keypoints:
                self.assertAllClose(self.keypoints[sample_index],
                                    call_kwargs["keypoints"])
            if provide_instance_masks:
                self.assertAllClose(
                    self.instance_masks[sample_index].astype(np.uint8),
                    call_kwargs["instance_masks"])

        self.assertEqual(self.batch_size,
                         visualize_fn.call_count)
        resulted_image_float = resulted_image.astype(np.float32) / 255.0
        result_must = {
            "images_with_objects": np.stack(
                [resulted_image_float, resulted_image_float], 0)}
        self.assertAllClose(result_must,
                            result)

    @parameterized.parameters({"provide_ids": True},
                              {"provide_ids": False})
    def test_on_iteration_end_run(self, provide_ids):
        callback = ObjectDrawerCallback(num_classes=self.num_classes,
                                        max_boxes_to_draw=100,
                                        line_thickness=3,
                                        score_threshold=0.1,
                                        inbound_nodes=[]).build()
        object_instance_ids = self.object_instance_ids if provide_ids else None
        result = callback.on_iteration_end(
            images=self.images, object_boxes=self.object_boxes,
            object_classes=self.object_classes,
            object_scores=self.object_scores,
            object_instance_ids=object_instance_ids)
        self.assertSetEqual({"images_with_objects"},
                            set(result))
        images_with_objects = result["images_with_objects"]
        self.assertAllEqual(self.images.shape,
                            images_with_objects.shape)
        self.assertDTypeEqual(images_with_objects,
                              np.float32)


class TestNonMaxSuppressionCallback(parameterized.TestCase, tf.test.TestCase):

    def setUp(self):
        np.random.seed(65475)
        self.num_classes = 5
        self.batch_size = 2
        self.object_boxes = np.array([[[0.1, 0.1, 0.3, 0.4],
                                       [0., 0., 0., 0.],
                                       [0., 0., 0., 0.]],
                                      [[0.2, 0.3, 0.4, 0.5],
                                       [0., 0., 0., 0.],
                                       [0.5, 0.6, 0.7, 0.7]]])
        self.object_classes = np.array([[1, 0, 1],
                                        [2, 0, 5]])
        self.object_scores = np.array([[0.5, 0, 0],
                                       [0.5, 0, 0.9]])
        self.object_instance_ids = np.array([[10, 0, 2],
                                             [3, 0, 1]])

    @parameterized.parameters({"with_instance_ids": True},
                              {"with_instance_ids": False})
    def test_on_iteration_end_run(self, with_instance_ids):
        callback = NonMaxSuppressionCallback(num_classes=self.num_classes,
                                             nms_iou_threshold=0.5,
                                             score_threshold=0.1,
                                             inbound_nodes=[]).build()
        instance_ids = self.object_instance_ids if with_instance_ids else None
        result = callback.on_iteration_end(
            object_boxes=self.object_boxes,
            object_classes=self.object_classes,
            object_scores=self.object_scores,
            object_instance_ids=instance_ids)
        self.assertSetEqual(set(NonMaxSuppressionCallback.generated_keys_all),
                            set(result))

        result_must = {
            "object_boxes": np.array([[[0.1, 0.1, 0.3, 0.4],
                                       [0., 0., 0., 0.]],
                                      [[0.5, 0.6, 0.7, 0.7],
                                       [0.2, 0.3, 0.4, 0.5]]]),
            "object_classes": np.array([[1, 0],
                                        [5, 2]]),
            "object_scores": np.array([[0.5, 0],
                                       [0.9, 0.5]]),
            "num_objects": np.array([1, 2])
        }
        if with_instance_ids:
            result_must["object_instance_ids"] = np.array([[10, -1],
                                                           [1, 3]])
        else:
            result_must["object_instance_ids"] = np.zeros([2, 2]) - 1

        self.assertAllClose(result_must,
                            result)


class TestObjectSaver(parameterized.TestCase, tf.test.TestCase):

    def setUp(self):
        np.random.seed(65475)
        self.batch_size = 2
        self.images = np.random.rand(self.batch_size, 30, 20, 3)
        self.object_boxes = np.array([[[0.1, 0.1, 0.3, 0.4],
                                       [0., 0., 0., 0.],
                                       [0., 0., 0., 0.]],
                                      [[0.2, 0.3, 0.4, 0.5],
                                       [0., 0., 0., 0.],
                                       [0.5, 0.6, 0.7, 0.7]]])
        self.object_classes = np.array([[1, 0, 1],
                                        [2, 0, 5]])
        self.object_scores = np.array([[0.5, 0, 0],
                                       [0.5, 0, 0.9]])
        self.object_instance_ids = np.array([[10, 0, 2],
                                             [3, 0, 1]])
        self.object_keypoints = np.array([
            [[[0, 1], [2, 3], [4, 5]],
             [[6, 7], [8, 9], [10, 11]],
             [[12, 13], [0, 0], [0, 0]]],
            [[[10, 20], [30, 40], [50, 60]],
             [[70, 80], [0, 0], [0, 0]],
             [[0, 0], [0, 10], [50, 90]]]
        ])
        self.object_keypoints_scores = np.array([
            [[0.1, 0.2, 0.3],
             [0.4, 0, 0],
             [0.5, 0.1, 0]],
            [[10, 20, 30],
             [40, 0, 50],
             [70, 60, 0]]
        ])
        self.object_keypoints_visibilities = np.array([
            [[0, 1, 1],
             [2, 2, 2],
             [1, 1, 0]],
            [[2, 2, 0],
             [0, 2, 1],
             [0, 1, 2]]
        ])

    @parameterized.parameters(
        {"with_instance_ids": True, "with_scores": True},
        {"with_instance_ids": False, "with_scores": False},
        {"with_instance_ids": True, "with_scores": True,
         "with_keypoints": True, "with_keypoints_scores": True,
         "with_keypoints_visibilities": True},
        {"with_instance_ids": False, "with_scores": False,
         "with_keypoints": True, "with_keypoints_scores": False,
         "with_keypoints_visibilities": False},
    )
    def test_on_iteration_end(self, with_instance_ids, with_scores,
                              with_keypoints=False, with_keypoints_scores=False,
                              with_keypoints_visibilities=False):
        temp_dir = self.get_temp_dir()
        callback = ObjectsSaver(inbound_nodes=[]).build()
        callback.log_dir = temp_dir

        images_fnames = np.array(
            ["fname_%d" % i for i in range(self.batch_size)])
        inputs = {
            "object_boxes": self.object_boxes,
            "object_classes": self.object_classes,
            "save_names": images_fnames,
        }
        if with_instance_ids:
            inputs["object_instance_ids"] = self.object_instance_ids
        if with_scores:
            inputs["object_scores"] = self.object_scores
        if with_keypoints:
            inputs["keypoints"] = self.object_keypoints
        if with_keypoints_scores:
            inputs["keypoints_scores"] = self.object_keypoints_scores
        if with_keypoints_visibilities:
            inputs["keypoints_visibilities"] = (
                self.object_keypoints_visibilities)

        callback.on_iteration_end(**inputs)
        base_names_must = [each_fname + ".json" for each_fname in images_fnames]
        self.assertSetEqual(set(base_names_must),
                            set(os.listdir(temp_dir)))
        for sample_i, each_fname in enumerate(base_names_must):
            with open(os.path.join(temp_dir, each_fname), "r") as f:
                objects_loaded = json.load(f)
            bbox_sample = self.object_boxes[sample_i]
            objects_loaded_must = [
                {
                    "bbox": {
                        "h": bbox_sample[i][2] - bbox_sample[i][0],
                        "w": bbox_sample[i][3] - bbox_sample[i][1],
                        "ymin": bbox_sample[i][0],
                        "xmin": bbox_sample[i][1],
                        "ymax": bbox_sample[i][2],
                        "xmax": bbox_sample[i][3],
                    },
                    "class_label": self.object_classes[sample_i][i],
                    "id": (self.object_instance_ids[sample_i][i]
                           if with_instance_ids else -1),
                    "score": (self.object_scores[sample_i][i]
                              if with_scores else 1),
                    "keypoints": self.object_keypoints[sample_i][i].tolist(),
                    "keypoints_scores":
                        self.object_keypoints_scores[sample_i][i].tolist(),
                    "keypoints_visibilities":
                        self.object_keypoints_visibilities[sample_i][i].tolist()
                }
                for i in range(len(bbox_sample))
                if bbox_sample[i][2] - bbox_sample[i][0] > 0
            ]
            if not with_keypoints:
                for i, each_obj in enumerate(objects_loaded_must):
                    each_obj.pop("keypoints")
            if not with_keypoints_scores:
                for i, each_obj in enumerate(objects_loaded_must):
                    each_obj.pop("keypoints_scores")
            if not with_keypoints_visibilities:
                for i, each_obj in enumerate(objects_loaded_must):
                    each_obj.pop("keypoints_visibilities")

            self.assertAllClose(objects_loaded_must,
                                objects_loaded)


class ConverterToImageFrameCallbackTest(tf.test.TestCase):

    def setUp(self):
        self.images = np.random.rand(3, 25, 73, 3)
        object_boxes_min = np.random.rand(3, 10, 2)
        object_boxes_height_width = np.random.rand(3, 10, 2)
        self.object_boxes_normalized = np.concatenate(
            [object_boxes_min, object_boxes_min + object_boxes_height_width],
            -1)
        self.object_boxes_image_coordinates = (
                self.object_boxes_normalized
                * np.reshape([25, 73, 25, 73], [1, 1, 4]))

    def test_on_iteration_end(self):
        callback = ConverterToImageFrameCallback(
            inbound_nodes=[]).build()
        inputs = {"images": self.images,
                  "object_boxes": self.object_boxes_normalized}
        outputs = callback.on_iteration_end(**inputs)
        self.assertSetEqual(set(callback.generated_keys),
                            set(outputs))

        outputs_must = {
            "object_boxes": self.object_boxes_image_coordinates
        }
        self.assertAllClose(outputs_must,
                            outputs)


class NormalizeCoordinatesCallbackTest(tf.test.TestCase):

    def setUp(self):
        self.images = np.random.rand(3, 25, 73, 3)
        object_boxes_min = np.random.rand(3, 10, 2)
        object_boxes_height_width = np.random.rand(3, 10, 2)
        self.object_boxes_normalized = np.concatenate(
            [object_boxes_min, object_boxes_min + object_boxes_height_width],
            -1)
        self.object_boxes_image_coordinates = (
                self.object_boxes_normalized
                * np.reshape([25, 73, 25, 73], [1, 1, 4]))

    def test_on_iteration_end(self):
        callback = NormalizeCoordinatesCallback(
            inbound_nodes=[]).build()
        inputs = {"images": self.images,
                  "object_boxes": self.object_boxes_image_coordinates}
        outputs = callback.on_iteration_end(**inputs)
        self.assertSetEqual(set(callback.generated_keys),
                            set(outputs))

        outputs_must = {
            "object_boxes": self.object_boxes_normalized
        }
        self.assertAllClose(outputs_must,
                            outputs)
