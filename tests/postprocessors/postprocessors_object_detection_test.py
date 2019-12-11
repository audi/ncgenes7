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

from ncgenes7.postprocessors.object_detection import (
    ConverterToImageFramePostprocessor)
from ncgenes7.postprocessors.object_detection import (
    DetectionsFilterPostprocessor)
from ncgenes7.postprocessors.object_detection import DetectionsPostprocessor
from ncgenes7.postprocessors.object_detection import (
    ExtractKeypointsFromHeatmaps)
from ncgenes7.postprocessors.object_detection import InstanceMasksToImageFrame
from ncgenes7.postprocessors.object_detection import (
    KeypointsFilterPostprocessor)
from ncgenes7.postprocessors.object_detection import (
    NonMaxSuppressionPostprocessor)
from ncgenes7.postprocessors.object_detection import ObjectClassesCombiner


class TestDetectionsPostprocessor(parameterized.TestCase,
                                  tf.test.TestCase):

    def setUp(self):
        np.random.seed(6547)
        self.max_num_detections_np = 10
        self.batch_size = 3
        self.num_object_detections_np = np.array([2, 3, 0])
        self.detection_object_boxes_np = np.random.rand(
            self.batch_size, self.max_num_detections_np, 4)
        self.detection_object_scores_np = np.random.rand(
            self.batch_size, self.max_num_detections_np)
        self.detection_object_classes_np = np.random.randint(
            0, 3, size=(self.batch_size, self.max_num_detections_np))
        self.detection_object_instance_ids_np = np.random.randint(
            0, 3, size=(self.batch_size, self.max_num_detections_np))
        self.additional_input1_np = np.random.rand(
            self.batch_size, self.max_num_detections_np, 2, 5)
        self.additional_input2_np = np.random.rand(
            self.batch_size, self.max_num_detections_np)

        self.num_object_detections = tf.placeholder(tf.int32, [None])
        self.detection_object_boxes = tf.placeholder(
            tf.float32, [None, None, 4])
        self.detection_object_scores = tf.placeholder(
            tf.float32, [None, None])
        self.detection_object_classes = tf.placeholder(
            tf.int32, [None, None])
        self.detection_object_instance_ids = tf.placeholder(
            tf.int32, [None, None])
        self.additional_input1 = tf.placeholder(
            tf.float32, [None, None, 2, 5])
        self.additional_input2 = tf.placeholder(
            tf.float32, [None, None])

    @parameterized.parameters(
        {"provide_classes": True, "offset_detection_classes": True},
        {"provide_classes": True, "offset_detection_classes": True,
         "provide_additional_inputs": True},
        {"provide_classes": True, "offset_detection_classes": False},
        {"provide_classes": False, "offset_detection_classes": True},
        {"provide_classes": False, "offset_detection_classes": False},
        {"provide_classes": True, "offset_detection_classes": True,
         "provide_ids": True},
        {"provide_classes": False, "offset_detection_classes": False,
         "provide_ids": True},
        {"provide_classes": False, "offset_detection_classes": False,
         "provide_ids": True, "provide_additional_inputs": True},
    )
    def test_process(self, provide_classes, offset_detection_classes,
                     provide_ids=False, provide_additional_inputs=False):
        postprocessor = DetectionsPostprocessor(
            offset_detection_classes=offset_detection_classes,
            inbound_nodes=[]).build()

        inputs = {"detection_object_boxes": self.detection_object_boxes,
                  "detection_object_scores": self.detection_object_scores,
                  "num_object_detections": self.num_object_detections}
        if provide_classes:
            inputs["detection_object_classes"] = self.detection_object_classes
        if provide_ids:
            inputs["detection_object_instance_ids"] = (
                self.detection_object_instance_ids)
        if provide_additional_inputs:
            inputs["key1"] = self.additional_input1
            inputs["key2"] = self.additional_input2

        outputs = postprocessor.process(**inputs)
        generated_keys_must = list(postprocessor.generated_keys)
        if provide_additional_inputs:
            generated_keys_must += ["key1", "key2"]

        self.assertSetEqual(set(generated_keys_must),
                            set(outputs))

        feed_dict = {
            self.detection_object_boxes: self.detection_object_boxes_np,
            self.detection_object_scores: self.detection_object_scores_np,
            self.num_object_detections: self.num_object_detections_np
        }
        if provide_classes:
            feed_dict[self.detection_object_classes] = (
                self.detection_object_classes_np)
        if provide_ids:
            feed_dict[self.detection_object_instance_ids] = (
                self.detection_object_instance_ids_np)
        if provide_additional_inputs:
            feed_dict[self.additional_input1] = self.additional_input1_np
            feed_dict[self.additional_input2] = self.additional_input2_np
        with self.test_session() as sess:
            outputs_eval = sess.run(outputs, feed_dict=feed_dict)

        detections_mask = np.expand_dims(np.arange(
            self.max_num_detections_np), 0)
        detections_mask = np.less(
            detections_mask,
            np.expand_dims(self.num_object_detections_np, -1))

        outputs_must = {
            "detection_object_boxes": np.where(
                np.expand_dims(detections_mask, -1),
                self.detection_object_boxes_np,
                np.zeros_like(self.detection_object_boxes_np)),
            "detection_object_scores": np.where(
                detections_mask,
                self.detection_object_scores_np,
                np.zeros_like(self.detection_object_scores_np)),
            "num_object_detections": self.num_object_detections_np,
            "detection_object_instance_ids":
                np.zeros_like(detections_mask, np.int32)
        }
        if provide_classes:
            outputs_must["detection_object_classes"] = np.where(
                detections_mask,
                self.detection_object_classes_np,
                np.zeros_like(self.detection_object_classes_np))
            if offset_detection_classes:
                outputs_must["detection_object_classes"] += (
                    detections_mask.astype(np.int32))
        else:
            outputs_must["detection_object_classes"] = (
                detections_mask.astype(np.int32))
        if provide_ids:
            outputs_must["detection_object_instance_ids"] = np.where(
                detections_mask,
                self.detection_object_instance_ids_np,
                np.zeros_like(self.detection_object_instance_ids_np))

        if provide_additional_inputs:
            outputs_must["key1"] = np.where(
                detections_mask[:, :, np.newaxis, np.newaxis],
                self.additional_input1_np,
                np.zeros_like(self.additional_input1_np))
            outputs_must["key2"] = np.where(
                detections_mask,
                self.additional_input2_np,
                np.zeros_like(self.additional_input2_np))

        self.assertAllClose(outputs_must,
                            outputs_eval)


class TestDetectionsFilterPostprocessor(parameterized.TestCase,
                                        tf.test.TestCase):
    def setUp(self):
        np.random.seed(6547)
        self.max_num_detections_np = 16
        self.batch_size = 3
        self.num_object_detections_np = np.array([6, 14, 0])
        object_boxes_min = np.random.rand(
            self.batch_size, self.max_num_detections_np, 2)
        object_boxes_height_width = np.random.rand(
            self.batch_size, self.max_num_detections_np, 2)
        self.detection_object_boxes_np = np.concatenate(
            [object_boxes_min, object_boxes_min + object_boxes_height_width],
            -1)
        self.detection_object_scores_np = np.random.rand(
            self.batch_size, self.max_num_detections_np)
        self.detection_object_classes_np = np.random.randint(
            1, 4, size=(self.batch_size, self.max_num_detections_np))
        self.detection_object_instance_ids_np = np.random.randint(
            0, 3, size=(self.batch_size, self.max_num_detections_np))
        self.additional_input1_np = np.random.rand(
            self.batch_size, self.max_num_detections_np, 2, 5)
        self.additional_input2_np = np.random.rand(
            self.batch_size, self.max_num_detections_np)

        self.num_object_detections = tf.placeholder(tf.int32, [None])
        self.detection_object_boxes = tf.placeholder(
            tf.float32, [None, None, 4])
        self.detection_object_scores = tf.placeholder(
            tf.float32, [None, None])
        self.detection_object_classes = tf.placeholder(
            tf.int32, [None, None])
        self.detection_object_instance_ids = tf.placeholder(
            tf.int32, [None, None])
        self.additional_input1 = tf.placeholder(
            tf.float32, [None, None, 2, 5])
        self.additional_input2 = tf.placeholder(
            tf.float32, [None, None])

    @parameterized.parameters(
        {"provide_classes": False, "provide_ids": False},
        {"provide_classes": False, "provide_ids": False,
         "provide_additional_inputs": True},
        {"provide_classes": False, "provide_ids": False,
         "min_object_height": 0.1, "max_object_height": 0.7,
         "min_object_width": 0.2, "max_object_width": 0.6,
         "provide_additional_inputs": True},
        {"min_object_height": 0.1, "max_object_height": 0.7,
         "min_object_width": 0.2, "max_object_width": 0.6},
        {"min_object_height": 0.1, "max_object_height": 0.7,
         "min_object_width": 0.2, "max_object_width": 0.6,
         "score_threshold": 0.4},
        {"min_object_height": 0.1, "max_object_height": 0.7,
         "min_object_width": 0.2, "max_object_width": 0.6,
         "score_threshold": 0.4, "provide_additional_inputs": True},
        {"min_object_height": 1.0, "max_object_height": 2.0,
         "min_object_width": 0., "max_object_width": 0.},
        {"min_object_height": 1.0, "max_object_height": 2.0,
         "min_object_width": 0., "max_object_width": 0.,
         "provide_additional_inputs": True},
        {"min_object_height": [0.0, 0.1, 0.1],
         "max_object_height": 0.0,
         "score_threshold": [0, 0.2, 0.3],
         "min_object_width": 0., "max_object_width": 0.,
         "provide_additional_inputs": True},
        {"min_object_height": 0.1,
         "max_object_height": 0.0,
         "score_threshold": [0, 0.2, 0.3],
         "min_object_width": 0., "max_object_width": 2.,
         "provide_additional_inputs": False},
        {"min_object_height": 1.0, "max_object_height": 2.0,
         "min_object_width": 0., "max_object_width": 0.,
         "provide_additional_inputs": True,
         "reorder_filtered": False},
        {"min_object_height": [0.0, 0.1, 0.1],
         "max_object_height": 0.0,
         "score_threshold": [0, 0.2, 0.3],
         "min_object_width": 0., "max_object_width": 0.,
         "provide_additional_inputs": True,
         "reorder_filtered": False},
        {"min_object_height": 0.1,
         "max_object_height": 0.0,
         "score_threshold": [0, 0.2, 0.3],
         "min_object_width": 0., "max_object_width": 2.,
         "provide_additional_inputs": False,
         "reorder_filtered": False},
        {"min_object_height": 0.1,
         "max_object_height": 0.0,
         "score_threshold": [0, 0.2, 0.3],
         "min_object_width": 0., "max_object_width": 2.,
         "provide_additional_inputs": False,
         "classes_to_select": 1,
         "reorder_filtered": False},
        {"min_object_height": 0.1,
         "max_object_height": 0.0,
         "score_threshold": [0, 0.2, 0.3],
         "min_object_width": 0., "max_object_width": 2.,
         "provide_additional_inputs": False,
         "classes_to_select": [1, 2],
         "reorder_filtered": True},
    )
    def test_process(self,
                     min_object_height=0,
                     max_object_height=0,
                     min_object_width=0,
                     max_object_width=0,
                     score_threshold=0,
                     provide_classes=True,
                     provide_ids=True,
                     provide_additional_inputs=False,
                     reorder_filtered=True,
                     classes_to_select=None):
        postprocessor = DetectionsFilterPostprocessor(
            num_classes=3,
            min_object_height=min_object_height,
            max_object_height=max_object_height,
            min_object_width=min_object_width,
            max_object_width=max_object_width,
            score_threshold=score_threshold,
            reorder_filtered=reorder_filtered,
            classes_to_select=classes_to_select,
            inbound_nodes=[],
        ).build()

        inputs = {"detection_object_boxes": self.detection_object_boxes,
                  "detection_object_scores": self.detection_object_scores,
                  "num_object_detections": self.num_object_detections}
        if provide_classes:
            inputs["detection_object_classes"] = self.detection_object_classes
        if provide_ids:
            inputs["detection_object_instance_ids"] = (
                self.detection_object_instance_ids)
        if provide_additional_inputs:
            inputs["key1"] = self.additional_input1
            inputs["key2"] = self.additional_input2

        outputs = postprocessor.process(**inputs)
        generated_keys_must = list(postprocessor.generated_keys)
        if provide_additional_inputs:
            generated_keys_must += ["key1", "key2"]

        self.assertSetEqual(set(generated_keys_must),
                            set(outputs))

        feed_dict = {
            self.detection_object_boxes: self.detection_object_boxes_np,
            self.detection_object_scores: self.detection_object_scores_np,
            self.num_object_detections: self.num_object_detections_np
        }
        if provide_classes:
            feed_dict[self.detection_object_classes] = (
                self.detection_object_classes_np)
        if provide_ids:
            feed_dict[self.detection_object_instance_ids] = (
                self.detection_object_instance_ids_np)
        if provide_additional_inputs:
            feed_dict[self.additional_input1] = self.additional_input1_np
            feed_dict[self.additional_input2] = self.additional_input2_np
        with self.test_session() as sess:
            outputs_eval = sess.run(outputs, feed_dict=feed_dict)

        class_masks = [self.detection_object_classes_np == i + 1
                       for i in range(3)]

        object_widths = (self.detection_object_boxes_np[..., 3]
                         - self.detection_object_boxes_np[..., 1])
        object_heights = (self.detection_object_boxes_np[..., 2]
                          - self.detection_object_boxes_np[..., 0])

        mask_by_width = object_widths > min_object_width
        if max_object_width > 0:
            mask_by_width = np.logical_and(object_widths < max_object_width,
                                           mask_by_width)
        if isinstance(min_object_height, list):
            mask_by_height = np.any(np.stack(
                [np.logical_and(each_class_mask,
                                object_heights > each_max_height)
                 for each_class_mask, each_max_height in zip(
                    class_masks, min_object_height)
                 ], -1), -1
            )
        else:
            mask_by_height = object_heights > min_object_height
        if max_object_height > 0:
            mask_by_height = np.logical_and(object_heights < max_object_height,
                                            mask_by_height)

        if isinstance(score_threshold, list):
            object_scores = self.detection_object_scores_np
            mask_by_scores = np.any(np.stack(
                [np.logical_and(each_class_mask,
                                object_scores > each_score_th)
                 for each_class_mask, each_score_th in zip(
                    class_masks, score_threshold)
                 ], -1), -1
            )
        else:
            mask_by_scores = self.detection_object_scores_np >= score_threshold
        mask_filtered = np.logical_and(
            mask_by_scores,
            np.logical_and(mask_by_height, mask_by_width))
        detections_mask = np.expand_dims(np.arange(
            self.max_num_detections_np), 0)
        detections_mask = np.less(
            detections_mask,
            np.expand_dims(self.num_object_detections_np, -1))
        mask_filtered = np.logical_and(mask_filtered, detections_mask)
        if classes_to_select:
            classes_to_select = (
                classes_to_select if isinstance(classes_to_select, list)
                else [classes_to_select])
            mask_classes = np.any(np.stack(
                [self.detection_object_classes_np == each_class
                 for each_class in classes_to_select], -1), -1)
            mask_filtered = np.logical_and(mask_filtered, mask_classes)

        if reorder_filtered:
            num_objects_filtered_must = mask_filtered.sum(-1)
        else:
            num_objects_filtered_must = self.num_object_detections_np

        inputs_all = {
            "detection_object_boxes": self.detection_object_boxes_np,
            "detection_object_scores": self.detection_object_scores_np,
            "detection_object_classes": (
                self.detection_object_classes_np if provide_classes
                else np.zeros_like(self.detection_object_classes_np)),
            "detection_object_instance_ids": (
                self.detection_object_instance_ids_np if provide_ids
                else np.zeros_like(self.detection_object_instance_ids_np))
        }
        if provide_additional_inputs:
            inputs_all["key1"] = self.additional_input1_np
            inputs_all["key2"] = self.additional_input2_np
        self.assertAllClose(num_objects_filtered_must,
                            outputs_eval["num_object_detections"])
        for each_sample_i in range(self.batch_size):
            num_objects_filtered_i = num_objects_filtered_must[each_sample_i]
            result_empty_must = {
                k: np.zeros_like(v[each_sample_i][num_objects_filtered_i:])
                for k, v in inputs_all.items()
                if k != "num_object_detections"}
            result_empty_eval = {
                k: v[each_sample_i][num_objects_filtered_i:]
                for k, v in outputs_eval.items()
                if k != "num_object_detections"}
            self.assertAllClose(result_empty_must,
                                result_empty_eval)

            if reorder_filtered:
                inds_must = np.argsort(np.logical_not(
                    mask_filtered[each_sample_i]), -1)[:num_objects_filtered_i]
            else:
                inds_must = np.arange(num_objects_filtered_i)

            if reorder_filtered:
                result_not_empty_must = {
                    k: v[each_sample_i][inds_must]
                    for k, v in inputs_all.items()
                    if k != "num_object_detections"}
            else:
                result_not_empty_must = {}
                for k, v in inputs_all.items():
                    if k == "num_object_detections":
                        continue
                    mask = mask_filtered[each_sample_i]
                    shape_diff = len(v[each_sample_i].shape) - len(mask.shape)
                    if shape_diff:
                        mask = np.reshape(
                            mask, mask.shape + tuple([1]*shape_diff))
                    result_not_empty_must[k] = (
                            mask * v[each_sample_i])[inds_must]

            result_not_empty_eval = {
                k: v[each_sample_i][:num_objects_filtered_i]
                for k, v in outputs_eval.items()
                if k != "num_object_detections"}

            self.assertAllClose(result_not_empty_must,
                                result_not_empty_eval)


class TestNonMaxSuppressionPostprocessor(parameterized.TestCase,
                                         tf.test.TestCase):

    def setUp(self):
        self.num_classes = 4
        self.batch_size = 2
        self.max_num_detections_np = 5
        self.detection_object_boxes_np = np.array(
            [[[0.1, 0.2, 0.3, 0.4],
              [0.1, 0.2, 0.31, 0.41],
              [0.5, 0.6, 0.7, 0.8],
              [0.5, 0.6, 0.7, 0.82],
              [0, 0, 0, 0]],
             [[0.5, 0.2, 0.6, 0.1],
              [0.1, 0.2, 0.31, 0.41],
              [0.5, 0.2, 0.6, 0.1],
              [0, 0, 0, 0],
              [0, 0, 0, 0]]],
            np.float32
        )
        self.num_object_detections_np = np.array(
            [4, 3])
        self.detection_object_classes_np = np.array(
            [[0, 0, 2, 2, 0],
             [1, 0, 3, 0, 0]], np.int32)
        self.detection_object_scores_np = np.array(
            [[0.5, 0.6, 0.7, 0.2, 0],
             [0.7, 0.6, 0.1, 0, 0]], np.float32)
        self.detection_object_instance_ids_np = np.array(
            [[0, 1, 2, 10, -1],
             [10, 5, 1, -1, -1]]
        )
        self.additional_input1_np = np.random.rand(
            self.batch_size, self.max_num_detections_np, 2, 5)
        self.additional_input2_np = np.random.rand(
            self.batch_size, self.max_num_detections_np)

        self.num_object_detections = tf.placeholder(tf.int32, [None])
        self.detection_object_boxes = tf.placeholder(
            tf.float32, [None, None, 4])
        self.detection_object_scores = tf.placeholder(
            tf.float32, [None, None])
        self.detection_object_classes = tf.placeholder(
            tf.int32, [None, None])
        self.detection_object_instance_ids = tf.placeholder(
            tf.int32, [None, None])
        self.additional_input1 = tf.placeholder(
            tf.float32, [None, None, 2, 5])
        self.additional_input2 = tf.placeholder(
            tf.float32, [None, None])

        self.iou_threshold = [0.5, 0.6, 0.5, 0.1]
        self.max_size_per_class = 50
        self.max_total_size = 60

    @parameterized.parameters(
        {"provide_classes": False, "provide_ids": False},
        {"provide_classes": False, "provide_ids": False,
         "provide_additional_inputs": True},
        {"provide_classes": True, "provide_ids": True},
        {"provide_classes": True, "provide_ids": True,
         "provide_additional_inputs": True},
    )
    def test_process(self, provide_classes=True, provide_ids=True,
                     provide_additional_inputs=False):
        num_classes = self.num_classes if provide_ids else 1
        postprocessor = NonMaxSuppressionPostprocessor(
            num_classes=num_classes,
            iou_threshold=self.iou_threshold if num_classes > 1 else 0.5,
            max_total_size=self.max_total_size,
            max_size_per_class=self.max_size_per_class,
            inbound_nodes=[]).build()

        inputs = {"detection_object_boxes": self.detection_object_boxes,
                  "detection_object_scores": self.detection_object_scores,
                  "num_object_detections": self.num_object_detections}
        if provide_classes:
            inputs["detection_object_classes"] = self.detection_object_classes
        if provide_ids:
            inputs["detection_object_instance_ids"] = (
                self.detection_object_instance_ids)
        if provide_additional_inputs:
            inputs["key1"] = self.additional_input1
            inputs["key2"] = self.additional_input2

        outputs = postprocessor.process(**inputs)
        generated_keys_must = list(postprocessor.generated_keys)
        if provide_additional_inputs:
            generated_keys_must += ["key1", "key2"]

        feed_dict = {
            self.detection_object_boxes: self.detection_object_boxes_np,
            self.detection_object_scores: self.detection_object_scores_np,
            self.num_object_detections: self.num_object_detections_np
        }
        if provide_classes:
            feed_dict[self.detection_object_classes] = (
                self.detection_object_classes_np)
        if provide_ids:
            feed_dict[self.detection_object_instance_ids] = (
                self.detection_object_instance_ids_np)
        if provide_additional_inputs:
            feed_dict[self.additional_input1] = self.additional_input1_np
            feed_dict[self.additional_input2] = self.additional_input2_np

        with self.test_session() as sess:
            outputs_eval = sess.run(outputs, feed_dict=feed_dict)
        outputs_must = self._get_outputs_must(provide_classes, provide_ids,
                                              provide_additional_inputs)
        self.assertAllClose(outputs_must,
                            outputs_eval)

    def _get_outputs_must(self, provide_classes, provide_ids,
                          provide_additional_inputs):
        if provide_classes:
            selected_inds = np.array([[2, 1, 0, 0, 0],
                                      [0, 1, 2, 0, 0]])
            result_must = {
                "detection_object_boxes": np.array(
                    [[[0.5, 0.6, 0.7, 0.8],
                      [0.1, 0.2, 0.31, 0.41],
                      [0, 0, 0, 0],
                      [0, 0, 0, 0],
                      [0, 0, 0, 0]],
                     [[0.5, 0.2, 0.6, 0.1],
                      [0.1, 0.2, 0.31, 0.41],
                      [0.5, 0.2, 0.6, 0.1],
                      [0, 0, 0, 0],
                      [0, 0, 0, 0]]],
                    np.float32
                ),
                "detection_object_scores": np.array(
                    [[0.7, 0.6, 0.0, 0.0, 0],
                     [0.7, 0.6, 0.1, 0, 0]], np.float32),
                "detection_object_classes": np.array(
                    [[2, 0, 0, 0, 0],
                     [1, 0, 3, 0, 0]], np.int32),
                "num_object_detections": np.array(
                    [2, 3]),
            }
            if provide_ids:
                detection_object_instance_ids = np.array(
                    [[2, 1, 0, 0, 0],
                     [10, 5, 1, 0, 0]]
                )
            else:
                detection_object_instance_ids = np.zeros_like(
                    self.detection_object_instance_ids_np)
            result_must["detection_object_instance_ids"] = (
                detection_object_instance_ids)
        else:
            selected_inds = np.array([[2, 1, 0, 0, 0],
                                      [0, 1, 0, 0, 0]])
            result_must = {
                "detection_object_boxes": np.array(
                    [[[0.5, 0.6, 0.7, 0.8],
                      [0.1, 0.2, 0.31, 0.41],
                      [0, 0, 0, 0],
                      [0, 0, 0, 0],
                      [0, 0, 0, 0]],
                     [[0.5, 0.2, 0.6, 0.1],
                      [0.1, 0.2, 0.31, 0.41],
                      [0, 0, 0, 0],
                      [0, 0, 0, 0],
                      [0, 0, 0, 0]]],
                    np.float32
                ),
                "detection_object_scores": np.array(
                    [[0.7, 0.6, 0.0, 0.0, 0],
                     [0.7, 0.6, 0.0, 0, 0]], np.float32),
                "detection_object_classes": np.array(
                    [[0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0]], np.int32),
                "num_object_detections": np.array(
                    [2, 2]),
            }
            if provide_ids:
                detection_object_instance_ids = np.array(
                    [[2, 1, 0, 0, 0],
                     [10, 5, 0, 0, 0]]
                )
            else:
                detection_object_instance_ids = np.zeros_like(
                    self.detection_object_instance_ids_np)
            result_must["detection_object_instance_ids"] = (
                detection_object_instance_ids)

        if provide_additional_inputs:
            num_object_detections = result_must["num_object_detections"]
            additional_results_must = {}
            additional_inputs = {"key1": self.additional_input1_np,
                                 "key2": self.additional_input2_np}
            for each_key in ["key1", "key2"]:
                additional_result = []
                for i in range(self.batch_size):
                    inds = selected_inds[i]
                    num_objects = num_object_detections[i]
                    additional_sample_result = (
                        additional_inputs[each_key][i][inds])
                    additional_sample_result[num_objects:] = 0
                    additional_result.append(additional_sample_result)

                additional_results_must[each_key] = np.stack(
                    additional_result, 0)
            result_must.update(additional_results_must)
        return result_must


class TestConverterToImageFramePostprocessor(tf.test.TestCase):

    def setUp(self):
        self.images_np = np.random.rand(3, 25, 73, 3)
        object_boxes_min = np.random.rand(3, 10, 2)
        object_boxes_height_width = np.random.rand(3, 10, 2)
        self.detection_object_boxes_np = np.concatenate(
            [object_boxes_min, object_boxes_min + object_boxes_height_width],
            -1)

        self.images = tf.placeholder(tf.float32, [None, None, None, 3])
        self.detection_object_boxes = tf.placeholder(
            tf.float32, [None, None, 4])

    def test_process(self):
        postprocessor = ConverterToImageFramePostprocessor(
            inbound_nodes=[]).build()
        inputs = {"images": self.images,
                  "detection_object_boxes": self.detection_object_boxes}
        outputs = postprocessor.process(**inputs)
        self.assertSetEqual(set(postprocessor.generated_keys),
                            set(outputs))
        feed_dict = {
            self.images: self.images_np,
            self.detection_object_boxes: self.detection_object_boxes_np,
        }
        with self.test_session() as sess:
            outputs_eval = sess.run(outputs, feed_dict=feed_dict)

        detection_object_boxes_must = (
                self.detection_object_boxes_np
                * np.reshape([25, 73, 25, 73], [1, 1, 4]))

        outputs_must = {
            "detection_object_boxes": detection_object_boxes_must
        }
        self.assertAllClose(outputs_must,
                            outputs_eval)


class TestExtractKeypointsFromHeatmaps(tf.test.TestCase,
                                       parameterized.TestCase):

    def setUp(self):
        self.boxes_np = np.array([
            [[0, 0, 0.2, 0.5],
             [0.4, 0.1, 0.7, 0.3]],
            [[0.1, 0.5, 0.3, 0.9],
             [0, 0, 0, 0]]
        ], np.float32)
        self.num_keypoints = 5
        self.keypoints_heatmaps_np = np.random.rand(
            2, 2, 11, 17, self.num_keypoints)

        self.boxes = tf.placeholder(tf.float32, [None, None, 4])
        self.keypoints_heatmaps = tf.placeholder(
            tf.float32, [None, None, None, None, self.num_keypoints])

    @parameterized.parameters(
        {"smoothing_kernel_size": 1, "normalize_smoothing_kernel": True},
        {"smoothing_kernel_size": 3, "normalize_smoothing_kernel": True},
        {"smoothing_kernel_size": 1, "normalize_smoothing_kernel": False})
    def test_process(self, smoothing_kernel_size,
                     normalize_smoothing_kernel):
        postprocessor = ExtractKeypointsFromHeatmaps(
            smoothing_kernel_size=smoothing_kernel_size,
            normalize_smoothing_kernel=normalize_smoothing_kernel
        ).build()
        result = postprocessor.process(
            detection_object_keypoints_heatmaps=self.keypoints_heatmaps,
            detection_object_boxes=self.boxes)
        self.assertSetEqual(set(postprocessor.generated_keys),
                            set(result))
        self.assertListEqual(
            [None, None, self.num_keypoints, 2],
            result["detection_object_keypoints"].shape.as_list())
        self.assertListEqual(
            [None, None, self.num_keypoints],
            result["detection_object_keypoints_scores"].shape.as_list())
        with self.test_session() as sess:
            feed_dict = {
                self.keypoints_heatmaps: self.keypoints_heatmaps_np,
                self.boxes: self.boxes_np}
            _ = sess.run(result, feed_dict=feed_dict)


class TestKeypointsFilterPostprocessor(tf.test.TestCase):

    def setUp(self):
        self.keypoints_np = np.array([
            [[[5, 10],
              [5, 20],
              [1, 5],
              [0, 0],
              [0, 5]],
             [[5, 15],
              [5, 7],
              [6, 10],
              [15, 17],
              [5, 7]]],
            [[[9, 4],
              [5, 1],
              [1, 0],
              [5, 4],
              [1, 5]],
             [[0, 0],
              [0, 0],
              [0, 0],
              [0, 0],
              [0, 0]]]
        ], np.float32)
        self.keypoints_scores_np = np.array([
            [[0.5, 0.4, 0.3, 0.3, 0.9],
             [0.1, 0.3, 0.5, 0.1, 0.1]],
            [[0.7, 0.9, 0.1, 0.1, 0.1],
             [0.0, 0.0, 0.0, 0.0, 0.0]]
        ], np.float32)
        self.score_threshold = 0.3

        self.keypoints = tf.placeholder(tf.float32, [None, None, None, 2])
        self.keypoints_scores = tf.placeholder(tf.float32, [None, None, None])
        self.scores_must = np.array([
            [[0.5, 0.4, 0.3, 0.3, 0.9],
             [0.0, 0.3, 0.5, 0.0, 0.0]],
            [[0.7, 0.9, 0.0, 0.0, 0.0],
             [0.0, 0.0, 0.0, 0.0, 0.0]]
        ], np.float32)
        self.keypoints_must = np.array([
            [[[5, 10],
              [5, 20],
              [1, 5],
              [0, 0],
              [0, 5]],
             [[0, 0],
              [5, 7],
              [6, 10],
              [0, 0],
              [0, 0]]],
            [[[9, 4],
              [5, 1],
              [0, 0],
              [0, 0],
              [0, 0]],
             [[0, 0],
              [0, 0],
              [0, 0],
              [0, 0],
              [0, 0]]]
        ], np.float32)

    def test_process(self):
        postprocessor = KeypointsFilterPostprocessor(
            score_threshold=self.score_threshold).build()
        result = postprocessor.process(
            detection_object_keypoints=self.keypoints,
            detection_object_keypoints_scores=self.keypoints_scores)

        self.assertSetEqual(set(postprocessor.generated_keys),
                            set(result))
        self.assertListEqual(
            [None, None, None, 2],
            result["detection_object_keypoints"].shape.as_list())
        self.assertListEqual(
            [None, None, None],
            result["detection_object_keypoints_scores"].shape.as_list())
        with self.test_session() as sess:
            feed_dict = {
                self.keypoints: self.keypoints_np,
                self.keypoints_scores: self.keypoints_scores_np}
            result_eval = sess.run(result, feed_dict=feed_dict)

        result_must = {
            "detection_object_keypoints": self.keypoints_must,
            "detection_object_keypoints_scores": self.scores_must,
        }
        self.assertAllClose(result_must,
                            result_eval)


class TestInstanceMasksToImageFrame(tf.test.TestCase,
                                    parameterized.TestCase):

    @parameterized.parameters({"masks_num_channels": 0},
                              {"masks_num_channels": 3})
    def test_process(self, masks_num_channels):
        images = tf.placeholder(tf.float32, [None, None, None, 3])
        boxes = tf.placeholder(tf.float32, [None, None, 4])
        if masks_num_channels > 0:
            object_instance_masks = tf.placeholder(
                tf.float32, [None, None, None, None, masks_num_channels])
        else:
            object_instance_masks = tf.placeholder(
                tf.float32, [None, None, None, None])

        postprocessor = InstanceMasksToImageFrame().build()
        result = postprocessor.process(
            images=images,
            object_boxes=boxes,
            object_instance_masks=object_instance_masks)
        self.assertSetEqual(set(postprocessor.generated_keys),
                            set(result))
        self.assertListEqual(
            [None, None, None, None],
            result["object_instance_masks_on_image"].shape.as_list())


class TestObjectClassesCombiner(tf.test.TestCase):

    def setUp(self):
        self.initial_classes_np = np.array(
            [[4, 1, 2, 2, 0],
             [1, 1, 3, 0, 0]], np.int32)
        self.additional_class1_np = np.array(
            [[1, 1, 0, 0, 0],
             [1, 1, 2, 0, 0]], np.int32)
        self.additional_class2_np = np.array(
            [[0, 3, 5, 0, 0],
             [4, 2, 0, 0, 9]], np.int32)
        self.num_classes_to_combine = [
            {"initial": 5}, {"class1": 10}, {"class2": 20}
        ]
        self.classes_combined_must = {
            "object_classes":
                np.array([[9, 156, 252, 2, 0],
                          [206, 106, 13, 0, 0]], np.int32)
        }

        self.object_classes = tf.placeholder(tf.int32, (None, None))
        self.additional_class1 = tf.placeholder(tf.int32, (None, None))
        self.additional_class2 = tf.placeholder(tf.int32, (None, None))

    def test_process(self):
        postprocessor = ObjectClassesCombiner(
            num_classes_to_combine=self.num_classes_to_combine).build()

        result = postprocessor.process(initial=self.object_classes,
                                       class1=self.additional_class1,
                                       class2=self.additional_class2)

        with self.test_session() as sess:
            feed_dict = {
                self.object_classes: self.initial_classes_np,
                self.additional_class1: self.additional_class1_np,
                self.additional_class2: self.additional_class2_np
            }
            result_eval = sess.run(result, feed_dict)

        self.assertAllClose(self.classes_combined_must,
                            result_eval)
