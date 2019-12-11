# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

import json
import os

from absl.testing import parameterized
from nucleus7.utils import tf_data_utils
import numpy as np
import tensorflow as tf

from ncgenes7.data_fields.images import ImageDataFields
from ncgenes7.data_fields.object_detection import ObjectDataFields
from ncgenes7.data_readers.object_detection import BoxAdjusterByKeypoints
from ncgenes7.data_readers.object_detection import KeypointsReaderTfRecords
from ncgenes7.data_readers.object_detection import ObjectClassSelectorTf
from ncgenes7.data_readers.object_detection import ObjectDetectionReader
from ncgenes7.data_readers.object_detection import ObjectDetectionReaderTF
from ncgenes7.data_readers.object_detection import (
    ObjectDetectionReaderTfRecords)
from ncgenes7.utils import object_detection_utils


class TestObjectDetectionReader(tf.test.TestCase, parameterized.TestCase):
    def setUp(self):
        self.image_size = [400, 500]
        self.batch_size = 2
        self.detections = [
            {'id': 1, 'class_label': 2,
             'bbox': {'xmin': 10, 'ymin': 20, 'xmax': 30, 'ymax': 40},
             'score': 0.1},
            {'id': 2, 'class_label': 3,
             'bbox': {'xmin': 100, 'ymin': 200, 'xmax': 400, 'ymax': 600},
             'score': 0.2}
        ]
        self.object_fname = os.path.join(self.get_temp_dir(), "objects.json")
        self.result_must = {
            "object_boxes": np.array(
                [[20, 10, 40, 30],
                 [200, 100, 600, 400]]),
            "object_classes": np.array([2, 3]),
            "object_scores": np.array([0.1, 0.2]),
            "object_instance_ids": [1, 2],
            "num_objects": 2,
            "object_fnames": os.path.basename(self.object_fname),
        }
        with open(self.object_fname, "w") as f:
            json.dump(self.detections, f)

    @parameterized.parameters({"normalize_boxes": False},
                              {"normalize_boxes": True})
    def test_read(self, normalize_boxes):
        data_feeder = ObjectDetectionReader(
            image_size=self.image_size,
            normalize_boxes=normalize_boxes,
        ).build()
        result = data_feeder.read(labels=self.object_fname)
        result_must = self.result_must
        if normalize_boxes:
            result_must["object_boxes"] = (
                object_detection_utils.normalize_bbox_np(
                    result_must["object_boxes"], self.image_size))
        self.assertSetEqual(set(result_must),
                            set(result))
        for each_key in result_must:
            if each_key == "object_fnames":
                self.assertEqual(result_must[each_key],
                                 result[each_key])
            else:
                self.assertAllClose(result_must[each_key],
                                    result[each_key])


class TestObjectDetectionReaderTF(tf.test.TestCase, parameterized.TestCase):
    def setUp(self):
        self.image_size = [400, 500]
        self.batch_size = 2
        self.detections = [
            {'id': 1, 'class_label': 2,
             'bbox': {'xmin': 10, 'ymin': 20, 'xmax': 30, 'ymax': 40},
             'score': 0.1},
            {'id': 2, 'class_label': 3,
             'bbox': {'xmin': 100, 'ymin': 200, 'xmax': 400, 'ymax': 600},
             'score': 0.2}
        ]
        self.object_fname = os.path.join(self.get_temp_dir(), "objects.json")
        self.result_must = {
            "object_boxes": np.array(
                [[20, 10, 40, 30],
                 [200, 100, 600, 400]]),
            "object_classes": np.array([2, 3]),
            "object_scores": np.array([0.1, 0.2]),
            "object_instance_ids": [1, 2],
            "num_objects": 2,
            "object_fnames": os.path.basename(self.object_fname),
        }
        with open(self.object_fname, "w") as f:
            json.dump(self.detections, f)

    @parameterized.parameters({"normalize_boxes": False},
                              {"normalize_boxes": True})
    def test_read(self, normalize_boxes):
        data_feeder = ObjectDetectionReaderTF(
            image_size=self.image_size,
            normalize_boxes=normalize_boxes,
        ).build()
        object_fname_tensor = tf.convert_to_tensor(self.object_fname)
        result = data_feeder.read(labels=object_fname_tensor)

        result_eval = self.evaluate(result)
        result_must = self.result_must
        if normalize_boxes:
            result_must["object_boxes"] = (
                object_detection_utils.normalize_bbox_np(
                    result_must["object_boxes"], self.image_size))
        self.assertSetEqual(set(result_must),
                            set(result))
        for each_key in result_must:
            if each_key == "object_fnames":
                self.assertEqual(result_must[each_key],
                                 result_eval[each_key].decode())
            else:
                self.assertAllClose(result_must[each_key],
                                    result_eval[each_key])


class TestObjectDetectionDatasetTfRecords(tf.test.TestCase,
                                          parameterized.TestCase):

    def setUp(self):
        np.random.seed(546)
        tf.reset_default_graph()
        self.image_sizes = np.array([11, 14], np.int32)
        self.num_objects = 5
        self.object_fnames = "sample_fname"
        self.object_classes = np.random.randint(0, 11, size=[self.num_objects]
                                                ).astype(np.int64)
        self.object_instance_ids = np.arange(
            1, self.num_objects + 1).reshape([self.num_objects]).astype(
            np.int64)
        self.object_boxes = np.random.random(size=[self.num_objects, 4]
                                             ).astype(np.float32)
        self.object_scores = np.random.random(size=[self.num_objects]
                                              ).astype(np.float32)

    @parameterized.parameters(
        {"with_fname": True, "with_classes": True, "with_instance_ids": True,
         "with_scores": True},
        {"with_fname": True, "with_classes": True, "with_instance_ids": True,
         "with_scores": True, "normalize_boxes": True},
        {"with_fname": True, "with_classes": True, "with_instance_ids": True,
         "with_scores": True, "convert_boxes_to_image_frame": True},
        {"with_fname": False, "with_classes": False, "with_instance_ids": True},
        {"with_fname": False, "with_classes": True, "with_instance_ids": False})
    def test_parse_tfrecord_example(self,
                                    with_fname=True, with_classes=True,
                                    with_instance_ids=True,
                                    with_scores=False,
                                    convert_boxes_to_image_frame=False,
                                    normalize_boxes=False):
        dataset = ObjectDetectionReaderTfRecords(
            normalize_boxes=normalize_boxes,
            convert_boxes_to_image_frame=convert_boxes_to_image_frame
        ).build()

        example = self._get_tf_example(
            with_fname=with_fname,
            with_classes=with_classes,
            with_instance_ids=with_instance_ids,
            with_scores=with_scores,
            with_image_sizes=normalize_boxes or convert_boxes_to_image_frame)
        result = dataset.parse_tfrecord_example(example)
        output_types_must = {
            ObjectDataFields.object_boxes: tf.float32,
            ObjectDataFields.object_classes: tf.int64,
            ObjectDataFields.object_instance_ids: tf.int64,
            ObjectDataFields.object_scores: tf.float32,
            ObjectDataFields.num_objects: tf.int64,
            ObjectDataFields.object_fnames: tf.string,
        }
        output_shapes_must = {
            ObjectDataFields.object_boxes: [None, 4],
            ObjectDataFields.object_classes: [None],
            ObjectDataFields.object_instance_ids: [None],
            ObjectDataFields.object_scores: [None],
            ObjectDataFields.num_objects: [],
            ObjectDataFields.object_fnames: [],
        }
        data_parsed_output_shapes = {
            k: v.get_shape().as_list()
            for k, v in result.items()}
        data_parsed_output_types = {
            k: v.dtype
            for k, v in result.items()}
        self.assertDictEqual(output_types_must,
                             data_parsed_output_types)
        self.assertDictEqual(output_shapes_must,
                             data_parsed_output_shapes)
        result_eval = self.evaluate(result)
        result_must = self._get_result_must(
            with_fname=with_fname,
            with_classes=with_classes,
            with_instance_ids=with_instance_ids,
            with_scores=with_scores,
            normalize_boxes=normalize_boxes,
            convert_boxes_to_image_frame=convert_boxes_to_image_frame,
        )
        for each_key in result_must:
            if each_key == "object_fnames":
                self.assertEqual(result_must[each_key],
                                 result_eval[each_key].decode())
            else:
                self.assertAllClose(result_must[each_key],
                                    result_eval[each_key])

    def _get_tf_example(self, with_fname, with_classes, with_instance_ids,
                        with_scores, with_image_sizes=False):
        feature_data = {ObjectDataFields.object_boxes: self.object_boxes}
        if with_fname:
            feature_data[ObjectDataFields.object_fnames] = self.object_fnames
        if with_classes:
            feature_data[ObjectDataFields.object_classes] = self.object_classes
        if with_instance_ids:
            feature_data[ObjectDataFields.object_instance_ids] = (
                self.object_instance_ids)
        if with_scores:
            feature_data[ObjectDataFields.object_scores] = (
                self.object_scores)
        if with_image_sizes:
            feature_data[ImageDataFields.image_sizes] = self.image_sizes
        feature = tf_data_utils.nested_to_tfrecords_feature(
            feature_data)
        example = tf.train.Example(
            features=tf.train.Features(feature=feature))
        serialized_example = example.SerializePartialToString()
        return serialized_example

    def _get_result_must(self,
                         with_fname, with_classes, with_instance_ids,
                         with_scores,
                         convert_boxes_to_image_frame,
                         normalize_boxes):
        object_boxes = self.object_boxes
        if convert_boxes_to_image_frame:
            object_boxes = object_detection_utils.local_to_image_coordinates_np(
                object_boxes, self.image_sizes)
        if normalize_boxes:
            object_boxes = object_detection_utils.normalize_bbox_np(
                object_boxes, self.image_sizes)
        result = {ObjectDataFields.object_boxes: object_boxes,
                  ObjectDataFields.num_objects: np.asarray(
                      self.num_objects, np.int32)}

        if with_fname:
            result[ObjectDataFields.object_fnames] = self.object_fnames
        else:
            result[ObjectDataFields.object_fnames] = "no_file_name"
        if with_classes:
            result[ObjectDataFields.object_classes] = self.object_classes
        else:
            result[ObjectDataFields.object_classes] = np.zeros(
                [self.num_objects], np.int64)
        if with_instance_ids:
            result[ObjectDataFields.object_instance_ids] = (
                self.object_instance_ids)
        else:
            result[ObjectDataFields.object_instance_ids] = np.zeros(
                [self.num_objects], np.int64) - 1
        if with_scores:
            result[ObjectDataFields.object_scores] = (
                self.object_scores)
        else:
            result[ObjectDataFields.object_scores] = np.zeros(
                [self.num_objects], np.float32) - 1

        return result


class TestKeypointsReaderTfRecords(tf.test.TestCase,
                                   parameterized.TestCase):

    def setUp(self):
        np.random.seed(546)
        tf.reset_default_graph()
        self.num_objects = 5
        self.num_keypoints = 7
        self.object_keypoints = np.random.random(
            [self.num_objects, self.num_keypoints, 2]).astype(np.float32)
        self.object_keypoints_visibilities = np.random.randint(
            0, 3, size=[self.num_objects, self.num_keypoints]).astype(np.int32)

    @parameterized.parameters({"with_visibilities": True},
                              {"with_visibilities": False})
    def test_parse_tfrecord_example(self, with_visibilities):
        dataset = KeypointsReaderTfRecords(
            num_keypoints=self.num_keypoints
        ).build()

        example = self._get_tf_example(
            with_visibilities=with_visibilities)
        result = dataset.parse_tfrecord_example(example)
        output_types_must = {
            ObjectDataFields.object_keypoints: tf.float32,
            ObjectDataFields.object_keypoints_visibilities: tf.int32,
        }
        output_shapes_must = {
            ObjectDataFields.object_keypoints: [None, self.num_keypoints, 2],
            ObjectDataFields.object_keypoints_visibilities:
                [None, self.num_keypoints],
        }
        data_parsed_output_shapes = {
            k: v.get_shape().as_list()
            for k, v in result.items()}
        data_parsed_output_types = {
            k: v.dtype
            for k, v in result.items()}
        self.assertDictEqual(output_types_must,
                             data_parsed_output_types)
        self.assertDictEqual(output_shapes_must,
                             data_parsed_output_shapes)
        result_eval = self.evaluate(result)
        result_must = self._get_result_must(
            with_visibilities=with_visibilities,
        )
        self.assertAllClose(result_must,
                            result_eval)

    def _get_tf_example(self, with_visibilities):
        feature_data = {
            ObjectDataFields.object_keypoints: self.object_keypoints}

        if with_visibilities:
            feature_data[ObjectDataFields.object_keypoints_visibilities] = (
                self.object_keypoints_visibilities)

        feature = tf_data_utils.nested_to_tfrecords_feature(
            feature_data)
        example = tf.train.Example(
            features=tf.train.Features(feature=feature))
        serialized_example = example.SerializePartialToString()
        return serialized_example

    def _get_result_must(self, with_visibilities):
        result = {ObjectDataFields.object_keypoints: self.object_keypoints}
        if with_visibilities:
            result[ObjectDataFields.object_keypoints_visibilities] = (
                self.object_keypoints_visibilities)
        else:
            result[ObjectDataFields.object_keypoints_visibilities] = (
                    np.ones_like(self.object_keypoints_visibilities) * 2)
        return result


class TestObjectClassSelectorTf(tf.test.TestCase):
    def setUp(self):
        self.classes_to_select = [1, 5, 7]

        self.object_classes_np = np.array([
            1, 2, 2, 5, 1, 3,
        ], np.int32)
        self.object_boxes_np = np.array([
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [13, 14, 15, 16],
            [17, 18, 19, 20],
            [21, 22, 23, 24],
        ], np.float32)
        self.instance_ids_np = np.array([
            1, 2, 3, 4, 5, 6,
        ], np.int32)
        self.num_objects_np = 6

        self.object_boxes = tf.placeholder(tf.float32, [None, 4])
        self.object_classes = tf.placeholder(tf.int32, [None])
        self.instance_ids = tf.placeholder(tf.int32, [None])
        self.num_objects = tf.placeholder(tf.int32, [])

        mask_selected = np.array([0, 3, 4])
        self.result_must = {
            "object_classes": self.object_classes_np[mask_selected],
            "object_boxes": self.object_boxes_np[mask_selected],
            "instance_ids": self.instance_ids_np[mask_selected],
            "num_objects": 3,
        }

    def test_process(self):
        processor = ObjectClassSelectorTf(
            classes_to_select=self.classes_to_select,
        ).build()
        result = processor.process(
            object_classes=self.object_classes,
            object_boxes=self.object_boxes,
            instance_ids=self.instance_ids,
            num_objects=self.num_objects,
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


class TestBoxesAdjusterByKeypoints(tf.test.TestCase, parameterized.TestCase):

    def setUp(self):
        self.keypoints_np = np.array([
            [[0, 0],
             [1, 20],
             [5, 4]],
            [[0, 0],
             [0, 0],
             [0, 0]],
            [[0, 0],
             [2, 3],
             [1, 4]],
            [[0, 0],
             [0, 0],
             [0, 0]],
        ]).astype(np.float32)
        self.boxes_np = np.array([
            [0, 5, 10, 20],
            [0, 1, 1, 5],
            [1, 2, 2, 4],
            [0, 0, 0, 0]
        ])

        self.keypoints = tf.placeholder(tf.float32, [None, None, 2])
        self.boxes = tf.placeholder(tf.float32, [None, 4])
        self.boxes_from_keypoints_must = np.array([
            [1, 4, 5, 20],
            [0, 0, 0, 0],
            [1, 3, 2, 4],
            [0, 0, 0, 0],
        ])
        self.boxes_adjusted_must = np.array([
            [0, 4, 10, 20],
            [0, 1, 1, 5],
            [1, 2, 2, 4],
            [0, 0, 0, 0],
        ])

    @parameterized.parameters({"provide_boxes": True},
                              {"provide_boxes": False})
    def test_process(self, provide_boxes):
        processor = BoxAdjusterByKeypoints().build()
        result = processor.process(
            object_keypoints=self.keypoints,
            object_boxes=self.boxes if provide_boxes else None)
        feed_dict = {self.keypoints: self.keypoints_np}
        if provide_boxes:
            feed_dict[self.boxes] = self.boxes_np

        with self.test_session() as sess:
            result_eval = sess.run(result, feed_dict)

        boxes_must = (self.boxes_adjusted_must if provide_boxes
                      else self.boxes_from_keypoints_must)
        self.assertAllClose({"object_boxes": boxes_must},
                            result_eval)
