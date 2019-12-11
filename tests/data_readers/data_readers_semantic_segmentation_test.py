# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

import io
import json
import os

from absl.testing import parameterized
from nucleus7.utils import tf_data_utils
import numpy as np
import skimage.io
import skimage.transform
import tensorflow as tf

from ncgenes7.data_fields.images import ImageDataFields
from ncgenes7.data_fields.semantic_segmentation import SegmentationDataFields
from ncgenes7.data_readers.semantic_segmentation import EdgesReaderTfRecords
from ncgenes7.data_readers.semantic_segmentation import (
    SemanticSegmentationReader)
from ncgenes7.data_readers.semantic_segmentation import (
    SemanticSegmentationReaderTfRecords)


class TestSemanticSegmentationDataFeeder(parameterized.TestCase,
                                         tf.test.TestCase):
    def setUp(self):
        self.original_image_size = [20, 30]
        possible_rgb_values = [
            (10, 20, 30), (50, 2, 7), (0, 0, 0), (30, 15, 16),
            (50, 90, 120), (4, 8, 16)]
        segmentation_indices = np.random.choice(
            6, size=self.original_image_size,
            p=[0.05, 0.1, 0.4, 0.2, 0.1, 0.15])

        self.segmentation_rgb = np.array(
            possible_rgb_values)[segmentation_indices].astype(np.uint8)

        self.segmentation_classes = np.expand_dims(
            np.zeros_like(segmentation_indices), -1)
        self.segmentation_classes[segmentation_indices == 0] = 1
        self.segmentation_classes[segmentation_indices == 1] = 2
        self.segmentation_classes[segmentation_indices == 3] = 3
        self.segmentation_classes[segmentation_indices == 4] = 3

        self.input_dir = self.get_temp_dir()
        self.save_dir = os.path.join(self.input_dir, "save_dir")
        self.image_fname = os.path.join(
            self.input_dir, "inputs", "images", "002.png")
        self.segmentation_fname = os.path.join(
            self.input_dir, "inputs", "segmentation", "002.png")

        self.rgb_class_ids_mapping = {
            "class_ids": [
                1,
                2,
                3,
                3,
            ],
            "rgb": [
                (10, 20, 30),
                (50, 2, 7),
                (30, 15, 16),
                (50, 90, 120),
            ]
        }

        os.mkdir(os.path.join(self.input_dir, "annotations"))
        os.mkdir(os.path.join(self.input_dir, "inputs"))
        os.mkdir(os.path.join(self.input_dir, "inputs", "segmentation"))
        self.file_name_rgb_class_ids_mapping = os.path.join(
            self.input_dir, "annotations", "rgb_class_mapping.json")
        with open(self.file_name_rgb_class_ids_mapping, "w") as f:
            json.dump(self.rgb_class_ids_mapping, f, sort_keys=True, indent=2)

        skimage.io.imsave(self.segmentation_fname, self.segmentation_rgb)

    @parameterized.parameters(
        {"extract_class_ids": True, "extract_edges": True},
        {"extract_class_ids": True, "extract_edges": True,
         "image_size": [40, 60]},
        {"extract_class_ids": True, "extract_edges": False},
        {"extract_class_ids": False, "extract_edges": True})
    def test_read_element_from_file_names(self, extract_class_ids,
                                          extract_edges,
                                          image_size=None):
        data_feeder = SemanticSegmentationReader(
            image_size=image_size,
            rgb_class_mapping=self.file_name_rgb_class_ids_mapping,
            extract_class_ids=extract_class_ids,
            extract_edges=extract_edges,
        ).build()

        result = data_feeder.read(labels=self.segmentation_fname)
        if image_size is not None:
            segmentation_classes_must = (skimage.transform.resize(
                self.segmentation_classes.astype(np.uint8), image_size, 0,
                mode='constant') * 255).astype(np.uint8)
        else:
            segmentation_classes_must = self.segmentation_classes

        result_keys_must = set()
        if extract_class_ids:
            result_keys_must.add("segmentation_classes")
        if extract_edges:
            result_keys_must.add("segmentation_edges")
        self.assertSetEqual(result_keys_must,
                            set(result))

        if extract_class_ids:
            self.assertAllClose(segmentation_classes_must,
                                result["segmentation_classes"])
        if extract_edges:
            shape_must = (image_size if image_size else self.original_image_size
                          ) + [1]
            self.assertListEqual(shape_must,
                                 list(result["segmentation_edges"].shape))


class TestSemanticSegmentationReaderTfRecords(tf.test.TestCase,
                                              parameterized.TestCase):

    def setUp(self):
        self.images_fnames = "sample_fname"
        self.original_image_size = [20, 30]

    @parameterized.parameters(
        {"image_encoding": None, "image_size": None},
        {"image_encoding": None, "image_size": [30, 45]},
        {"image_encoding": None, "image_size": [30, 45],
         "with_fname": False},
        {"image_encoding": "PNG", "image_size": None},
        {"image_encoding": "PNG", "image_size": [30, 45]},
        {"image_encoding": "PNG", "image_size": [30, 45],
         "with_fname": False},
    )
    def test_parse_tfrecord_example(self, image_encoding=None, image_size=None,
                                    with_fname=True):
        image, serialized_example = _get_image_example(
            self.original_image_size, image_encoding, with_fname,
            self.images_fnames)

        reader = SemanticSegmentationReaderTfRecords(
            image_encoding=image_encoding,
            image_size=image_size,
        ).build()
        result = reader.parse_tfrecord_example(serialized_example)

        self.assertSetEqual(
            {SegmentationDataFields.segmentation_classes,
             SegmentationDataFields.segmentation_classes_fnames},
            set(result))
        if image_encoding:
            tensor_image_shape_must = (image_size if image_size
                                       else [None, None]
                                       ) + [1]
        else:
            tensor_image_shape_must = (image_size if image_size
                                       else self.original_image_size
                                       ) + [1]

        result_image = result[SegmentationDataFields.segmentation_classes]
        self.assertListEqual(tensor_image_shape_must,
                             result_image.shape.as_list())
        self.assertEqual(result_image.dtype,
                         tf.uint8)
        self.assertEqual(
            result[SegmentationDataFields.segmentation_classes_fnames].dtype,
            tf.string)

        result_eval = self.evaluate(result)
        fname_must = self.images_fnames if with_fname else "no_file_name"

        image_must = image
        if image_size:
            # need to use tf resize method since skimage or open cv resize
            # has slightly different interpolation
            image_resized = tf.image.resize_images(
                image_must, image_size, method=1, align_corners=True)
            image_resized = tf.cast(image_resized, tf.uint8)
            image_must = self.evaluate(image_resized)

        self.assertAllClose(
            image_must,
            result_eval[SegmentationDataFields.segmentation_classes])
        self.assertEqual(
            fname_must,
            result_eval[
                SegmentationDataFields.segmentation_classes_fnames].decode())


class TestEdgesReaderTfRecords(tf.test.TestCase,
                               parameterized.TestCase):

    def setUp(self):
        self.images_fnames = "sample_fname"
        self.original_image_size = [20, 30]

    @parameterized.parameters(
        {"image_encoding": None, "image_size": None},
        {"image_encoding": None, "image_size": [30, 45]},
        {"image_encoding": None, "image_size": [30, 45],
         "with_fname": False},
        {"image_encoding": "PNG", "image_size": None},
        {"image_encoding": "PNG", "image_size": [30, 45]},
        {"image_encoding": "PNG", "image_size": [30, 45],
         "with_fname": False},
    )
    def test_parse_tfrecord_example(self, image_encoding=None, image_size=None,
                                    with_fname=True):
        image, serialized_example = _get_image_example(
            self.original_image_size, image_encoding, with_fname,
            self.images_fnames, is_edges=True)

        reader = EdgesReaderTfRecords(
            image_encoding=image_encoding,
            image_size=image_size,
        ).build()
        result = reader.parse_tfrecord_example(serialized_example)

        self.assertSetEqual(
            {SegmentationDataFields.segmentation_edges,
             SegmentationDataFields.segmentation_edges_fnames},
            set(result))
        if image_encoding:
            tensor_image_shape_must = (image_size if image_size
                                       else [None, None]
                                       ) + [1]
        else:
            tensor_image_shape_must = (image_size if image_size
                                       else self.original_image_size
                                       ) + [1]

        result_image = result[SegmentationDataFields.segmentation_edges]
        self.assertListEqual(tensor_image_shape_must,
                             result_image.shape.as_list())
        self.assertEqual(tf.uint8,
                         result_image.dtype)
        self.assertEqual(
            tf.string,
            result[SegmentationDataFields.segmentation_edges_fnames].dtype)

        result_eval = self.evaluate(result)
        fname_must = self.images_fnames if with_fname else "no_file_name"

        image_must = image
        if image_size:
            # need to use tf resize method since skimage or open cv resize
            # has slightly different interpolation
            image_resized = tf.image.resize_images(
                image_must, image_size, method=1, align_corners=True)
            image_resized = tf.cast(image_resized, tf.uint8)
            image_must = self.evaluate(image_resized)

        self.assertAllClose(
            image_must,
            result_eval[SegmentationDataFields.segmentation_edges])
        self.assertEqual(
            fname_must,
            result_eval[
                SegmentationDataFields.segmentation_edges_fnames].decode())


def _get_image_example(original_image_size, image_encoding, with_fname,
                       images_fnames, is_edges=False):
    image = np.random.randint(
        0, 255, size=[*original_image_size, 1]
    ).astype(np.uint8)

    image_key = (
        SegmentationDataFields.segmentation_classes if not is_edges
        else SegmentationDataFields.segmentation_edges)
    fname_key = (
        SegmentationDataFields.segmentation_classes_fnames if not is_edges
        else SegmentationDataFields.segmentation_edges_fnames)

    feature_data = {}
    if with_fname:
        feature_data[fname_key] = images_fnames
    if not image_encoding:
        feature_data[image_key] = image
        image_size = np.asarray(image.shape[:2], np.int32)
        feature_data[ImageDataFields.image_sizes] = image_size
    else:
        with io.BytesIO() as bytes_io:
            image_to_save = np.squeeze(image, -1)
            skimage.io.imsave(bytes_io, image_to_save,
                              format_str=image_encoding)
            image_png = bytes_io.getvalue()
        feature_data[image_key + "_PNG"] = image_png

    feature = tf_data_utils.nested_to_tfrecords_feature(
        feature_data)
    example = tf.train.Example(
        features=tf.train.Features(feature=feature))
    serialized_example = example.SerializePartialToString()
    return image, serialized_example
