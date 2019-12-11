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
import numpy as np
import skimage.io
import skimage.transform
import tensorflow as tf

from ncgenes7.data_readers.coco import CocoObjectsReader
from ncgenes7.data_readers.coco import CocoPersonKeypointsReader
from ncgenes7.data_readers.coco import CocoSemanticSegmentationReader


class TestCocoObjectsReader(tf.test.TestCase, parameterized.TestCase):

    def setUp(self):
        np.random.seed(546)
        self.image_size = [20, 30]
        self.file_name_instance_annotations = {}

        self.instance_annotations = {
            "categories": [
                {'id': 1, 'name': 'person', 'supercategory': 'person'},
                {'id': 10, 'name': 'bicycle', 'supercategory': 'vehicle'},
                {'id': 11, 'name': 'car', 'supercategory': 'vehicle'},
                {'id': 16, 'name': 'motorcycle', 'supercategory': 'vehicle'},
                {'id': 17, 'name': 'traffic light', 'supercategory': 'outdoor'},
            ],
            "images": [
                {
                    "id": 2,
                    "file_name": "002.jpg",
                    "width": 30,
                    "height": 20,
                },
                {
                    "id": 144,
                    "file_name": "144.jpg",
                    "width": 40,
                    "height": 60,
                },
            ],
            "annotations": [
                {
                    "bbox": [10.5, 5.6, 12.0, 11.8],
                    "id": 5,
                    "category_id": 16,
                    "image_id": 2,
                },
                {
                    "bbox": [0.3, 2.2, 20.1, 6.4],
                    "id": 1,
                    "category_id": 10,
                    "image_id": 2,
                },
                {
                    "bbox": [1.5, 4.3, 6.0, 4.5],
                    "id": 3,
                    "category_id": 11,
                    "image_id": 144,
                },
            ]
        }

        self.image_fname = os.path.join(
            self.get_temp_dir(), "images", "002.jpg")

        os.mkdir(os.path.join(self.get_temp_dir(), "annotations"))
        self.file_name_instance_annotations = os.path.join(
            self.get_temp_dir(), "annotations", "instance_annotations.json")
        with open(self.file_name_instance_annotations, "w") as f:
            json.dump(self.instance_annotations, f, sort_keys=True, indent=2)
        self.objects_must_removed_unused = {
            "object_classes": np.array([2, 4]),
            "object_instance_ids": np.array([1, 5]),
            "object_boxes": np.array([
                [0.11, 0.01, 0.43, 0.68],
                [0.28, 0.35, 0.87, 0.75],
            ]),
        }
        self.objects_must = {
            "object_classes": np.array([10, 16]),
            "object_instance_ids": np.array([1, 5]),
            "object_boxes": np.array([
                [0.11, 0.01, 0.43, 0.68],
                [0.28, 0.35, 0.87, 0.75],
            ]),
        }

    @parameterized.parameters({"remove_unused_classes": True},
                              {"remove_unused_classes": False}, )
    def test_read(self, remove_unused_classes):
        reader = CocoObjectsReader(
            file_name_instance_annotations=self.file_name_instance_annotations,
            remove_unused_classes=remove_unused_classes,
        ).build()

        result = reader.read(images=self.image_fname)
        if remove_unused_classes:
            result_must = self.objects_must_removed_unused
        else:
            result_must = self.objects_must
        self.assertSetEqual(set(result_must.keys()),
                            set(result.keys()))
        for each_key in result:
            if isinstance(result_must[each_key], str):
                self.assertEqual(result_must[each_key],
                                 result[each_key])
            else:
                self.assertAllClose(result_must[each_key],
                                    result[each_key])


class TestCocoSemanticSegmentationReader(tf.test.TestCase,
                                         parameterized.TestCase):

    def setUp(self):
        np.random.seed(546)
        possible_panoptic = [(10, 20, 30), (50, 2, 7), (0, 0, 0), (30, 15, 16)]
        self.panoptic_indices = np.random.choice(
            4, size=(8, 10), p=[0.1, 0.2, 0.4, 0.3])
        self.panoptic = np.array(possible_panoptic)[
            self.panoptic_indices].astype(
            np.uint8)

        self.panoptic_annotations = {
            "categories": [
                {'id': 1, 'isthing': 1, 'name': 'person',
                 'supercategory': 'person'},
                {'id': 3, 'isthing': 1, 'name': 'bicycle',
                 'supercategory': 'vehicle'},
                {'id': 7, 'isthing': 1, 'name': 'car',
                 'supercategory': 'vehicle'},
            ],
            "images": [
                {
                    "id": 2,
                    "file_name": "002.jpg",
                    "width": 30,
                    "height": 20,
                },
                {
                    "id": 144,
                    "file_name": "144.jpg",
                    "width": 40,
                    "height": 60,
                },
            ],
            "annotations": [
                {
                    "file_name": "002.png",
                    "image_id": 2,
                    "segments_info": [
                        {
                            "id": 1971210,  # RGB: (10, 20, 30)
                            "category_id": 1,
                        },
                        {
                            "id": 459314,  # RGB: (50, 2, 7)
                            "category_id": 7,
                        },
                    ]

                },
                {
                    "file_name": "144.png",
                    "image_id": 144,
                    "segments_info": [
                        {
                            "id": 444756,
                            "category_id": 1,
                        },
                    ]
                }
            ]
        }

        self.panoptic_fname = os.path.join(
            self.get_temp_dir(), "panoptic", "002.png")
        os.mkdir(os.path.join(self.get_temp_dir(), "annotations"))
        os.mkdir(os.path.join(self.get_temp_dir(), "panoptic"))

        self.file_name_panoptic_annotations = os.path.join(
            self.get_temp_dir(), "annotations", "panoptic_annotations.json")
        with open(self.file_name_panoptic_annotations, "w") as f:
            json.dump(self.panoptic_annotations, f, sort_keys=True, indent=2)
        skimage.io.imsave(self.panoptic_fname, self.panoptic)

    @parameterized.parameters(
        {"image_size": None, "remove_unused_classes": True},
        {"image_size": None, "remove_unused_classes": False},
        {"image_size": [50, 60]})
    def test_read(self, image_size, remove_unused_classes=False):
        reader = CocoSemanticSegmentationReader(
            file_name_panoptic_annotations=self.file_name_panoptic_annotations,
            image_size=image_size,
            remove_unused_classes=remove_unused_classes,
        ).build()
        result = reader.read(panoptic=self.panoptic_fname)

        result_must = self._get_result(image_size, remove_unused_classes)
        self.assertSetEqual(set(result_must.keys()),
                            set(result.keys()))
        for each_key in result:
            if isinstance(result_must[each_key], str):
                self.assertEqual(result_must[each_key],
                                 result[each_key])
            else:
                self.assertAllClose(result_must[each_key],
                                    result[each_key])

    def _get_result(self, image_size, remove_unused_classes):
        panoptic_image_must = np.expand_dims(
            np.zeros_like(self.panoptic_indices), -1)
        panoptic_image_must[self.panoptic_indices == 0] = 1
        if remove_unused_classes:
            panoptic_image_must[self.panoptic_indices == 1] = 3
        else:
            panoptic_image_must[self.panoptic_indices == 1] = 7

        panoptic_image_must = panoptic_image_must.astype(np.uint8)
        if image_size:
            panoptic_image_must = (
                    skimage.transform.resize(
                        panoptic_image_must,
                        image_size, order=0) * 255).astype(np.uint8)

        panoptic_must = {"segmentation_classes": panoptic_image_must}
        return panoptic_must


class TestCocoKeypointsReader(tf.test.TestCase):

    def setUp(self):
        np.random.seed(546)
        self.image_size = [20, 30]
        self.file_name_person_annotations = {}

        keypoints_object1 = [0, 0, 0,
                             0, 0, 0,
                             0, 0, 0,
                             0, 0, 0,
                             0, 0, 0,
                             10, 20, 1,
                             6, 7, 2,
                             4, 5, 2,
                             20, 11, 2,
                             10, 30, 2,
                             1, 1, 2,
                             5, 20, 2,
                             19, 7, 2,
                             15, 6, 1,
                             7, 9, 2,
                             0, 0, 0,
                             4, 9, 2]
        keypoints_object2 = [7, 6, 0,
                             1, 4, 1,
                             5, 7, 2,
                             0, 0, 0,
                             9, 11, 1,
                             11, 23, 2,
                             17, 20, 2,
                             16, 54, 2,
                             3, 28, 2,
                             23, 19, 2,
                             44, 16, 2,
                             14, 7, 2,
                             2, 4, 2,
                             0, 0, 0,
                             2, 1, 1,
                             0, 0, 0,
                             5, 4, 2]

        self.instance_annotations = {
            "categories": [
                {'id': 1, 'name': 'person', 'supercategory': 'person'},
                {'id': 10, 'name': 'bicycle', 'supercategory': 'vehicle'},
                {'id': 11, 'name': 'car', 'supercategory': 'vehicle'},
                {'id': 16, 'name': 'motorcycle', 'supercategory': 'vehicle'},
                {'id': 17, 'name': 'traffic light', 'supercategory': 'outdoor'},
            ],
            "images": [
                {
                    "id": 2,
                    "file_name": "002.jpg",
                    "width": 30,
                    "height": 20,
                },
                {
                    "id": 144,
                    "file_name": "144.jpg",
                    "width": 40,
                    "height": 60,
                },
            ],
            "annotations": [
                {
                    "bbox": [10.5, 5.6, 12.0, 11.8],
                    "id": 5,
                    "category_id": 1,
                    "image_id": 2,
                    "keypoints": keypoints_object1,
                },
                {
                    "bbox": [0.3, 2.2, 20.1, 6.4],
                    "id": 1,
                    "category_id": 1,
                    "image_id": 2,
                    "keypoints": keypoints_object2,
                },
                {
                    "bbox": [1.5, 4.3, 6.0, 4.5],
                    "id": 3,
                    "category_id": 11,
                    "image_id": 144,
                    "keypoints": [0] * 17 * 3,
                },
            ]
        }

        self.image_fname = os.path.join(
            self.get_temp_dir(), "images", "002.jpg")

        os.mkdir(os.path.join(self.get_temp_dir(), "annotations"))
        self.file_name_person_annotations = os.path.join(
            self.get_temp_dir(), "annotations", "person_annotations.json")
        with open(self.file_name_person_annotations, "w") as f:
            json.dump(self.instance_annotations, f, sort_keys=True, indent=2)

        keypoints_object1_must, visibilities_object1_must = (
            self._get_keypoints_must(keypoints_object1))
        keypoints_object2_must, visibilities_object2_must = (
            self._get_keypoints_must(keypoints_object2))
        self.objects_must = {
            "object_classes": np.array([1, 1]),
            "object_instance_ids": np.array([1, 5]),
            "object_boxes": np.array([
                [0.11, 0.01, 0.43, 0.68],
                [0.28, 0.35, 0.87, 0.75],
            ]),
            "object_keypoints": np.stack(
                [keypoints_object2_must, keypoints_object1_must], 0),
            "object_keypoints_visibilities": np.stack(
                [visibilities_object2_must, visibilities_object1_must], 0),
        }

    def test_read(self):
        reader = CocoPersonKeypointsReader(
            file_name_person_keypoints_annotations=
            self.file_name_person_annotations,
        ).build()

        result = reader.read(images=self.image_fname)
        result_must = self.objects_must
        self.assertSetEqual(set(result_must.keys()),
                            set(result.keys()))
        for each_key in result:
            if isinstance(result_must[each_key], str):
                self.assertEqual(result_must[each_key],
                                 result[each_key])
            else:
                self.assertAllClose(result_must[each_key],
                                    result[each_key])

    def _get_keypoints_must(self, keypoints_raw):
        keypoints_reshaped = np.reshape(keypoints_raw, [17, 3])
        keypoints_must = np.stack(
            [keypoints_reshaped[:, 1] / self.image_size[0],
             keypoints_reshaped[:, 0] / self.image_size[1]],
            -1).astype(np.float32)
        visibilities_must = keypoints_reshaped[:, -1].astype(
            np.int32)
        return keypoints_must, visibilities_must
