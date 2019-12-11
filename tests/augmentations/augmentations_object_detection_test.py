# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

import numpy as np
import tensorflow as tf

from ncgenes7.augmentations.object_detection import ObjectsFlipUpDown
from ncgenes7.augmentations.object_detection import ObjectsHorizontalFlip
from ncgenes7.augmentations.object_detection import ObjectsRandomCrop
from ncgenes7.augmentations.object_detection import ObjectsRandomCutout
from ncgenes7.augmentations.object_detection import ObjectsRandomRotation


class TestObjectsHorizontalFlip(tf.test.TestCase):
    def test_process(self):
        np.random.seed(6546)
        object_boxes = np.array([
            [0.1, 0.2, 0.3, 0.4],
            [0.0, 0.2, 0.31, 0.41],
            [0, 0, 0, 0],
            [0.5, 0.6, 0.7, 0.82],
            [0.5, 0.2, 0.6, 0.3],
            [0.1, 0.2, 0.31, 0.41],
            [0.1, 0.0, 0.6, 0.2],
            [0, 0, 0, 0],
        ], np.float32)
        object_boxes_must = np.array([
            [0.1, 0.6, 0.3, 0.8],
            [0.0, 0.59, 0.31, 0.8],
            [0, 0, 0, 0],
            [0.5, 0.18, 0.7, 0.4],
            [0.5, 0.7, 0.6, 0.8],
            [0.1, 0.59, 0.31, 0.8],
            [0.1, 0.8, 0.6, 1.0],
            [0, 0, 0, 0],
        ], np.float32)

        object_keypoints = np.array([
            [[0, 0],
             [0.1, 0.0]],
            [[0.11, 0.25],
             [0.2, 0.4]],
            [[0.0, 0.0],
             [0.0, 0.0]],
            [[0.7, 0.82],
             [0.61, 0.75]],
            [[0.5, 0.25],
             [0.0, 0.0]],
            [[0.11, 0.3],
             [0.3, 0.4]],
            [[0.55, 0.1],
             [0.3, 0.2]],
            [[0.0, 0.0],
             [0.0, 0.0]],
        ], np.float32)
        object_keypoints_must = np.array([
            [[0, 0],
             [0.1, 1.0]],
            [[0.11, 0.75],
             [0.2, 0.6]],
            [[0.0, 0.0],
             [0.0, 0.0]],
            [[0.7, 0.18],
             [0.61, 0.25]],
            [[0.5, 0.75],
             [0.0, 0.0]],
            [[0.11, 0.7],
             [0.3, 0.6]],
            [[0.55, 0.9],
             [0.3, 0.8]],
            [[0.0, 0.0],
             [0.0, 0.0]],
        ], np.float32)

        object_classes = np.array([1, 0, 3, 1, 2, 1, 1, 0], np.int32)
        object_other_data = np.random.rand(8, 10)

        augmentation = ObjectsHorizontalFlip(
            augmentation_probability=1.0).build()
        result = augmentation.process(
            object_boxes=tf.constant(object_boxes),
            object_keypoints=tf.constant(object_keypoints),
            object_classes=tf.constant(object_classes),
            object_other_data=tf.constant(object_other_data))
        result_eval = self.evaluate(result)
        result_must = {
            "augment": True,
            "object_boxes": object_boxes_must,
            "object_keypoints": object_keypoints_must,
            "object_classes": object_classes,
            "object_other_data": object_other_data,
        }
        self.assertAllClose(result_must,
                            result_eval)


class TestObjectsFlipUpDown(tf.test.TestCase):
    def test_process(self):
        np.random.seed(6546)
        object_boxes = np.array([
            [0.1, 0.2, 0.3, 0.4],
            [0.0, 0.2, 0.31, 0.41],
            [0, 0, 0, 0],
            [0.5, 0.6, 0.7, 0.82],
            [0.5, 0.2, 0.6, 0.3],
            [0.1, 0.2, 0.31, 0.41],
            [0.1, 0.0, 0.6, 0.2],
            [0, 0, 0, 0],
        ], np.float32)
        object_boxes_must = np.array([
            [0.7, 0.2, 0.9, 0.4],
            [0.69, 0.2, 1.0, 0.41],
            [0, 0, 0, 0],
            [0.3, 0.6, 0.5, 0.82],
            [0.4, 0.2, 0.5, 0.3],
            [0.69, 0.2, 0.9, 0.41],
            [0.4, 0.0, 0.9, 0.2],
            [0, 0, 0, 0],
        ], np.float32)

        object_keypoints = np.array([
            [[0, 0],
             [0.1, 0.0]],
            [[0.11, 0.25],
             [0.2, 0.4]],
            [[0.0, 0.0],
             [0.0, 0.0]],
            [[0.7, 0.82],
             [0.61, 0.75]],
            [[0.5, 0.25],
             [0.0, 0.0]],
            [[0.11, 0.3],
             [0.3, 0.4]],
            [[0.55, 0.1],
             [0.3, 0.2]],
            [[0.0, 0.0],
             [0.0, 0.0]],
        ], np.float32)
        object_keypoints_must = np.array([
            [[0, 0],
             [0.9, 0.0]],
            [[0.89, 0.25],
             [0.8, 0.4]],
            [[0.0, 0.0],
             [0.0, 0.0]],
            [[0.3, 0.82],
             [0.39, 0.75]],
            [[0.5, 0.25],
             [0.0, 0.0]],
            [[0.89, 0.3],
             [0.7, 0.4]],
            [[0.45, 0.1],
             [0.7, 0.2]],
            [[0.0, 0.0],
             [0.0, 0.0]],
        ], np.float32)

        object_classes = np.array([1, 0, 3, 1, 2, 1, 1, 0], np.int32)
        object_other_data = np.random.rand(2, 4, 10)

        augmentation = ObjectsFlipUpDown(
            augmentation_probability=1.0).build()
        result = augmentation.process(
            object_boxes=tf.constant(object_boxes),
            object_keypoints=tf.constant(object_keypoints),
            object_classes=tf.constant(object_classes),
            object_other_data=tf.constant(object_other_data))
        result_eval = self.evaluate(result)
        result_must = {
            "augment": True,
            "object_boxes": object_boxes_must,
            "object_keypoints": object_keypoints_must,
            "object_classes": object_classes,
            "object_other_data": object_other_data,
        }
        self.assertAllClose(result_must,
                            result_eval)


class TestObjectsRandomRotation(tf.test.TestCase):

    def test_process(self):
        rotation_angle = 30.0 * np.pi / 180
        object_boxes = np.array([
            [0.4, 0.3, 0.7, 0.5],
            [0.7, 0.6, 0.9, 0.8],
            [0.0, 0.6, 0.2, 0.9],
            [0.9, 0.9, 0.95, 0.95],
            [0.0, 0.3, 0.5, 0.7],
            [0, 0, 0, 0]
        ], np.float32)
        object_boxes_must = np.array([
            [0.3134, 0.2268, 0.6732, 0.55],
            [0.723205, 0.386603, 0.99641, 0.659808],
            [0.117, 0.7366, 0.4402, 1.0],
            [0.0, 0.326795, 0.6, 0.923205],
            [0, 0, 0, 0],
            [0, 0, 0, 0]
        ])

        object_keypoints = np.array([
            [[0.5, 0.45],
             [0.0, 0.0]],
            [[0.8, 0.6],
             [0.83, 0.75]],
            [[0.1, 0.66],
             [0.02, 0.89]],
            [[0.93, 0.93],
             [0.0, 0.0]],
            [[0.0, 0.0],
             [0.3, 0.6]],
            [[0.0, 0.0],
             [0.0, 0.0]],
        ], np.float32)
        object_keypoints_must = np.array([
            [[0.475, 0.4567],
             [0.0, 0.0]],
            [[0.8098, 0.4366],
             [0.9108, 0.5515]],
            [[0.2336, 0.8386],
             [0.0, 0.0]],
            [[0.0, 0.0],
             [0.3768, 0.6866]],
            [[0.0, 0.0],
             [0.0, 0.0]],
            [[0.0, 0.0],
             [0.0, 0.0]],
        ], np.float32)

        object_classes = np.array([1, 3, 1, 2, 1, 0], np.int32)
        object_classes_must = np.array([1, 3, 1, 1, 0, 0], np.int32)
        object_other_data = np.random.rand(6, 10) * 100
        object_other_data_must = np.concatenate(
            [object_other_data[:3, :],
             object_other_data[4:5, :],
             np.zeros_like(object_other_data[:2, :])], 0)

        augmentation = ObjectsRandomRotation(
            max_angle=0,
            augmentation_probability=1.0).build()
        result = augmentation.process(
            rotation_angle=tf.constant(rotation_angle),
            object_boxes=tf.constant(object_boxes),
            object_keypoints=tf.constant(object_keypoints),
            object_classes=tf.constant(object_classes),
            object_other_data=tf.constant(object_other_data))
        result_eval = self.evaluate(result)
        result_must = {
            "augment": True,
            "rotation_angle": rotation_angle,
            "object_boxes": object_boxes_must,
            "object_keypoints": object_keypoints_must,
            "object_classes": object_classes_must,
            "object_other_data": object_other_data_must,
        }
        self.assertAllClose(result_must,
                            result_eval, atol=4e-5)


class TestObjectsRandomCrop(tf.test.TestCase):

    def test_process(self):
        crop_offset = np.array([0.1, 0.2], np.float32)
        crop_scale = 0.7

        object_boxes = np.array([
            [0.4, 0.3, 0.7, 0.5],
            [0.7, 0.6, 0.9, 0.8],
            [0.0, 0.6, 0.2, 0.9],
            [0.9, 0.9, 0.95, 0.95],
            [0.0, 0.15, 0.4, 0.65],
            [0, 0, 0, 0]
        ], np.float32)
        object_boxes_must = np.array([
            [0.4286, 0.1429, 0.8572, 0.4286],
            [0.8571, 0.5714, 1.0, 0.8571],
            [0.0, 0.5714, 0.1428, 1.0],
            [0.0, 0.0, 0.4285, 0.6429],
            [0, 0, 0, 0],
            [0, 0, 0, 0]
        ])

        object_keypoints = np.array([
            [[0.5, 0.45],
             [0.0, 0.0]],
            [[0.75, 0.6],
             [0.83, 0.75]],
            [[0.1, 0.66],
             [0.02, 0.89]],
            [[0.93, 0.93],
             [0.0, 0.0]],
            [[0.0, 0.0],
             [0.3, 0.45]],
            [[0.0, 0.0],
             [0.0, 0.0]],
        ], np.float32)
        object_keypoints_must = np.array([
            [[0.5714, 0.3571],
             [0.0, 0.0]],
            [[0.9286, 0.5714],
             [0.0, 0.0]],
            [[0.0, 0.0],
             [0.0, 0.0]],
            [[0.0, 0.0],
             [0.2857, 0.3571]],
            [[0.0, 0.0],
             [0.0, 0.0]],
            [[0.0, 0.0],
             [0.0, 0.0]],
        ], np.float32)

        object_classes = np.array([1, 3, 1, 2, 1, 0], np.int32)
        object_classes_must = np.array([1, 3, 1, 1, 0, 0], np.int32)
        object_other_data = np.random.rand(6, 10) * 100
        object_other_data_must = np.concatenate(
            [object_other_data[:3, :],
             object_other_data[4:5, :],
             np.zeros_like(object_other_data[:2, :])], 0)

        augmentation = ObjectsRandomCrop(
            min_scale=0,
            max_scale=0,
            augmentation_probability=1.0).build()
        result = augmentation.process(
            crop_offset=tf.constant(crop_offset),
            crop_scale=tf.constant(crop_scale),
            object_boxes=tf.constant(object_boxes),
            object_keypoints=tf.constant(object_keypoints),
            object_classes=tf.constant(object_classes),
            object_other_data=tf.constant(object_other_data))
        result_eval = self.evaluate(result)
        result_must = {
            "augment": True,
            "crop_offset": crop_offset,
            "crop_scale": crop_scale,
            "object_boxes": object_boxes_must,
            "object_keypoints": object_keypoints_must,
            "object_classes": object_classes_must,
            "object_other_data": object_other_data_must,
        }
        self.assertAllClose(result_must,
                            result_eval, atol=1e-4)


class TestObjectsRandomCutout(tf.test.TestCase):

    def test_process(self):
        cut_lengths = np.array([0.4, 0.35], np.float32)
        cut_offset = np.array([0.2, 0.2], np.float32)
        max_occlusion = 0.5

        object_boxes = np.array([
            [0.4, 0.3, 0.7, 0.5],
            [0.7, 0.6, 0.9, 0.8],
            [0.0, 0.6, 0.2, 0.9],
            [0.9, 0.9, 0.95, 0.95],
            [0.0, 0.15, 0.4, 0.65],
            [0, 0, 0, 0]
        ], np.float32)
        object_boxes_must = np.array([
            [0.7, 0.6, 0.9, 0.8],
            [0.0, 0.6, 0.2, 0.9],
            [0.9, 0.9, 0.95, 0.95],
            [0.0, 0.15, 0.4, 0.65],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ])

        object_keypoints = np.array([
            [[0.5, 0.45],
             [0.0, 0.0]],
            [[0.75, 0.6],
             [0.83, 0.75]],
            [[0.1, 0.66],
             [0.02, 0.89]],
            [[0.93, 0.93],
             [0.0, 0.0]],
            [[0.0, 0.0],
             [0.3, 0.45]],
            [[0.0, 0.0],
             [0.0, 0.0]],
        ], np.float32)
        object_keypoints_must = np.array([
            [[0.75, 0.6],
             [0.83, 0.75]],
            [[0.1, 0.66],
             [0.02, 0.89]],
            [[0.93, 0.93],
             [0.0, 0.0]],
            [[0.0, 0.0],
             [0.0, 0.0]],
            [[0.0, 0.0],
             [0.0, 0.0]],
            [[0.0, 0.0],
             [0.0, 0.0]],
        ], np.float32)

        object_classes = np.array([1, 3, 1, 2, 1, 0], np.int32)
        object_classes_must = np.array([3, 1, 2, 1, 0, 0], np.int32)
        object_other_data = np.random.rand(6, 10) * 100
        object_other_data_must = np.concatenate(
            [object_other_data[1:5, :],
             np.zeros_like(object_other_data[:2, :])], 0)

        augmentation = ObjectsRandomCutout(
            min_cut_length=0.0,
            max_cut_length=0.0,
            max_occlusion=max_occlusion,
            augmentation_probability=1.0).build()
        result = augmentation.process(
            cut_lengths=tf.constant(cut_lengths),
            cut_offset=tf.constant(cut_offset),
            object_boxes=tf.constant(object_boxes),
            object_keypoints=tf.constant(object_keypoints),
            object_classes=tf.constant(object_classes),
            object_other_data=tf.constant(object_other_data))
        result_eval = self.evaluate(result)
        result_must = {
            "augment": True,
            "cut_lengths": cut_lengths,
            "cut_offset": cut_offset,
            "object_boxes": object_boxes_must,
            "object_keypoints": object_keypoints_must,
            "object_classes": object_classes_must,
            "object_other_data": object_other_data_must,
        }
        self.assertAllClose(result_must["object_keypoints"],
                            result_eval["object_keypoints"])
        self.assertAllClose(result_must,
                            result_eval)
