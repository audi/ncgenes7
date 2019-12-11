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
from skimage.transform import resize as skimage_resize
from skimage.transform import rotate as skimage_rotate
import tensorflow as tf

from ncgenes7.augmentations.image import ImageRandomCrop
from ncgenes7.augmentations.image import ImageFlipUpDown
from ncgenes7.augmentations.image import ImageHorizontalFlip
from ncgenes7.augmentations.image import ImageRandomBrightness
from ncgenes7.augmentations.image import ImageRandomContrast
from ncgenes7.augmentations.image import ImageRandomCutout
from ncgenes7.augmentations.image import ImageRandomHue
from ncgenes7.augmentations.image import ImageRandomSaturation
from ncgenes7.augmentations.image import ImageRandomRotation


class TestImageRandomBrightness(parameterized.TestCase, tf.test.TestCase):

    @patch("object_detection.utils.visualization_utils."
           "tf.image.adjust_brightness")
    def test_process(self, augmentation_fn):
        augmentation_fn.return_value = "inputs_augmented"
        inputs = "inputs"
        max_delta = 1.0

        augmentation = ImageRandomBrightness(
            max_delta=max_delta
        ).build()
        augmentation.create_random_variables()
        result = augmentation.process(images=inputs)
        augmentation_fn.assert_called_once_with(
            inputs, augmentation.random_variables["brightness_delta"])

        self.assertSetEqual(set(augmentation.generated_keys_all),
                            set(result))


class TestImageRandomContrast(parameterized.TestCase, tf.test.TestCase):

    @patch("object_detection.utils.visualization_utils."
           "tf.image.adjust_contrast")
    def test_process(self, augmentation_fn):
        augmentation_fn.return_value = "inputs_augmented"
        inputs = "inputs"

        augmentation = ImageRandomContrast(
            lower=0.1, upper=0.5,
        ).build()
        augmentation.create_random_variables()
        result = augmentation.process(images=inputs)
        augmentation_fn.assert_called_once_with(
            inputs, augmentation.random_variables["contrast_factor"])

        self.assertSetEqual(set(augmentation.generated_keys_all),
                            set(result))


class TestImageRandomHue(parameterized.TestCase, tf.test.TestCase):

    @patch("object_detection.utils.visualization_utils."
           "tf.image.adjust_hue")
    def test_process(self, augmentation_fn):
        augmentation_fn.return_value = "inputs_augmented"
        inputs = "inputs"

        augmentation = ImageRandomHue(
            max_delta=0.1
        ).build()
        augmentation.create_random_variables()
        result = augmentation.process(images=inputs)
        augmentation_fn.assert_called_once_with(
            inputs, augmentation.random_variables["hue_delta"])

        self.assertSetEqual(set(augmentation.generated_keys_all),
                            set(result))


class TestImageRandomSaturation(parameterized.TestCase, tf.test.TestCase):

    @patch("object_detection.utils.visualization_utils."
           "tf.image.adjust_saturation")
    def test_process(self, augmentation_fn):
        augmentation_fn.return_value = "inputs_augmented"
        inputs = "inputs"

        augmentation = ImageRandomSaturation(
            lower=0.1, upper=0.5,
        ).build()
        augmentation.create_random_variables()
        result = augmentation.process(images=inputs)
        augmentation_fn.assert_called_once_with(
            inputs, augmentation.random_variables["saturation_factor"])

        self.assertSetEqual(set(augmentation.generated_keys_all),
                            set(result))


class TestImageHorizontalFlip(parameterized.TestCase, tf.test.TestCase):
    def test_process(self):
        np.random.seed(6546)
        inputs_np = np.random.randn(20, 30, 3)
        augmentation = ImageHorizontalFlip(augmentation_probability=1.0).build()
        result = augmentation.process(images=tf.constant(inputs_np))
        result_eval = self.evaluate(result)
        result_must = inputs_np[:, ::-1, :]
        self.assertSetEqual(set(augmentation.generated_keys_all),
                            set(result))
        self.assertAllClose(result_must,
                            result_eval["images"])


class TestImageFlipUpDown(parameterized.TestCase, tf.test.TestCase):
    def test_process(self):
        np.random.seed(6546)
        inputs_np = np.random.randn(20, 30, 3)
        augmentation = ImageFlipUpDown(augmentation_probability=1.0).build()
        result = augmentation.process(images=tf.constant(inputs_np))
        result_eval = self.evaluate(result)
        result_must = inputs_np[::-1, :, :]
        self.assertSetEqual(set(augmentation.generated_keys_all),
                            set(result))
        self.assertAllClose(result_must,
                            result_eval["images"])


class TestImageRandomRotation(parameterized.TestCase, tf.test.TestCase):

    @patch("ncgenes7.utils.image_utils.interpolation_method_by_dtype",
           return_value="NEAREST")
    def test_process(self, interp_fn):
        inputs_np = np.random.randn(20, 30, 3)
        angle = 30
        angle_rad = angle * np.pi / 180
        augmentation = ImageRandomRotation(
            max_angle=angle_rad,
            augmentation_probability=1.0,
        ).build()

        result = augmentation.process(images=tf.constant(inputs_np))
        result_eval = self.evaluate(result)
        used_angle = result_eval["rotation_angle"]
        result_must = skimage_rotate(
            inputs_np, used_angle * 180 / np.pi, order=0)
        self.assertSetEqual(set(augmentation.generated_keys_all),
                            set(result))
        self.assertAllClose(result_must,
                            result_eval["images"])


class TestImageRandomCrop(parameterized.TestCase, tf.test.TestCase):

    @parameterized.parameters(
        {"min_scale": 0.5, "max_scale": None},
        {"min_scale": 0.5, "max_scale": 0.7},
        {"min_scale": 0.5, "resize_to_original": True}
    )
    @patch("ncgenes7.utils.image_utils.interpolation_method_by_dtype",
           return_value=1)
    def test_process(self,
                     interp_fn,
                     min_scale, max_scale=None,
                     resize_to_original=None):
        tf.reset_default_graph()
        inputs_np = np.random.rand(20, 30, 3).astype(np.float32)
        augmentation = ImageRandomCrop(
            min_scale=min_scale, max_scale=max_scale,
            resize_to_original=resize_to_original,
            augmentation_probability=1.0,
        ).build()
        result = augmentation.process(images=tf.constant(inputs_np))
        self.assertSetEqual(set(augmentation.generated_keys_all),
                            set(result))
        result_eval = self.evaluate(result)
        offset_eval = np.floor(
            result_eval["crop_offset"] * np.array([20, 30])).astype(np.int32)
        size_eval = np.floor(
            result_eval["crop_scale"] * np.array([20, 30])).astype(np.int32)

        result_must = inputs_np[
                      offset_eval[0]:offset_eval[0] + size_eval[0],
                      offset_eval[1]:offset_eval[1] + size_eval[1]]

        if resize_to_original:
            result_must = skimage_resize(result_must, [20, 30], 0,
                                         anti_aliasing=False)
        self.assertAllClose(result_must,
                            result_eval["images"])


class TestImageRandomCutout(parameterized.TestCase, tf.test.TestCase):
    @parameterized.parameters({"is_normalized": True},
                              {"is_normalized": False})
    def test_process(self, is_normalized):
        tf.reset_default_graph()
        inputs_np = np.random.randn(10, 15, 3).astype(np.float32)
        augmentation = ImageRandomCutout(
            min_cut_length=[0.1, 0.1], max_cut_length=[0.5, 0.5],
            is_normalized=is_normalized,
            augmentation_probability=1.0,
        ).build()
        result = augmentation.process(images=tf.constant(inputs_np))
        self.assertSetEqual(set(augmentation.generated_keys_all),
                            set(result))
        result_eval = self.evaluate(result)

        cut_lengths = result_eval["cut_lengths"]
        cut_offset = result_eval["cut_offset"]

        offset_eval = np.floor(cut_offset * np.array([10, 15])).astype(np.int32)
        size_eval = np.floor(
            cut_lengths * np.array([10, 15])).astype(np.int32) + 1

        result_must = np.copy(inputs_np)
        if is_normalized:
            replace_value = 0
        else:
            replace_value = np.mean(result_must, (0, 1))
        result_must[
            slice(offset_eval[0], offset_eval[0] + size_eval[0]),
            slice(offset_eval[1], offset_eval[1] + size_eval[1])
        ] = replace_value
        self.assertAllClose(result_must,
                            result_eval["images"])
