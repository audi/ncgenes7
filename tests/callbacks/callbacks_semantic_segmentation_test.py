# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

import os

import numpy as np
import skimage.io
import tensorflow as tf

from ncgenes7.callbacks.semantic_segmentation import SemanticSegmentationSaver
from ncgenes7.utils import image_utils


class TestSemanticSegmentationImageSaver(tf.test.TestCase):

    def setUp(self):
        self.batch_size = 2
        self.num_classes = 4
        self.images = np.random.rand(self.batch_size, 5, 10, 3)
        self.logits = np.random.randn(self.batch_size, 5,
                                      10, self.num_classes) * 10
        self.rgb_class_mapping = {0: [1, 10, 20],
                                  1: [10, 20, 30],
                                  2: [20, 30, 40],
                                  3: [40, 50, 60]}
        self.images_fnames = np.array(['fname_{}'.format(i)
                                       for i in range(self.batch_size)])

        self.segmentation_classes = np.argmax(self.logits, -1)
        self.segmentation_classes_rgb = image_utils.labels2rgb(
            self.segmentation_classes, self.rgb_class_mapping)
        self.fnames_classes_must = [("fname_%s_segmentation_classes.png" % i)
                                    for i in range(self.batch_size)]
        self.fnames_probabilities_must = [
            ("fname_%s_probabilities.npy" % i)
            for i in range(self.batch_size)]

    def test_save_names(self):
        temp_dir = self.get_temp_dir()
        saver_callback = self._get_saver_callback()
        saver_callback.log_dir = temp_dir
        saver_callback.save_probabilities = True
        saver_callback.on_iteration_end(
            images=self.images,
            segmentation_class_logits=self.logits,
            save_names=self.images_fnames)
        fnames_must = self.fnames_classes_must + self.fnames_probabilities_must
        fnames_result = os.listdir(temp_dir)
        self.assertListEqual(sorted(fnames_must),
                             sorted(fnames_result))

    def test_segmentation_classes_only(self):
        temp_dir = self.get_temp_dir()
        self.rgb_class_mapping = None
        saver_callback = self._get_saver_callback()
        saver_callback.log_dir = temp_dir
        saver_callback.on_iteration_end(
            segmentation_classes=self.segmentation_classes,
            save_names=self.images_fnames)
        segmentation_classes_result = np.stack(
            [skimage.io.imread(os.path.join(temp_dir, each_fname))
             for each_fname in self.fnames_classes_must])
        np.testing.assert_array_equal(self.segmentation_classes,
                                      segmentation_classes_result[:, :, :, 1])

    def test_segmentation_classes_with_logits(self):
        temp_dir = self.get_temp_dir()
        self.rgb_class_mapping = None
        saver_callback = self._get_saver_callback()
        saver_callback.log_dir = temp_dir
        saver_callback.on_iteration_end(
            segmentation_class_logits=self.logits,
            segmentation_classes=self.segmentation_classes,
            save_names=self.images_fnames)
        segmentation_classes_result = np.stack(
            [skimage.io.imread(os.path.join(temp_dir, each_fname))
             for each_fname in self.fnames_classes_must])
        np.testing.assert_array_equal(self.segmentation_classes,
                                      segmentation_classes_result[:, :, :, 1])

    def test_segmentation_classes_rgb_with_classes(self):
        temp_dir = self.get_temp_dir()
        saver_callback = self._get_saver_callback()
        saver_callback.log_dir = temp_dir
        saver_callback.on_iteration_end(
            segmentation_classes=self.segmentation_classes,
            save_names=self.images_fnames)
        segmentation_classes_rgb_result = np.stack(
            [skimage.io.imread(os.path.join(temp_dir, each_fname))
             for each_fname in self.fnames_classes_must])
        np.testing.assert_array_equal(
            self.segmentation_classes_rgb,
            segmentation_classes_rgb_result)

    def test_segmentation_classes_rgb_with_logits(self):
        temp_dir = self.get_temp_dir()
        saver_callback = self._get_saver_callback()
        saver_callback.log_dir = temp_dir
        saver_callback.on_iteration_end(
            segmentation_class_logits=self.logits,
            save_names=self.images_fnames)
        segmentation_classes_rgb_result = np.stack(
            [skimage.io.imread(os.path.join(temp_dir, each_fname))
             for each_fname in self.fnames_classes_must])
        np.testing.assert_array_equal(
            self.segmentation_classes_rgb,
            segmentation_classes_rgb_result)

    def test_input_segmentation_rgb_blend(self):
        temp_dir = self.get_temp_dir()
        saver_callback = self._get_saver_callback()
        saver_callback.log_dir = temp_dir
        saver_callback.on_iteration_end(
            images=self.images,
            segmentation_classes=self.segmentation_classes,
            save_names=self.images_fnames)
        blended_segmentation_must = image_utils.blend_images(
            self.images,
            self.segmentation_classes_rgb.astype(np.float32) / 255.)
        blended_segmentation_result = np.stack(
            [skimage.io.imread(os.path.join(temp_dir, each_fname))
             for each_fname in self.fnames_classes_must])
        blended_segmentation_result = blended_segmentation_result.astype(
            np.float32) / 255.
        self.assertAllClose(blended_segmentation_must,
                            blended_segmentation_result, atol=1 / 255.)

    def test_save_probabilities(self):
        temp_dir = self.get_temp_dir()
        saver_callback = self._get_saver_callback()
        saver_callback.save_probabilities = True
        saver_callback.log_dir = temp_dir
        saver_callback.on_iteration_end(
            segmentation_class_logits=self.logits,
            save_names=self.images_fnames)
        probabilities_must = _probabilities_from_logits(self.logits)
        probabilities_result = np.stack([
            np.load(os.path.join(temp_dir, each_fname), allow_pickle=False)
            for each_fname in self.fnames_probabilities_must])
        np.testing.assert_array_equal(probabilities_must, probabilities_result)

    def test_save_probabilites_without_logits(self):
        temp_dir = self.get_temp_dir()
        saver_callback = self._get_saver_callback()
        saver_callback.save_probabilities = True
        saver_callback.log_dir = temp_dir
        with self.assertRaises(ValueError):
            saver_callback.on_iteration_end(
                segmentation_classes=self.segmentation_classes,
                save_names=self.images_fnames)

    def _get_saver_callback(self):
        saver_callback = SemanticSegmentationSaver(
            rgb_class_mapping=self.rgb_class_mapping,
            inbound_nodes=[])
        return saver_callback


def _probabilities_from_logits(logits):
    exp = np.exp(logits - np.amax(logits, axis=-1, keepdims=True))
    return exp / np.sum(exp, axis=-1, keepdims=True)
