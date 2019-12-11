# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

import math

from absl.testing import parameterized
import numpy as np
import skimage.transform
import tensorflow as tf

from ncgenes7.utils import image_utils


class TestImageUtils(parameterized.TestCase, tf.test.TestCase):

    @parameterized.parameters({'use_mapping': True},
                              {'use_mapping': False})
    def test_labels2rgb(self, use_mapping):
        np.random.seed(45665)
        if use_mapping:
            rgb_class_mapping = {0: [1, 10, 20],
                                 1: [10, 20, 30],
                                 2: [20, 30, 40]}
        else:
            rgb_class_mapping = {}
        image = np.random.randint(0, 3, size=[2, 5])
        image_rgb = image_utils.labels2rgb(image, rgb_class_mapping)
        if use_mapping:
            image_rgb_must = np.zeros([2, 5, 3], np.int64)
            for cl in rgb_class_mapping:
                for ch in range(3):
                    image_rgb_must[..., ch][image == cl] = (
                        rgb_class_mapping[cl][ch])
        else:
            image_rgb_must = np.tile(np.expand_dims(image, -1), [1, 1, 3])
        self.assertAllClose(image_rgb, image_rgb_must)

    def test_decode_class_ids_from_rgb(self):
        rgb_class_mapping = {
            1: [10, 20, 30],
            2: [0, 1, 10],
            3: [5, 120, 255],
            4: [51, 60, 0],
        }
        rgb_class_ids_mapping_hashed = {
            sum([rgb[0] + rgb[1] * 1000 + rgb[2] * 10 ** 6]): k
            for k, rgb in rgb_class_mapping.items()}
        labels = np.random.randint(0, 4, size=[2, 5])
        labels_rgb = image_utils.labels2rgb(labels, rgb_class_mapping)
        labels_decoded = image_utils.decode_class_ids_from_rgb(
            labels_rgb, rgb_class_ids_mapping_hashed)
        self.assertAllClose(labels,
                            labels_decoded)

    def test_maybe_resize_and_expand_image(self):
        image1 = np.random.rand(2, 10, 20)
        image2 = np.random.rand(2, 20, 30, 3)
        image1_resized = image_utils.maybe_resize_and_expand_image(
            image1, image2)
        image1_resized_must = np.stack(
            [skimage.transform.resize(
                np.tile(np.expand_dims(each_image, -1), [1, 1, 3]),
                image2.shape[1:3], mode='reflect')
                for each_image in image1], 0)
        self.assertAllClose(image1_resized_must,
                            image1_resized)

    @parameterized.parameters({'ncols': 3},
                              {'nrows': 3},
                              {'ncols': 1},
                              {'nrows': 1},
                              {'nrows': 1, 'image_size': [250, 300]},
                              {'ncols': 1, 'image_size': [250, 300]})
    def test_concatenate_images(self, nrows=-1, ncols=-1, image_size=None):
        images_num = 7
        images_np = [np.random.rand(10, 50 - i * 5, 90 - i * 6, 3)
                     for i in range(images_num)]
        if image_size is None:
            image_h, image_w = images_np[0].shape[1:3]
        else:
            image_h, image_w = image_size
        images = [tf.constant(image_np, tf.float32) for image_np in images_np]
        image_rescale = 1.0,
        image_grid_border_width = 3
        image_concat = image_utils.concatenate_images(
            images, nrows, ncols, image_rescale,
            image_grid_border_width,
            image_size=image_size)
        image_concat_ = self.evaluate(image_concat)
        if nrows == -1:
            nrows = math.ceil(images_num / ncols)
        if ncols == -1:
            ncols = math.ceil(images_num / nrows)
        image_w_must = ncols * image_w + (ncols - 1) * image_grid_border_width
        image_h_must = nrows * image_h + (nrows - 1) * image_grid_border_width
        image_shape_must = [10, image_h_must, image_w_must, 3]
        self.assertListEqual(list(image_concat_.shape), image_shape_must)
