# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

import os

from absl.testing import parameterized
import numpy as np
import skimage.io
import tensorflow as tf

from ncgenes7.data_fields.images import ImageDataFields
from ncgenes7.utils import image_io_utils


class TestImageIOUtils(parameterized.TestCase, tf.test.TestCase):

    def test_decode_images_with_fnames(self):
        file_list = {ImageDataFields.images: 'image.png',
                     'labels': 'labels.png'}
        image_size = [100, 50]
        num_channels = {ImageDataFields.images: 3, 'labels': 1}
        cast_dtypes = {ImageDataFields.images: tf.float32, 'labels': tf.int32}
        res = image_io_utils.decode_images_with_fnames(
            file_list, image_size, num_channels, cast_dtypes)
        self.assertSetEqual(
            set(res.keys()),
            {ImageDataFields.images, 'labels',
             ImageDataFields.images_fnames, 'labels_fnames'})
        self.assertListEqual(image_size + [3],
                             res[ImageDataFields.images].shape.as_list())
        self.assertListEqual(image_size + [1], res['labels'].shape.as_list())
        self.assertEqual(res['labels_fnames'], 'labels.png')
        self.assertEqual(res[ImageDataFields.images_fnames], 'image.png')
        self.assertEqual(res[ImageDataFields.images].dtype, tf.float32)
        self.assertEqual(res['labels'].dtype, tf.int32)

    def test_save_isolines_run(self):
        save_dir = self.get_temp_dir()
        save_fname = os.path.join(save_dir, "image.png")
        image = np.random.rand(100, 200, 3)
        depth_image = np.random.rand(100, 200) * 100
        max_isodist = 30
        image_io_utils.save_isolines(save_fname, depth_image, image,
                                     max_isodist=max_isodist)

    def test_save16bit(self):
        image_16_bit = np.random.randint(
            0, 2 ** 16, size=(200, 300)).astype(np.uint16)
        image = image_16_bit.astype(np.float32) / 2 ** 16
        save_dir = self.get_temp_dir()
        save_fname = os.path.join(save_dir, "image.png")
        image_io_utils.save_16bit_png(save_file_name=save_fname,
                                      image=image)

        image_loaded = skimage.io.imread(save_fname)
        self.assertAllClose(image_16_bit,
                            image_loaded, atol=1)
