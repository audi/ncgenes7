# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

import itertools
import os

from absl.testing import parameterized
import nucleus7 as nc7
import numpy as np
import skimage.io
import skimage.transform
import tensorflow as tf

from ncgenes7.data_feeders.image import ImageDataFeeder
from ncgenes7.data_feeders.image import ImageMultiScaleDataFeeder
from ncgenes7.data_fields.images import ImageDataFields


class TestImageDataFeeder(parameterized.TestCase, tf.test.TestCase):
    def setUp(self):
        temp_dir = self.get_temp_dir()
        self.fname_image = os.path.join(temp_dir, 'image.png')
        self.file_names = {'images': [self.fname_image]}
        self.file_list = nc7.data.FileList.from_matched_file_names(
            self.file_names)

    @parameterized.parameters(
        {'image_format': 'uint8', 'num_channels': 1},
        {'image_format': 'uint8', 'num_channels': 3},
        {'image_format': 'uint16', 'num_channels': 1},
        {'image_format': 'uint16', 'num_channels': 3})
    def test_read_element_from_fname(self, image_format, num_channels):
        image, dtype_max = self._get_image(image_format, num_channels)
        skimage.io.imsave(self.fname_image, image)
        file_list = self.file_list
        data_feeder = ImageDataFeeder(file_list=file_list,
                                      image_number_of_channels=num_channels
                                      ).build()
        image_read = data_feeder.read_element_from_file_names(self.fname_image)

        image_must = (skimage.io.imread(self.fname_image).astype(np.float32) /
                      dtype_max)

        if num_channels == 1:
            image_must = np.expand_dims(image_must, -1)
        self.assertAllClose(image_read[ImageDataFields.images], image_must)
        self.assertEqual(image_read[ImageDataFields.images_fnames],
                         self.fname_image)

    @staticmethod
    def _get_image(image_format, num_channels):
        if num_channels == 1:
            image = np.random.rand(20, 20)
        else:
            image = np.random.rand(20, 20, num_channels)
        # for some reason, it is possible to store only 8 bit for 3 channels
        if (image_format == 'uint8' or
                (image_format == 'uint16' and num_channels > 1)):
            dtype_max = 255.
            image *= dtype_max
            image = image.astype(np.uint8)
        else:
            dtype_max = 65535.
            image *= dtype_max
            image = image.astype(np.uint16)
        return image, dtype_max


class TestImageMultiScaleDataFeeder(parameterized.TestCase, tf.test.TestCase):
    def setUp(self):
        self.batch_size = 3
        temp_dir = self.get_temp_dir()
        self.fnames_images = [os.path.join(temp_dir, 'image_{}.png'.format(i))
                              for i in range(self.batch_size)]
        self.file_list = nc7.data.FileList.from_matched_file_names(
            {'images': self.fnames_images})

    @parameterized.parameters(
        {'image_format': 'uint8', 'num_channels': 1, 'scales': [1.0],
         'use_hflip': True, "interpolation_order": 1},
        {'image_format': 'uint8', 'num_channels': 3, 'scales': [1, 0.5],
         'use_hflip': True, "interpolation_order": 1},
        {'image_format': 'uint16', 'num_channels': 1,
         'scales': [0.5, 0.25], 'use_hflip': False,
         "interpolation_order": 0},
        {'image_format': 'uint16', 'num_channels': 3,
         'scales': [0.5, 0.25], 'use_hflip': True,
         "interpolation_order": 0})
    def test_feed_batch(self, image_format, num_channels, scales,
                        use_hflip, interpolation_order):
        images = []

        file_names = self.fnames_images
        for i in range(self.batch_size):
            image, dtype_max = self._get_image(image_format, num_channels)
            images.append(image)
            skimage.io.imsave(file_names[i], image)

        data_feeder = ImageMultiScaleDataFeeder(
            file_list=self.file_list, scales=scales,
            use_horizontal_flip=use_hflip,
            image_number_of_channels=num_channels,
            interpolation_order=interpolation_order).build()
        data_batch = data_feeder.get_batch(self.batch_size)

        list_len_must = len(scales) * (2 if use_hflip else 1)
        self.assertEqual(len(data_batch), list_len_must)
        images_load = [skimage.io.imread(f).astype(np.float32) / dtype_max
                       for f in file_names]
        if num_channels == 1:
            images_load = [np.expand_dims(img, -1) for img in images_load]

        images_must = [
            [skimage.transform.rescale(img, scale, interpolation_order,
                                       mode='constant')
             for img in images_load] for scale in scales]

        if use_hflip:
            images_flipped = [[img[:, ::-1, :] for img in img_sc]
                              for img_sc in images_must]
            images_must = list(itertools.chain(
                *zip(images_must, images_flipped)))
        images_must = [np.stack(imgs_sc, 0) for imgs_sc in images_must]

        data_only_images = [imgs_sc[ImageDataFields.images]
                            for imgs_sc in data_batch]

        data_only_fnames = [imgs_sc[ImageDataFields.images_fnames]
                            for imgs_sc in data_batch]

        self.assertAllClose(images_must,
                            data_only_images)
        self.assertListEqual(file_names,
                             list(data_only_fnames[0]))
        for i in range(1, list_len_must):
            self.assertListEqual(list(data_only_fnames[i]),
                                 list(data_only_fnames[0]))

    def _get_image(self, image_format, num_channels):
        if num_channels == 1:
            image = np.random.rand(20, 20)
        else:
            image = np.random.rand(20, 20, num_channels)
        # for some reason, it is possible to store only 8 bit for 3 channels
        if (image_format == 'uint8' or
                (image_format == 'uint16' and num_channels > 1)):
            dtype_max = 255.
            image *= dtype_max
            image = image.astype(np.uint8)
        else:
            dtype_max = 65535.
            image *= dtype_max
            image = image.astype(np.uint16)
        return image, dtype_max
