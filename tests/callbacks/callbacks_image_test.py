# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

import os

from absl.testing import parameterized
from nucleus7.coordinator.configs import RunIterationInfo
import numpy as np
import skimage.io
import tensorflow as tf

from ncgenes7.callbacks.image import DepthSaver
from ncgenes7.callbacks.image import ImageSaver
from ncgenes7.data_fields.images import ImageDataFields


class TestImageSaver(parameterized.TestCase, tf.test.TestCase):

    def test_on_iteration_end(self):
        np.random.seed(6546)
        temp_dir = self.get_temp_dir()
        iteration_info = RunIterationInfo(epoch_number=2,
                                          iteration_number=5,
                                          execution_time=0.123)
        image_saver_callback = ImageSaver(inbound_nodes=[])
        image_saver_callback.iteration_info = iteration_info
        image_saver_callback.log_dir = temp_dir
        images = np.random.randint(0, 255, size=(3, 20, 30, 3))
        data = {"images": images,
                "save_names": np.array(["file1.ext",
                                        "file2.ext",
                                        "file3.ext"]).astype(bytes)}
        image_saver_callback.on_iteration_end(**data)

        base_saved_fnames = ["file1.png", "file2.png", "file3.png"]
        self.assertSetEqual(set(base_saved_fnames),
                            set(os.listdir(temp_dir)))

        for each_saved_fname, input_image in zip(base_saved_fnames, images):
            fname_full = os.path.join(temp_dir, each_saved_fname)
            image_read = skimage.io.imread(fname_full)
            self.assertAllClose(input_image,
                                image_read)


class TestStereoDepthImageSaver(parameterized.TestCase, tf.test.TestCase):

    @parameterized.parameters(
        {'save_16bit': True, 'save_isolines': True},
        {'save_16bit': True, 'save_isolines': False},
        {'save_16bit': False, 'save_isolines': True})
    def test_on_iteration_end(self, save_16bit, save_isolines):
        np.random.seed(45665)
        batch_size = 2
        temp_dir = self.get_temp_dir()
        n_isoevel = 12
        max_isodist = 30
        max_distance = 100.
        iteration_info = RunIterationInfo(epoch_number=2,
                                          iteration_number=5,
                                          execution_time=0.123)

        image = np.random.rand(batch_size, 20, 30, 3)
        depth = np.random.rand(batch_size, 20, 30) * 100
        image_fnames = ('fname_{}'.format(i)
                        for i in range(batch_size))

        data = {ImageDataFields.images: image,
                "save_names": image_fnames,
                ImageDataFields.depth_images: depth}

        callback = DepthSaver(save_depth_isolines=save_isolines,
                              save_16bit_depth=save_16bit,
                              number_of_isolevels=n_isoevel,
                              max_isodist=max_isodist,
                              max_distance=max_distance,
                              border_offset=1,
                              inbound_nodes=[]).build()
        callback.log_dir = temp_dir
        callback.iteration_info = iteration_info
        callback.on_iteration_end(**data)

        base_saved_fnames_isolines = ["fname_{}_depth_isolines.png".format(i)
                                      for i in range(batch_size)]
        base_saved_fnames_16bit = ["fname_{}_depth_16bit.png".format(i)
                                   for i in range(batch_size)]
        base_saved_fnames = []
        if save_16bit:
            base_saved_fnames += base_saved_fnames_16bit
        if save_isolines:
            base_saved_fnames += base_saved_fnames_isolines

        self.assertSetEqual(set(base_saved_fnames),
                            set(os.listdir(temp_dir)))
