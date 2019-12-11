# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

"""
classes for general purpose callback implementation
"""

import os
import warnings

import matplotlib.pyplot as plt
import nucleus7 as nc7
import skimage.io

from ncgenes7.data_fields.images import ImageDataFields
from ncgenes7.utils import image_io_utils
from ncgenes7.utils import image_utils


class ImageSaver(nc7.coordinator.SaverCallback):
    """
    Image saver callback

    Parameters
    ----------
    save_file_extension
        extension of the save file

    Attributes
    ----------
    incoming_keys
        * images : image or list of images, [bs, h, w, num_channels], np.float

    Raises
    ------
    ValueError
        if key_for_basename was provided
    """
    incoming_keys = [
        ImageDataFields.images,
    ]

    def __init__(self,
                 save_file_extension: str = "png",
                 **callback_kwargs):
        super(ImageSaver, self).__init__(**callback_kwargs)

        if save_file_extension.startswith("."):
            save_file_extension = save_file_extension[1:]
        self.save_file_extension = save_file_extension

    def save_sample(self, images):
        # pylint: disable=arguments-differ
        # parent method has more generic signature
        save_name_without_ext = os.path.splitext(self.save_name)[0]
        save_fname = ".".join([save_name_without_ext, self.save_file_extension])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            skimage.io.imsave(save_fname, images)


class DepthFromDisparityCalculator(nc7.coordinator.CoordinatorCallback):
    """
    Calculate the depth out of disparity and camera properties

    Parameters
    ----------
    clip_distance
        values after that distance will be clipped, [m]
    baseline
        distance between stereo cameras, [m]
    focal_length
        focal length of the camera, [m]
    sensor_width
        sensor width, [m]

    Attributes
    ----------
    incoming_keys
        * disparity_images : disparity for left image, [bs, h, w], np.float
    generated_keys
        * depth_images : images with depth, [bs, h, w], np.float
    """
    incoming_keys = [
        ImageDataFields.disparity_images,
    ]
    generated_keys = [
        ImageDataFields.depth_images,
    ]

    def __init__(self, *,
                 baseline: float = 0.3,
                 focal_length: float = 1.4e-3,
                 sensor_width: float = 7.4e-3,
                 clip_distance=125,
                 **callback_kwargs):
        super(DepthFromDisparityCalculator, self).__init__(**callback_kwargs)
        self.baseline = baseline
        self.focal_length = focal_length
        self.sensor_width = sensor_width
        self.clip_distance = clip_distance

    def on_iteration_end(self, *, disparity_images):
        # pylint: disable=arguments-differ
        # parent method has more generic signature
        depth_images = image_utils.disparity2depth(
            disparity_images, baseline_dist=self.baseline,
            focal_length=self.focal_length, sensor_width=self.sensor_width,
            clip_distance=self.clip_distance)
        return {ImageDataFields.depth_images: depth_images}


class DepthSaver(nc7.coordinator.SaverCallback):
    """
    Save the depth as 16 bit png files and isolines if depth

    Following images are stored to log_dir:
        * depth 16 bit (optional): {image_fname}_depth_16bit.png
        * depth isolines (optional): {image_fname}_depth_isolines.png

    Parameters
    ----------
    save_16bit_depth
        if the 16bit image should be saved
    save_depth_isolines
        if the isolines images should be saved
    max_isodist
        max distance to draw the isolines on, [m]
    border_offset
        offset from border in pixels for iso levels calculation
    number_of_isolevels
        number of isolevels
    max_distance
        max distance in depth images, [m]

    Attributes
    ----------
    incoming_keys
        * images : left image,[bs, h, w, num_channels], np.float
        * depth_images : depth image, , [bs, h, w], np.float
        * images_fnames : image file name
    """
    incoming_keys = [
        ImageDataFields.depth_images,
        "_", ImageDataFields.images,
    ]

    def __init__(self, *,
                 save_16bit_depth: bool = False,
                 save_depth_isolines: bool = True,
                 max_isodist=30,
                 number_of_isolevels=12,
                 max_distance=125.0,
                 border_offset: int = 10,
                 **callback_kwargs):

        if "key_for_basename" in callback_kwargs:
            msg = ("do not provide key_for_basename for ImageSaver. It is all "
                   "the time images_fnames")
            raise ValueError(msg)
        super(DepthSaver, self).__init__(**callback_kwargs)
        self.save_16bit_depth = save_16bit_depth
        self.save_depth_isolines = save_depth_isolines
        self.max_isodist = max_isodist
        self.number_of_isolevels = number_of_isolevels
        self.border_offset = border_offset
        self.max_distance = max_distance

    def save_sample(self, depth_images, images=None):
        # pylint: disable=arguments-differ
        # parent method has more generic signature
        if self.save_depth_isolines:
            isolines_fname = self.save_name + "_depth_isolines.png"
            image_io_utils.save_isolines(
                isolines_fname, depth_images, images,
                max_isodist=self.max_isodist,
                number_of_isolevels=self.number_of_isolevels,
                border_offset=self.border_offset)

        if self.save_16bit_depth:
            depth_16bit_fname = self.save_name + "_depth_16bit.png"
            depth_images = depth_images / self.max_distance
            image_io_utils.save_16bit_png(depth_16bit_fname, depth_images)


class ImagePlotterCallback(nc7.coordinator.CoordinatorCallback):
    """
    Plot the image on GUI using matplotlib pyplot

    Attributes
    ----------
    incoming_keys
        * images : left image,[bs, h, w, num_channels], np.float
    """
    incoming_keys = [
        ImageDataFields.images,
    ]

    def on_iteration_end(self, images, images_fnames=None):
        # pylint: disable=arguments-differ
        # parent method has more generic signature
        iteration_info = self.iteration_info
        for each_sample_index, each_image in enumerate(images):
            title = "Epoch: {:3d}; iteration: {:5d}; sample: {:3d}".format(
                iteration_info.epoch_number, iteration_info.iteration_number,
                each_sample_index)
            if images_fnames is not None:
                sample_file_name = images_fnames[each_sample_index]
                title += "\nReference file name: {}".format(sample_file_name)
            plt.figure()
            plt.imshow(each_image)
            plt.title(title)
            plt.show()
