# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

"""
Image DataFeeders
"""

from datetime import datetime
import logging
from typing import Optional

# pylint: disable=wrong-import-order
# is issue with pylint
import cv2
import nucleus7 as nc7
from nucleus7.data.data_feeder import DataFeeder
from nucleus7.data.data_feeder import DataFeederFileList
import numpy as np
import skimage.io
import skimage.transform

from ncgenes7.data_fields.images import ImageDataFields


class ImageDataFeeder(DataFeederFileList):
    """
    Class for the feeding the images data to the tensorflow graph

    Parameters
    ----------
    image_number_of_channels
        number of image channels
    image_size
        size of image as  [height, width]; if specified, loaded images will be
        resized to it; if not specified, then images will have the
        original size; may cause errors if sizes are different across images,
        since it is not possible to combine them to the batch

    Attributes
    ----------
    generated_keys
        * images : images
        * images_fnames : file names of images
    """
    file_list_keys = [
        ImageDataFields.images,
    ]
    generated_keys = [
        ImageDataFields.images,
        ImageDataFields.images_fnames,
    ]

    def __init__(self, *,
                 file_list: nc7.data.FileList,
                 image_number_of_channels: int = 3,
                 image_size: Optional[list] = None,
                 **data_feeder_kwargs):
        self.image_number_of_channels = image_number_of_channels
        self.image_size = image_size
        super().__init__(file_list=file_list, **data_feeder_kwargs)

    def read_element_from_file_names(self, images: str):
        # pylint: disable=arguments-differ
        # base class has more generic signature
        res = {}
        images_fnames = images
        try:
            image = self._read_image(images_fnames)
        except OSError:
            image = self._get_default_image(images_fnames)
        res[ImageDataFields.images] = image
        res[ImageDataFields.images_fnames] = images_fnames
        return res

    def _read_image(self, fname: str) -> np.ndarray:
        image = skimage.io.imread(fname)
        image = self._process_image(image)
        return image

    def _get_default_image(self, images_fnames):
        logger = logging.getLogger(__name__)
        logger.warning("Image from %s is corrupted!", images_fnames)
        image_size = self.image_size or [100, 100]
        image = np.zeros(image_size
                         + [self.image_number_of_channels], np.float32)
        return image

    def _process_image(self, image: np.ndarray):
        image = _format_last_dimension(image, self.image_number_of_channels)
        image = _rescale_according_to_dtype(image)
        if self.image_size is not None:
            image = skimage.transform.resize(
                image, self.image_size, mode='reflect')
        return image


class ImageMultiScaleDataFeeder(ImageDataFeeder):
    """
    Class for the feeding the images data to the tensorflow graph
    in multi scales

    Result of get_batch function is then list of same like dicts
    for different scales and different hflips.
    In case of horizontal flips, order is following
    [scale0, flipped scale0, scale1, flipped scale1 ..]

    Parameters
    ----------
    scales
        list of scales of the data images to use;
    use_horizontal_flip
        if the images should be additionally flipped horizontally;
    interpolation_order
        interpolation order for scaling; see
        :obj:`skimage.transform.rescale`

    Attributes
    ----------
    generated_keys : list
        * image : image
        * image_fname : file name of read image
    """
    generated_keys = [ImageDataFields.images,
                      ImageDataFields.images_fnames]

    def __init__(self, *,
                 scales: Optional[list] = None,
                 use_horizontal_flip: bool = False,
                 interpolation_order: int = 1,
                 **data_feeder_kwargs):
        super().__init__(**data_feeder_kwargs)
        self.scales = scales or [1.0, 0.5]
        self.use_horizontal_flip = use_horizontal_flip
        self.interpolation_order = interpolation_order

    def get_batch(self, batch_size):
        if self._generator is None:
            self._generator = self.build_generator()
        logger = logging.getLogger(__name__)
        batch = self._get_multiscale_batch(batch_size)

        data_from_first_key = batch[0][self.generated_keys[0]]
        is_exhausted = (len(data_from_first_key) == 0 or
                        (len(data_from_first_key) < batch_size - 1))

        if is_exhausted and not self.allow_smaller_final_batch:
            logger.info('generator of %s is exhausted', self.__class__.__name__)
            raise StopIteration()

        feeded_values = [{k: np.stack(v, axis=0)
                          for k, v in batch_.items()} for batch_ in batch]

        return feeded_values

    def _get_multiscale_batch(self, batch_size):
        scales = self.scales
        if self.use_horizontal_flip:
            scales = [sc_ for sc in zip(scales, scales) for sc_ in sc]
        horizontal_flip_indicator = (
            [False, True] * len(self.scales)
            if self.use_horizontal_flip else
            [False] * len(self.scales)
        )
        batch = [{n: [] for n in self.generated_keys} for _ in scales]
        number_of_samples = 0
        while number_of_samples < batch_size:
            try:
                data_element = self.read_element_from_file_names(
                    **next(self._generator))

                if not self.data_filter_true(**data_element):
                    continue

                image = data_element[ImageDataFields.images]
                image_fname = data_element[ImageDataFields.images_fnames]
                for isc, (scale, perform_flip) in enumerate(
                        zip(scales, horizontal_flip_indicator)):
                    image_transformed = self._transform_image(
                        image, scale, perform_flip)
                    batch[isc][ImageDataFields.images].append(image_transformed)
                    batch[isc][ImageDataFields.images_fnames].append(
                        image_fname)
                number_of_samples += 1
            except StopIteration:
                break
        return batch

    def _transform_image(self, image, scale, perform_flip):
        image_transformed = skimage.transform.rescale(
            image, scale, order=self.interpolation_order,
            mode='constant')
        if perform_flip:
            image_transformed = image_transformed[:, ::-1, :]
        return image_transformed


class CameraImageDataFeeder(DataFeeder):
    """
    Class for the feeding the images data to the tensorflow graph

    Parameters
    ----------
    convert_image_to_gray
        controls if colored camera image should be converted to grey scale
    image_size
        size of image; if specified, loaded images will be resized to it;
        [h, w]
    camera_index
        camera index; to open camera :obj:`cv2.VideoCapture` is used
    number_of_iterations
        number of iterations until StopIteration occur; if set to 0 or None,
        will create infinite generator

    Attributes
    ----------
    generated_keys
        * images : images
        * images_fnames : strings in format cam_{camera_id}_{timestamp}
    """
    generated_keys = [ImageDataFields.images, ImageDataFields.images_fnames]

    def __init__(self, *,
                 convert_image_to_gray: bool = False,
                 image_size: Optional[list] = None,
                 camera_index: int = 0,
                 number_of_iterations: int = 100,
                 **data_feeder_kwargs):
        self.convert_image_to_gray = convert_image_to_gray
        self.image_size = image_size
        self.camera_index = camera_index
        self.number_of_iterations = number_of_iterations
        self._cap = cv2.VideoCapture(camera_index)
        if not self._cap.isOpened():
            raise ValueError("Camera {} is in use or it is not connected!"
                             "".format(camera_index))
        super().__init__(**data_feeder_kwargs)

    def build_generator(self):
        if self.number_of_iterations:
            for i in range(self.number_of_iterations):
                yield i
        else:
            while True:
                yield 0

    def create_data_from_generator(self, data):
        res = {}
        image, timestamp = self._read_image()
        res[ImageDataFields.images] = image
        res[ImageDataFields.images_fnames] = "cam_{}_{}".format(
            self.camera_index, timestamp)
        return res

    def _read_image(self):
        _, image = self._cap.read()
        timestamp = datetime.now().strftime('%Y-%m-%d_%H:%M:%S.%f')
        if self.convert_image_to_gray:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = np.expand_dims(image, -1)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = _rescale_according_to_dtype(image)
        if self.image_size is not None:
            image = skimage.transform.resize(
                image, self.image_size, mode='reflect')
        return image, timestamp

    def __del__(self):
        self._cap.release()


def _format_last_dimension(image: np.ndarray, image_number_of_channels: int
                           ) -> np.ndarray:
    if image.ndim == 2:
        image = np.expand_dims(image, -1)
    last_dimension = image.shape[-1]
    if last_dimension > image_number_of_channels:
        image = image[:, :, :image_number_of_channels]
    elif last_dimension < image_number_of_channels:
        repeat_factor = int(image_number_of_channels / last_dimension)
        image = np.tile(image, [1, 1, repeat_factor])
    return image


def _rescale_according_to_dtype(image):
    image_max_format = {np.dtype(np.uint16): 65535,
                        np.dtype(np.uint8): 255,
                        np.dtype(np.float32): 1,
                        np.dtype(np.float64): 1}[image.dtype]
    image = image.astype(np.float32)
    if image.max() > 1:
        image /= image_max_format
    return image
