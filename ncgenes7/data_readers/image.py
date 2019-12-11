# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

"""
Data readers for images
"""
import io
from typing import Optional
from typing import Union
import warnings

import nucleus7 as nc7
import numpy as np
import skimage.io
import tensorflow as tf

from ncgenes7.data_fields.images import ImageDataFields
from ncgenes7.utils import image_io_utils
from ncgenes7.utils import image_utils


class ImageDataReader(nc7.data.DataReader):
    """
    Read images from file name to numpy array

    Parameters
    ----------
    image_number_of_channels
        number of image channels
    image_size
        size of image as  [height, width]; if specified, loaded images will be
        resized to it; if not specified, then images will have the
        original size
    interpolation_order
        interpolation order to use if image_size was set; 0 - nearest neighbors,
        1 - bilinear, 2 - Bi-quadratic, 3 - Bi-cubic
    result_dtype
        dtype of the resulted image; if it is different from float32, first
        float32 image will be loaded with values in [0, 1] and then it will be
        multiplied with a constant so that values are in [0, max_of_dtype]

    Attributes
    ----------
    generated_keys
        * images : images, [width, height, num_channels], result_dtype
        * images_fnames : file names of images, str
        * image_sizes : image sizes with shape [width, height], np.int32
    """
    file_list_keys = [
        ImageDataFields.images,
    ]
    generated_keys = [
        ImageDataFields.images,
        ImageDataFields.images_fnames,
        ImageDataFields.image_sizes,
    ]

    def __init__(self, *,
                 image_number_of_channels: int = 3,
                 image_size: Optional[list] = None,
                 interpolation_order: int = 1,
                 result_dtype="float32",
                 **reader_kwargs):
        super().__init__(**reader_kwargs)
        if interpolation_order and not image_size:
            warnings.warn(
                "{}: interpolation_order is only used to resize to new "
                "image_size, which was not provided!".format(self.name))
        if image_size and len(image_size) != 2:
            raise ValueError(
                "{}: image_size should be list with [width, height]!".format(
                    self.name))
        self.image_number_of_channels = image_number_of_channels
        self.image_size = image_size
        self.interpolation_order = interpolation_order
        self.result_dtype = result_dtype

    def read(self, images: str):
        # pylint: disable=arguments-differ
        # parent method has more generic signature
        image = image_io_utils.read_image_with_number_of_channels(
            images, self.image_number_of_channels,
            image_size=self.image_size,
            interpolation_order=self.interpolation_order)
        image = self._cast_to_result_dtype(image)

        result = {
            ImageDataFields.images: image,
            ImageDataFields.image_sizes: np.asarray(image.shape[:2], np.int32),
            ImageDataFields.images_fnames: np.asarray(images)}
        return result

    def _cast_to_result_dtype(self, image):
        if image.dtype == np.dtype(self.result_dtype):
            return image

        image = image.astype(np.float32) / np.iinfo(np.uint8).max
        if self.result_dtype == "float32":
            return image
        if self.result_dtype == "float64":
            return image.astype(np.float64)

        max_dtype_value = np.iinfo(self.result_dtype).max
        image = (image * max_dtype_value).astype(self.result_dtype)
        return image


class ImageEncoder(nc7.data.DataProcessor):
    """
    Encodes array image using specified encoding

    Parameters
    ----------
    encoding
        encoding to use like "PNG", "JPEG" etc.
    additional_imsave_kwargs
        additional kwargs to pass to `skimage.io.imsave`

    Attributes
    ----------
    generated_keys
        * images_{ENCODING} : encoded image, bytes; key name will be
          `images_` + upper encoding
    """
    incoming_keys = [
        ImageDataFields.images,
    ]
    dynamic_generated_keys = True

    def __init__(self, *,
                 encoding="PNG",
                 additional_imsave_kwargs=None,
                 **reader_kwargs):
        super().__init__(**reader_kwargs)
        self.encoding = encoding.upper()
        self.additional_imsave_kwargs = additional_imsave_kwargs or {}

    def process(self, *, images):
        # pylint: disable=arguments-differ
        # parent method has more generic signature
        if images.ndim == 3 and images.shape[-1] == 1:
            images = np.squeeze(images, -1)
        with io.BytesIO() as bytes_io:
            skimage.io.imsave(bytes_io, images, format_str=self.encoding)
            image_encoded = bytes_io.getvalue()
        image_encoded_key = "_".join([ImageDataFields.images, self.encoding])
        return {image_encoded_key: image_encoded}


class ImageDataReaderTf(nc7.data.DataReader):
    """
    Read images from file name to tensorflow tensor

    Parameters
    ----------
    image_number_of_channels
        number of image channels
    image_size
        size of image as  [height, width]; if specified, loaded images will be
        resized to it; if not specified, then images will have the
        original size
    interpolation_order
        interpolation order to use if image_size was set; 0 - nearest neighbors,
        1 - bilinear, 2 - Bi-quadratic, 3 - Bi-cubic
    result_dtype
        dtype of the resulted image; if it is different from float32, first
        float32 image will be loaded with values in [0, 1] and then it will be
        multiplied with a constant so that values are in [0, max_of_dtype]
    align_corners
        see `tf.image.resize_images`
    preserve_aspect_ratio
        see `tf.image.resize_images`

    Attributes
    ----------
    generated_keys
        * images : images, [width, height, num_channels], np.uint8
        * images_fnames : file names of images, str
        * image_sizes : image sizes with shape [width, height], np.int32
    """
    file_list_keys = [
        ImageDataFields.images,
    ]
    generated_keys = [
        ImageDataFields.images,
        ImageDataFields.images_fnames,
        ImageDataFields.image_sizes,
    ]
    is_tensorflow = True

    def __init__(self, *,
                 image_number_of_channels: int = 3,
                 image_size: Optional[list] = None,
                 interpolation_order: int = 1,
                 result_dtype="float32",
                 preserve_aspect_ratio: bool = False,
                 align_corners: bool = True,
                 **reader_kwargs):
        super().__init__(**reader_kwargs)
        if interpolation_order and not image_size:
            warnings.warn(
                "{}: interpolation_order is only used to resize to new "
                "image_size, which was not provided!".format(self.name))
        if image_size and len(image_size) != 2:
            raise ValueError(
                "{}: image_size should be list with [width, height]!".format(
                    self.name))
        self.image_number_of_channels = image_number_of_channels
        self.image_size = image_size
        self.interpolation_order = interpolation_order
        self.result_dtype = result_dtype
        self.align_corners = align_corners
        self.preserve_aspect_ratio = preserve_aspect_ratio

    def read(self, *, images):
        # pylint: disable=arguments-differ
        # parent method has more generic signature
        images_fnames = tf.py_func(lambda x: x.decode(),
                                   [images], [tf.string])[0]
        images_fnames.set_shape([])
        image_str = tf.read_file(images)
        images = tf.image.decode_image(
            image_str, self.image_number_of_channels)
        images.set_shape([None, None, self.image_number_of_channels])
        images = _resize_image(
            images, self.image_size, self.interpolation_order,
            self.align_corners, self.preserve_aspect_ratio)
        images = _cast_to_result_dtype_tf(
            images, result_dtype=self.result_dtype)

        image_sizes = tf.cast(tf.shape(images)[:2], tf.int32)
        result = {ImageDataFields.images: images,
                  ImageDataFields.images_fnames: images_fnames,
                  ImageDataFields.image_sizes: image_sizes}
        return result


# pylint: disable=too-many-instance-attributes
# attributes cannot be combined or extracted further
class ImageDataReaderTfRecords(nc7.data.TfRecordsDataReader):
    """
    Image reader from tfrecords files. Images are stored as raw bytes and so
    have no compression inside.

    Images inside of tfrecords file must have 3 dimensions.

    Parameters
    ----------
    image_number_of_channels
        number of image channels
    image_encoding
        encoding to use like "PNG", "JPEG" etc.; if not provided, assumes that
        images are stored as a raw bytes array of uint8 and are under `images`
        key inside of tfrecords file; otherwise it looks for features with
        key "images_{ENCODING}" inside of tfrecords files and decodes it
        correspondingly; in case of encoding, `image_sizes` key from the
        tfrecords file will be not used
    image_size
        size of image as  [height, width]; if specified, loaded images will be
        resized to it; if not specified, then images will have the
        original size, which will be inferred from `image_sizes` field of
        tfrecords file; if none of the are specified, runtime error of
        tensorflow will be raised.
    interpolation_order
        interpolation order to use if image_size was set; 0 - nearest neighbors,
        1 - bilinear, 2 - cubic
    result_dtype
        result dtype
    align_corners
        see `tf.image.resize_images`
    preserve_aspect_ratio
        see `tf.image.resize_images`
    tfrecords_image_key
        key of image inside of tfrecords files; if image_encoding was specified,
        it should be a key without encoding suffix
    tfrecords_fname_key
        key of file name inside of tfrecords files
    tfrecords_sizes_key
        key of image sizes inside of tfrecords files if encoding was not
        provided

    Attributes
    ----------
    generated_keys
        * images : images, [width, height, num_channels], tf.uint8
        * images_fnames : file names of images, tf.string
        * image_sizes : image sizes with shape [width, height], tf.int32
    """
    generated_keys = [
        ImageDataFields.images,
        ImageDataFields.images_fnames,
        ImageDataFields.image_sizes,
    ]

    def __init__(self, *,
                 image_number_of_channels: int = 3,
                 image_size: Optional[list] = None,
                 interpolation_order: int = 1,
                 result_dtype="float32",
                 image_encoding="PNG",
                 preserve_aspect_ratio: bool = False,
                 align_corners: bool = True,
                 tfrecords_image_key: str = ImageDataFields.images,
                 tfrecords_fname_key: str = ImageDataFields.images_fnames,
                 tfrecords_sizes_key: str = ImageDataFields.image_sizes,
                 **reader_kwargs):
        super().__init__(**reader_kwargs)
        if image_encoding not in ["PNG", "JPEG", None]:
            raise ValueError(
                "{}: Image encoding must be in ['PNG', 'JPEG', None], "
                "provided {}".format(self.name, image_encoding))
        if interpolation_order not in (
                image_utils.INTERPOLATION_ORDER_TO_RESIZE_METHOD):
            msg = (
                "{}: not supported interpolation_order {}. "
                "Supported orders: {}".format(
                    self.name, interpolation_order,
                    image_utils.INTERPOLATION_ORDER_TO_RESIZE_METHOD))
            raise ValueError(msg)
        if not isinstance(getattr(tf, result_dtype, None), tf.DType):
            msg = "{}: dtype {} is not valid!".format(self.name, result_dtype)
            raise ValueError(msg)

        self.image_number_of_channels = image_number_of_channels
        self.image_size = image_size
        self.result_dtype = result_dtype
        self.interpolation_order = interpolation_order
        self.image_encoding = image_encoding
        self.align_corners = align_corners
        self.preserve_aspect_ratio = preserve_aspect_ratio
        self.tfrecords_image_key = tfrecords_image_key
        self.tfrecords_fname_key = tfrecords_fname_key
        self.tfrecords_sizes_key = tfrecords_sizes_key

    def get_tfrecords_output_types(self):
        output_types = {self.tfrecords_sizes_key: tf.int32}
        if not self.image_encoding:
            output_types[self.tfrecords_image_key] = tf.uint8
        return output_types

    def decode_field(self, field_name: str,
                     field_value: Union[tf.Tensor, tf.SparseTensor],
                     field_type: Optional[tf.DType] = None) -> tf.Tensor:
        if field_name == "_".join([self.tfrecords_image_key, "PNG"]):
            return tf.image.decode_png(
                field_value, channels=self.image_number_of_channels)
        if field_name == "_".join([self.tfrecords_image_key, "JPEG"]):
            return tf.image.decode_jpeg(
                field_value, channels=self.image_number_of_channels)

        return super().decode_field(field_name, field_value, field_type)

    def get_tfrecords_features(self):
        features = {
            self.tfrecords_fname_key: tf.FixedLenFeature((), tf.string,
                                                         "no_file_name")
        }
        if not self.image_encoding:
            default_image_size = self.image_size or [-1, -1]
            features.update({
                self.tfrecords_image_key: tf.FixedLenFeature((), tf.string),
                self.tfrecords_sizes_key: tf.FixedLenFeature(
                    (), tf.string,
                    np.array(default_image_size, np.int32).tostring())})
            return features

        image_encoded_key = "_".join([self.tfrecords_image_key,
                                      self.image_encoding])
        features[image_encoded_key] = tf.FixedLenFeature((), tf.string)
        return features

    def postprocess_tfrecords(self, **image_data) -> dict:
        # pylint: disable=arguments-differ
        # parent method has more generic signature
        images_fnames = image_data[self.tfrecords_fname_key]
        if not self.image_encoding:
            image_key = self.tfrecords_image_key
        else:
            image_key = "_".join([self.tfrecords_image_key,
                                  self.image_encoding])
        images = image_data[image_key]

        if not self.image_encoding:
            image_sizes = image_data[self.tfrecords_sizes_key]
            images = tf.reshape(images, [image_sizes[0], image_sizes[1],
                                         self.image_number_of_channels])

        images = _resize_image(
            images, self.image_size, self.interpolation_order,
            self.align_corners, self.preserve_aspect_ratio)
        images = _cast_to_result_dtype_tf(
            images, result_dtype=self.result_dtype)

        image_sizes = tf.cast(tf.shape(images)[:2], tf.int32)
        result = {ImageDataFields.images: images,
                  ImageDataFields.images_fnames: images_fnames,
                  ImageDataFields.image_sizes: image_sizes}
        return result


def _resize_image(images, image_size, interpolation_order,
                  align_corners=True, preserve_aspect_ratio=False):
    if not image_size:
        return images

    tf_resize_method = image_utils.INTERPOLATION_ORDER_TO_RESIZE_METHOD[
        interpolation_order]
    image_resized = tf.image.resize_images(
        images, image_size, method=tf_resize_method,
        align_corners=align_corners,
        preserve_aspect_ratio=preserve_aspect_ratio)
    return image_resized


def _cast_to_result_dtype_tf(image, result_dtype):
    result_dtype_tf = getattr(tf, result_dtype)
    if image.dtype == result_dtype_tf == tf.uint8:
        return image
    image = tf.cast(image, tf.float32) / tf.uint8.max
    if result_dtype == "float32":
        return image
    if result_dtype == "float64":
        return tf.cast(image, tf.float64)

    max_dtype_value = result_dtype_tf.max
    image = tf.cast(image * max_dtype_value, result_dtype_tf)
    return image
