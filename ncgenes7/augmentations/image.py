# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

"""
Image augmentations
"""
from typing import Dict
from typing import Optional
from typing import Sequence
from typing import Union

import nucleus7 as nc7
import numpy as np
import tensorflow as tf

from ncgenes7.data_fields.images import ImageDataFields
from ncgenes7.utils import image_utils
from ncgenes7.utils.general_utils import broadcast_with_expand_to


class ImageRandomBrightness(nc7.data.RandomAugmentationTf):
    """
    Random augmentation of brightness

    Parameters
    ----------
    max_delta
        see `tf.image.random_brightness`

    Attributes
    ----------
    incoming_keys
        * images : image or list of images, [h, w, num_channels],
          tf.float32
    generated_keys
        * images : image or list of images, [h, w, num_channels],
          tf.float32
    """
    incoming_keys = [
        ImageDataFields.images,
    ]
    generated_keys = [
        ImageDataFields.images,
    ]
    random_variables_keys = [
        "brightness_delta",
    ]

    def __init__(self, *, max_delta: float, **augmentation_kwargs):
        super().__init__(**augmentation_kwargs)
        self.max_delta = max_delta

    def create_random_variables(self):
        delta = tf.random_uniform([], -self.max_delta, self.max_delta,
                                  seed=self.get_random_seed())
        return {"brightness_delta": delta}

    def augment(self, *, images: tf.Tensor) -> Dict[str, tf.Tensor]:
        # pylint: disable=arguments-differ
        # base class has more generic signature
        images_augmented = tf.image.adjust_brightness(
            images, self.random_variables["brightness_delta"])
        return {ImageDataFields.images: images_augmented}


class ImageRandomContrast(nc7.data.RandomAugmentationTf):
    """
    Random augmentation of contrast

    Parameters
    ----------
    lower, upper
        see `tf.image.random_contrast`

    Attributes
    ----------
    incoming_keys
        * images : image or list of images, [h, w, num_channels],
          tf.float32
    generated_keys
        * images : image or list of images, [h, w, num_channels],
          tf.float32
    """
    incoming_keys = [
        ImageDataFields.images,
    ]
    generated_keys = [
        ImageDataFields.images,
    ]
    random_variables_keys = [
        "contrast_factor",
    ]

    def __init__(self, *, lower: float, upper: float, **augmentation_kwargs):
        super().__init__(**augmentation_kwargs)
        self.lower = lower
        self.upper = upper

    def create_random_variables(self):
        contrast_factor = tf.random_uniform(
            [], self.lower, self.upper, seed=self.get_random_seed())
        return {"contrast_factor": contrast_factor}

    def augment(self, *, images: tf.Tensor) -> Dict[str, tf.Tensor]:
        # pylint: disable=arguments-differ
        # base class has more generic signature
        images_augmented = tf.image.adjust_contrast(
            images, self.random_variables["contrast_factor"])
        return {ImageDataFields.images: images_augmented}


class ImageRandomHue(nc7.data.RandomAugmentationTf):
    """
    Random augmentation of hue

    Parameters
    ----------
    max_delta
        see `tf.image.random_hue`

    Attributes
    ----------
    incoming_keys
        * images : image or list of images, [h, w, num_channels],
          tf.float32
    generated_keys
        * images : image or list of images, [h, w, num_channels],
          tf.float32
    """
    incoming_keys = [
        ImageDataFields.images,
    ]
    generated_keys = [
        ImageDataFields.images,
    ]
    random_variables_keys = [
        "hue_delta",
    ]

    def __init__(self, *, max_delta: float, **augmentation_kwargs):
        super().__init__(**augmentation_kwargs)
        self.max_delta = max_delta

    def create_random_variables(self):
        delta = tf.random_uniform([], -self.max_delta, self.max_delta,
                                  seed=self.get_random_seed())
        return {"hue_delta": delta}

    def augment(self, *, images: tf.Tensor) -> Dict[str, tf.Tensor]:
        # pylint: disable=arguments-differ
        # base class has more generic signature
        images_augmented = tf.image.adjust_hue(
            images, self.random_variables["hue_delta"])
        return {ImageDataFields.images: images_augmented}


class ImageRandomSaturation(nc7.data.RandomAugmentationTf):
    """
    Random augmentation of saturation

    Parameters
    ----------
    lower, upper
        see `tf.image.random_saturation`

    Attributes
    ----------
    incoming_keys
        * images : image or list of images, [h, w, num_channels],
          tf.float32
    generated_keys
        * images : image or list of images, [h, w, num_channels],
          tf.float32
    """
    incoming_keys = [
        ImageDataFields.images,
    ]
    generated_keys = [
        ImageDataFields.images,
    ]
    random_variables_keys = [
        "saturation_factor",
    ]

    def __init__(self, *, lower: float, upper: float, **augmentation_kwargs):
        super().__init__(**augmentation_kwargs)
        self.lower = lower
        self.upper = upper

    def create_random_variables(self):
        saturation_factor = tf.random_uniform(
            [], self.lower, self.upper, seed=self.get_random_seed())
        return {"saturation_factor": saturation_factor}

    def augment(self, *, images: tf.Tensor) -> Dict[str, tf.Tensor]:
        # pylint: disable=arguments-differ
        # base class has more generic signature
        images_augmented = tf.image.adjust_saturation(
            images, self.random_variables["saturation_factor"])
        return {ImageDataFields.images: images_augmented}


class ImageHorizontalFlip(nc7.data.RandomAugmentationTf):
    """
    Random horizontal image flipping

    Attributes
    ----------
    incoming_keys
        * images : image or list of images, [h, w, num_channels],
          tf.float32
    generated_keys
        * images : image or list of images, [h, w, num_channels],
          tf.float32
    """
    incoming_keys = [
        ImageDataFields.images,
    ]
    generated_keys = [
        ImageDataFields.images,
    ]

    def augment(self, *, images: tf.Tensor) -> Dict[str, tf.Tensor]:
        # pylint: disable=arguments-differ
        # base class has more generic signature
        images_augmented = tf.image.flip_left_right(images)
        return {ImageDataFields.images: images_augmented}


class ImageFlipUpDown(nc7.data.RandomAugmentationTf):
    """
    Random up down flipping

    Attributes
    ----------
    incoming_keys
        * images : image or list of images, [h, w, num_channels],
          tf.float32
    generated_keys
        * images : image or list of images, [h, w, num_channels],
          tf.float32
    """
    incoming_keys = [
        ImageDataFields.images,
    ]
    generated_keys = [
        ImageDataFields.images,
    ]

    def augment(self, *, images: tf.Tensor) -> Dict[str, tf.Tensor]:
        # pylint: disable=arguments-differ
        # base class has more generic signature
        images_augmented = tf.image.flip_up_down(images)
        return {ImageDataFields.images: images_augmented}


class _ImageRandomRotation(nc7.data.RandomAugmentationTf
                           ):  # pylint: disable=abstract-method
    random_variables_keys = [
        "rotation_angle",
    ]

    def __init__(self, *,
                 max_angle: float,
                 **augmentation_kwargs):
        super().__init__(**augmentation_kwargs)
        self.max_angle = max_angle

    def create_random_variables(self):
        max_angle_rad = self.max_angle * np.pi / 180
        rotation_angle = tf.random_uniform([], -max_angle_rad, max_angle_rad,
                                           seed=self.get_random_seed())
        return {"rotation_angle": rotation_angle}


class ImageRandomRotation(_ImageRandomRotation):
    """
    Image rotation

    Parameters
    ----------
    max_angle
        defines boundaries for random angle generation in grads

    Attributes
    ----------
    incoming_keys
        * images : image or list of images, [h, w, num_channels],
          tf.float32
    generated_keys
        * images : image or list of images, [h, w, num_channels],
          tf.float32
    """
    incoming_keys = [
        ImageDataFields.images,
    ]
    generated_keys = [
        ImageDataFields.images,
    ]

    def augment(self, *, images: tf.Tensor) -> Dict[str, tf.Tensor]:
        # pylint: disable=arguments-differ
        # base class has more generic signature
        interpolation = image_utils.interpolation_method_by_dtype(
            images, as_str=True)
        images_augmented = tf.contrib.image.rotate(
            images, self.random_variables["rotation_angle"], interpolation)
        return {ImageDataFields.images: images_augmented}


class _ImageRandomCrop(nc7.data.RandomAugmentationTf
                       ):  # pylint: disable=abstract-method
    random_variables_keys = [
        "crop_offset",
        "crop_scale",
    ]

    def __init__(self, *,
                 min_scale: float,
                 max_scale: Optional[float],
                 resize_to_original: bool = True,
                 **augmentation_kwargs):
        super().__init__(**augmentation_kwargs)
        if max_scale:
            assert max_scale > min_scale, (
                "max_scale must be greater then min_scale")
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.resize_to_original = resize_to_original

    def create_random_variables(self):
        if self.max_scale is None:
            crop_scale = tf.constant(self.min_scale)
        else:
            crop_scale = tf.random_uniform(
                [], minval=self.min_scale, maxval=self.max_scale,
                dtype=tf.float32, seed=self.get_random_seed())

        crop_offset = tf.stack(
            [tf.random_uniform(
                [], minval=0, maxval=1 - crop_scale,
                seed=self.get_random_seed()) for i in range(2)])
        return {"crop_scale": crop_scale,
                "crop_offset": crop_offset}


class ImageRandomCrop(_ImageRandomCrop):
    """
    Crop images to random scale from [min_scale, max_scale].

    Parameters
    ----------
    min_scale
        min scale to crop the image
    max_scale
        max scale to crop the image; if not provided, then constant
        scale = min_scale will be used
    resize_to_original
        if resulted images should be resized to original size after cropping

    Attributes
    ----------
    incoming_keys
        * images : image or list of images, [h, w, num_channels],
          tf.float32
    generated_keys
        * images : image or list of images, [h, w, num_channels],
          tf.float32

    Raises
    ------
    AssertionError if both or none of size and scale are provided
    """
    incoming_keys = [
        ImageDataFields.images,
    ]
    generated_keys = [
        ImageDataFields.images,
    ]

    def augment(self, *, images: tf.Tensor) -> Dict[str, tf.Tensor]:
        # pylint: disable=arguments-differ
        # base class has more generic signature
        images_augmented = self._crop_resize_image(images)
        return {ImageDataFields.images: images_augmented}

    def _crop_resize_image(self, images: tf.Tensor) -> tf.Tensor:
        interpolation = image_utils.interpolation_method_by_dtype(images)
        images_size = tf.shape(images)[:-1]
        images_size_float = tf.cast(images_size, tf.float32)
        new_size_absolute = tf.cast(tf.floor(
            images_size_float * self.random_variables["crop_scale"]), tf.int32)
        new_size_absolute = tf.concat(
            [new_size_absolute, [images.get_shape().as_list()[-1]]], 0)

        offset_absolute = tf.cast(tf.floor(
            images_size_float * self.random_variables["crop_offset"]), tf.int32)
        offset_absolute = tf.concat([offset_absolute, [0]], 0)

        image_cropped = tf.slice(images, offset_absolute, new_size_absolute)

        if self.resize_to_original:
            image_cropped = tf.image.resize_images(
                image_cropped, images_size, method=interpolation)
        return image_cropped


class _ImageRandomCutout(nc7.data.RandomAugmentationTf
                         ):  # pylint: disable=abstract-method
    random_variables_keys = [
        "cut_lengths",
        "cut_offset",
    ]

    def __init__(self,
                 min_cut_length: Union[int, Sequence[float]] = 0.01,
                 max_cut_length: Union[int, Sequence[float]] = 0.05,
                 is_normalized: bool = False,
                 **augmentation_kwargs):
        super().__init__(**augmentation_kwargs)
        self.is_normalized = is_normalized
        if isinstance(min_cut_length, float):
            min_cut_length = (min_cut_length, min_cut_length)
        else:
            assert len(min_cut_length) == 2, (
                "min_cut_length must be either a float or have length of 2")
        if isinstance(max_cut_length, float):
            max_cut_length = (max_cut_length, max_cut_length)
        else:
            assert len(max_cut_length) == 2, (
                "max_cut_length must be either a float or have length of 2")
        self.min_cut_length = min_cut_length
        self.max_cut_length = max_cut_length

    def create_random_variables(self):
        cut_lengths = [
            tf.random_uniform([], min_len_dim, max_len_dim,
                              dtype=tf.float32, seed=self.get_random_seed())
            for min_len_dim, max_len_dim in zip(
                self.min_cut_length, self.max_cut_length)]
        cut_lengths = tf.stack(cut_lengths, 0)

        cut_offset = [
            tf.random_uniform([], minval=0, maxval=1 - cut_lengths[i],
                              seed=self.get_random_seed())
            for i in range(2)]
        cut_offset = tf.stack(cut_offset, 0)
        return {"cut_lengths": cut_lengths,
                "cut_offset": cut_offset}


class ImageRandomCutout(_ImageRandomCutout):
    """
    Set a random rectangle in the image to zero

    Parameters
    ----------
    min_cut_length
        Minimum rectangle length in each direction relative to image size, e.g.
        in [0, 1]; if single int, then it will be the same for all dimensions
    max_cut_length
        Maximum rectangle_length in each direction relative to image size, e.g.
        in [0, 1]; if single int, then it will be the same for all dimensions
    is_normalized
        Whether or not the image is already normalized, e.g. its mean = 0

    Attributes
    ----------
    incoming_keys
        * images : image or list of images, [h, w, num_channels],
          tf.float32
    generated_keys
        * images : image or list of images, [h, w, num_channels],
          tf.float32

    References
    ----------
    DeVries, T., & Taylor, G. W. (2017). Improved regularization of
    convolutional neural networks with cutout. arXiv preprint arXiv:1708.04552.
    """
    incoming_keys = [
        ImageDataFields.images,
    ]
    generated_keys = [
        ImageDataFields.images,
    ]

    def augment(self, *, images: tf.Tensor) -> Dict[str, tf.Tensor]:
        # pylint: disable=arguments-differ
        # base class has more generic signature
        if self.is_normalized:
            replace_value = 0.0
        else:
            replace_value = tf.reduce_mean(images, [0, 1], keepdims=True)
        images_augmented = self.cutout(images, replace_value)
        return {ImageDataFields.images: images_augmented}

    def cutout(self,
               image: tf.Tensor,
               replace_value: Union[float, tf.Tensor] = 0.0) -> tf.Tensor:
        """
        Apply cutout to the given image

        Parameters
        ----------
        image
            input image
        replace_value
            the value to which the rectangle is set

        Returns
        -------
        img_with_cutoff
            the input images after applying cutout
        """
        image_size = tf.shape(image)[:-1]
        image_size_float = tf.cast(image_size, tf.float32)
        cut_lengths_absolute = tf.cast(
            tf.floor(self.random_variables["cut_lengths"]
                     * image_size_float), tf.int32)
        cut_offset_absolute = tf.cast(
            tf.floor(self.random_variables["cut_offset"]
                     * image_size_float), tf.int32)

        mask = self._create_cutout_mask(
            cut_offset_absolute, cut_lengths_absolute, image_size)
        mask = broadcast_with_expand_to(mask, image)
        mask_with_replaced_value = tf.cast(mask, image.dtype) * replace_value
        image_with_cutoff = tf.where(mask,
                                     mask_with_replaced_value,
                                     image)

        return image_with_cutoff

    @staticmethod
    def _create_cutout_mask(cut_offset_absolute, cut_lengths_absolute,
                            image_size):
        grid = tf.meshgrid(tf.range(image_size[1]), tf.range(image_size[0]))
        limits = cut_lengths_absolute + cut_offset_absolute
        offset_exp = cut_offset_absolute[::-1, tf.newaxis, tf.newaxis]
        limits_exp = limits[::-1, tf.newaxis, tf.newaxis]
        mask = tf.reduce_all(
            tf.logical_and(tf.greater_equal(grid, offset_exp),
                           tf.less_equal(grid, limits_exp)), 0)
        return mask
