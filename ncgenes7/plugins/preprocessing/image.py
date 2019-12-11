# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

"""
Image preprocessing plugins
"""
from typing import Dict
from typing import Sequence
from typing import Union

import nucleus7 as nc7
import tensorflow as tf

from ncgenes7.data_fields.images import ImageDataFields


class ImageStandardization(nc7.model.ModelPlugin):
    """
    Plugin to standardize the images

    Parameters
    ----------
    parallel_iterations
        The number of iterations allowed to run in parallel for calls to
        tf.map_fn

    Attributes
    ----------
    incoming_keys
        * images : image or list of images, [bs, h, w, num_channels], tf.float32
    generated_keys
        * images : image or list of images, [bs, h, w, num_channels], tf.float32
    """
    incoming_keys = [
        ImageDataFields.images,
    ]
    generated_keys = [
        ImageDataFields.images,
    ]

    def __init__(self, *,
                 parallel_iterations: int = 16,
                 **plugin_kwargs):
        super(ImageStandardization, self).__init__(**plugin_kwargs)
        self.parallel_iterations = parallel_iterations

    def predict(self, *, images: tf.Tensor) -> Dict[str, tf.Tensor]:
        # pylint: disable=arguments-differ
        # base class has more generic signature
        images_standardized = tf.map_fn(
            tf.image.per_image_standardization,
            images, tf.float32,
            parallel_iterations=self.parallel_iterations)
        return {ImageDataFields.images: images_standardized}


class ImageLocalMeanSubtraction(nc7.model.ModelPlugin):
    """
    Remove the local mean from the image. The local mean is the mean of a
    patch. Useful if some parts of the image are darker than others

    Parameters
    ----------
    kernel_size
        The size of the patch

    Attributes
    ----------
    incoming_keys
        * images : image or list of images, [bs, h, w, num_channels], tf.float32
    generated_keys
        * images : image or list of images, [bs, h, w, num_channels], tf.float32
    """
    incoming_keys = [
        ImageDataFields.images,
    ]
    generated_keys = [
        ImageDataFields.images,
    ]

    def __init__(self,
                 kernel_size: Union[int, Sequence[int]] = 7,
                 **augmentation_kwargs):
        super().__init__(**augmentation_kwargs)
        if isinstance(kernel_size, int):
            kernel_size = [kernel_size, kernel_size]
        self.kernel_size = kernel_size

    def predict(self, *, images: tf.Tensor) -> Dict[str, tf.Tensor]:
        # pylint: disable=arguments-differ
        # base class has more generic signature
        images_subtracted = self._localized_mean(images)
        return {ImageDataFields.images: images_subtracted}

    def _localized_mean(self,
                        images: tf.Tensor) -> tf.Tensor:
        ksize = [1] + self.kernel_size + [1]
        images_local_mean = tf.nn.avg_pool(
            images, ksize, [1] * 4, padding="SAME")
        return images - images_local_mean
