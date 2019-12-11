# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

"""
Containers with names of input / outputs to use with images
"""


# pylint: disable=too-few-public-methods
# serves more as a container and not as an interface
class ImageDataFields:
    """
    Names for the input tensors.

    Holds the standard data field names to use for identifying input tensors.
    This should be used by the decoder to identify keys for the returned
    tensor_dict containing input tensors.
    And it should be used by the model to identify the tensors it needs.

    Attributes
    ----------
    images
        images
    images_PNG
        images encoded with PNG
    images_JPEG
        images encoded with JPG
    images_left
        left images
    images_right
        right images
    image_pairs
        paired images
    images_fnames
        file name of images
    images_original_size
        original sizes of image before it was resized
    images_left_fnames
        file name of left images
    image_sizes
        image sizes
    images_right_fnames
        file name of right images
    image_pairs_fnames
        file name of paired images
    depth_images
        depth images
    disparity_images
        disparity images
    disparity_images_left
        left disparity image
    disparity_images_right
        right disparity image
    """
    images = 'images'
    images_PNG = "images_PNG"
    images_JPEG = "images_JPEG"
    images_left = "images_left"
    images_right = "images_right"
    images_fnames = 'images_fnames'
    images_original_size = 'images_original_size'
    images_left_fnames = "images_left_fnames"
    images_right_fnames = "images_right_fnames"
    image_pairs = "image_pairs"
    image_pairs_fnames = "image_pairs_fnames"
    image_sizes = "image_sizes"
    depth_images = "depth_images"
    disparity_images = "disparity_images"
    disparity_images_left = "disparity_images_left"
    disparity_images_right = "disparity_images_right"
