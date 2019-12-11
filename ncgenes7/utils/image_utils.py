# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

"""
Utils for image operations
"""

import logging
from typing import Callable
from typing import List
from typing import Optional
from typing import Union

# pylint: disable=wrong-import-order
# is issue with pylint
import cv2
import math
import numpy as np
import skimage.transform
import tensorflow as tf

INTERPOLATION_ORDER_TO_RESIZE_METHOD = {
    0: tf.image.ResizeMethod.NEAREST_NEIGHBOR,
    1: tf.image.ResizeMethod.BILINEAR,
    3: tf.image.ResizeMethod.BICUBIC,
}


def rgb_hash_fun(rgb: np.ndarray, nd=False):  # pylint: disable=invalid-name
    """
    hash function for rgb values

    Parameters
    ----------
    rgb
        image encoding rgb value
    nd
        if True, than the hash is done with 3-dimensional array (images)
    """
    if nd:
        rgb = rgb.astype(np.int64)
        rgb_hashed = np.sum(rgb * np.resize(1000 ** np.arange(3), (1, 1, 3)),
                            -1)
        return rgb_hashed
    return sum([v * 1000 ** i for i, v in enumerate(rgb)])


def decode_class_ids_from_rgb(labels_rgb: np.ndarray,
                              rgb_class_ids_mapping_hashed: dict,
                              hash_fun: Callable = rgb_hash_fun
                              ) -> np.ndarray:
    """
    Decode class ids from rgb image

    Parameters
    ----------
    labels_rgb
        labels as image in rgb format
    rgb_class_ids_mapping_hashed
        dict mapping hashed rgb values to class ids
    hash_fun
        hash function to hash rgb values to one value

    Returns
    -------
    class_ids
        image with decoded class ids, np.uint8
    """
    logger = logging.getLogger(__name__)
    labels_rgb_hashed = hash_fun(labels_rgb, nd=True)

    unknown_hash_values = set.difference(
        set(labels_rgb_hashed.flatten()),
        set(rgb_class_ids_mapping_hashed.keys()))
    if unknown_hash_values:
        rgb_unknown_values = labels_rgb.reshape(
            [-1, 3])[np.in1d(labels_rgb_hashed, list(unknown_hash_values))]
        rgb_unknown_values = list(np.unique(rgb_unknown_values, axis=0))
        logger.warning('Not declared rgb values found: %s', rgb_unknown_values)

    class_ids = np.zeros_like(labels_rgb_hashed)
    # pylint: disable=no-member,unsupported-assignment-operation
    # some bug on some environments
    for each_hash, each_class in rgb_class_ids_mapping_hashed.items():
        class_ids[labels_rgb_hashed == each_hash] = each_class
    return class_ids.astype(np.uint8)


def extract_edges_from_classes(labels_rgb: np.ndarray,
                               class_ids: np.ndarray,
                               edge_classes: Optional[List[int]] = None,
                               edge_thickness: int = 3
                               ) -> np.ndarray:
    """
    Extract edges from class id image for specified class ids

    Parameters
    ----------
    labels_rgb
        labels as image in rgb format
    class_ids
        image with decoded class ids, np.uint8
    edge_classes
        semantic classes to use for edge extraction; if not specified, edges
        will be extracted for all classes
    edge_thickness
        thickness of generated edges
        both values [0, 0, 10] and [0, 10, 0] correspond to class id 2

    Returns
    -------
    edges
        edges image
    """
    edge_classes = edge_classes or list(range(class_ids.max() + 1))
    mask = np.in1d(class_ids, edge_classes).reshape(class_ids.shape)
    class_ids_masked = labels_rgb.copy()
    class_ids_masked[np.logical_not(mask)] = 0

    edges = cv2.Canny(class_ids_masked, 0, 0)
    kernel = np.ones((edge_thickness, edge_thickness), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1) // 255
    return edges


def labels2rgb(label_image: np.ndarray, rgb_class_mapping: dict) -> np.ndarray:
    """
    Encode rgb values from labels according to rgb_class_mapping.
    If mapping is not provided, returns same image

    Parameters
    ----------
    label_image
        image with class ids in last dimension
    rgb_class_mapping
        mapping of type {class: [R, G, B]}

    Returns
    -------
    rgb_image : array_like
        encoded rgb image
    """
    if not rgb_class_mapping:
        if len(label_image.shape) == 2:
            label_image = np.expand_dims(label_image, -1)
        label_image = np.tile(label_image, [1, 1, 3])
        return label_image

    def change_values(x, dict_, ind):
        labels = np.zeros_like(x)
        # pylint: disable=unsupported-assignment-operation
        # some bug on some environments
        for k, each_value in dict_.items():
            labels[x == k] = each_value[ind]
        return labels

    # pylint: disable=invalid-name
    r = change_values(label_image, rgb_class_mapping, 0)
    g = change_values(label_image, rgb_class_mapping, 1)
    b = change_values(label_image, rgb_class_mapping, 2)
    # pylint: enable=invalid-name
    if label_image.shape[-1] == 1:
        return np.concatenate([r, g, b], -1)
    return np.stack([r, g, b], -1)


def blend_images(image1: np.ndarray, image2: np.ndarray, alpha: float = 0.4
                 ) -> np.ndarray:
    """
    Overlap 2 images with specified alpha

    Parameters
    ----------
    image1
        first image
    image2
        second image
    alpha
        coefficient values of second image

    Returns
    -------
    blended_image
        image as weighted sum of image1*(1-alpha) + image2*(alpha)
    """
    assert 0 <= alpha <= 1, "alpha must be in [0, 1]"
    return np.clip(image1 * (1.0 - alpha) + image2 * alpha, 0, 1)


def maybe_resize_and_expand_image(image1: np.ndarray, image2: np.ndarray
                                  ) -> np.ndarray:
    """
    Resize and expand image1 to have shape of image2

    Parameters
    ----------
    image1
        image to image
    image2
        image with size to expand / resize to

    Returns
    -------
    image1_with_shape_of_image2
        image1 with shape of image2
    """
    if image1.ndim < image2.ndim:
        image1 = np.expand_dims(image1, -1)
    if image1.shape[-1] == 1 and image2.shape[-1] == 3:
        image1 = np.tile(image1, [1] * (image1.ndim - 1) + [3])
    image_resized = []
    for img in image1:
        image_resized.append(
            skimage.transform.resize(img, image2.shape[1:3],
                                     mode='reflect'))
    image1 = np.stack(image_resized, 0)
    return image1


def disparity2depth(disparity: np.ndarray,
                    baseline_dist: float = 0.3,
                    focal_length: float = 1.4e-3,
                    sensor_width: float = 7.4e-3,
                    clip_distance: float = 125) -> np.ndarray:
    """
    Calculate the depth array from disparity based on stereo camera setup

    Parameters
    ----------
    disparity
        disparity
    baseline_dist
        distance between cameras, [m]
    focal_length
        focal length of the camera, [m]
    sensor_width
        width of the sensor, [m]
    clip_distance
        all values above that distance will be capped, [m]

    Returns
    -------
    depth
        depth image, [m]
    """
    disparity_in_meters = disparity * sensor_width
    depth = baseline_dist * focal_length / disparity_in_meters
    return np.clip(depth, 0, clip_distance)


def interpolation_method_by_dtype(x: tf.Tensor, as_str=False
                                  ) -> Union[str, int]:
    """
    Get interpolation method by dtype of x

    Parameters
    ----------
    x
        array to interpolate
    as_str
        if result should be as string

    Returns
    -------
    interpolation_method
        0 or "BILINEAR" if x is float and 1 otherwise
    """
    interpolation = 0 if x.dtype in [tf.float32, tf.float64] else 1
    if as_str:
        interpolation = {1: 'NEAREST', 0: 'BILINEAR'}[interpolation]
    return interpolation


def concatenate_images(images, nrows=-1, ncols=1,
                       image_rescale=None, image_grid_border_width=3,
                       image_size=None):
    """
    Concatenates images over 1st dimension

    Images will be resized to size of first image.

    Parameters
    ----------
    images
        list of tensors with number of dimensions 3 or 4;
        if ndim == 4, then last dimension should be 3 or 1
    nrows
        number of rows for image concatenation
    ncols
        number of columns for image concatenation
    image_rescale
        image rescale factors; if provided, should be same as number of
        precessed images
    image_grid_border_width
        number of pixels as grid border; color of the grid is blue
    image_size
        single image size to use; if specified, all images will be resized to it

    Returns
    -------
    image_concat
        concatenated images according to image_concat_nrows and
        image_concat_ncols
    """
    fill_color = tf.reshape(tf.constant([0, 0, 1], dtype=tf.float32),
                            [1, 1, 1, 3])
    if nrows == -1:
        nrows = math.ceil(len(images) / ncols)
    if ncols == -1:
        ncols = math.ceil(len(images) / nrows)

    if image_size is None:
        image_size = tf.shape(images[0])[1:3]
    images_resized = _resize_images(images, image_size, image_rescale)

    batch_size = tf.shape(images[0])[0]

    grid_border_hor, grid_border_vert = _get_grid_borders(
        batch_size, fill_color, image_grid_border_width, image_size, ncols)

    tile_multiples_missed = tf.concat(
        [[batch_size], image_size, [1]], 0)
    image_missed = tf.tile(fill_color, tile_multiples_missed)

    image_concat = _concat_images_to_grid(
        images_resized, image_missed, grid_border_hor, grid_border_vert,
        ncols, nrows)
    return image_concat


def single_image_reshape_decoded_and_resize(
        image: tf.Tensor,
        decoded_image_size: Union[list, tf.Tensor],
        number_of_channels: int,
        resize_to: Optional[list] = None,
        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
        cast_back: bool = True,
        clip_values=None) -> tf.Tensor:
    """
    Reshape to decoded shape and resize to new spatial size if needed

    Parameters
    ----------
    image
        single image as vector with shape []
    decoded_image_size
        image size to reshape to in format [height, width]
    number_of_channels
        number of channels in decoded image
    resize_to
        image to resize to in format [height, width]
    method
        one of tf.image.ResizeMethod
    cast_back
        if the resized image should be casted back to initial dtype
    clip_values
        list of min and max values to clip before cast back

    Returns
    -------
    image_resized
        resized and reshaped image
    """
    image_reshaped = tf.reshape(
        image, [decoded_image_size[0], decoded_image_size[1],
                number_of_channels])
    if resize_to is None:
        return image_reshaped
    image_resized = tf.image.resize_images(
        image_reshaped, resize_to, method=method)
    if clip_values:
        if image_resized.dtype == tf.uint8:
            image_resized = tf.cast(image_resized, tf.int32)
        image_resized = tf.clip_by_value(image_resized, *clip_values)
    if cast_back:
        image_dtype = image.dtype
        image_resized = tf.cast(image_resized, image_dtype)
    return image_resized


def _concat_images_to_grid(images_resized, image_missed, grid_border_hor,
                           grid_border_vert, ncols, nrows):
    image_rows = []
    for col_num in range(nrows):
        start_ind = col_num * ncols
        last_ind = (col_num + 1) * ncols
        images_in_one_row = images_resized[start_ind:last_ind]
        if len(images_in_one_row) < ncols:
            for _ in range(ncols - len(images_in_one_row)):
                images_in_one_row.append(image_missed)
        images_in_one_row = _join_element_to_list(
            images_in_one_row, grid_border_vert)
        image_rows.append(tf.concat(images_in_one_row, 2))
    image_rows = _join_element_to_list(image_rows, grid_border_hor)
    image_concat = tf.concat(image_rows, 1)
    return image_concat


def _get_grid_borders(batch_size, fill_color, image_grid_border_width,
                      image_size, ncols):
    full_width = [image_size[-1] * ncols + (ncols - 1)
                  * image_grid_border_width]
    tile_multiples_border_vert = tf.concat(
        [[batch_size], image_size[:1], [image_grid_border_width, 1]], 0)
    tile_multiples_border_hor = tf.concat(
        [[batch_size], [image_grid_border_width], full_width, [1]], 0)
    grid_border_vert = tf.tile(fill_color, tile_multiples_border_vert)
    grid_border_hor = tf.tile(fill_color, tile_multiples_border_hor)
    return grid_border_hor, grid_border_vert


def _resize_images(images, image_size, image_rescale):
    images_resized = []
    for i, image in enumerate(images):
        image = tf.to_float(image)
        if len(image.shape) == 3:
            image = tf.expand_dims(image, -1)
        if image.get_shape().as_list()[-1] == 1:
            image = tf.tile(image, [1, 1, 1, 3])
        if image_rescale is None or not isinstance(image_rescale, list):
            image_rescale_ = image_rescale
        else:
            image_rescale_ = image_rescale[i]
        image = image / image_rescale_
        image = tf.image.resize_nearest_neighbor(image, image_size)
        images_resized.append(image)
    return images_resized


def _join_element_to_list(items, item_to_insert):
    result = [item_to_insert] * (len(items) * 2 - 1)
    result[0::2] = items
    return result
