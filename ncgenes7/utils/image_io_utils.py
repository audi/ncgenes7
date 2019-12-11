# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

"""
Utils for image IO operations
"""

from collections import defaultdict
from typing import Dict
from typing import Optional
from typing import Union

from matplotlib import pyplot as plt
from matplotlib import ticker as mpl_ticker
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
import png
import skimage.io
import skimage.transform
import tensorflow as tf

from ncgenes7.data_fields.images import ImageDataFields


def read_image_with_number_of_channels(image_file_name: str,
                                       number_of_channels: int = 3,
                                       image_size: Optional[list] = None,
                                       interpolation_order: Optional[int] = 1,
                                       ) -> np.ndarray:
    """
    Read image from file name and if needed adjust it so it has
    number_of_channels channels:

    * if image has less than number_of_channels, first channel will be
      repeated for number_of_channels times

    * if image has more than number_of_channels, only first number_of_channels
      channels will be selected

    Parameters
    ----------
    image_file_name
        image file name
    number_of_channels
        number of channels inside of the output image
    image_size
        if specified, image will be resized to it
    interpolation_order
        order of interpolation to resize

    Returns
    -------
    image_from_file
        image from file with np.uint8 dtype
    """
    image = skimage.io.imread(image_file_name)
    if image.ndim == 3 and image.shape[-1] > number_of_channels:
        image = image[..., :number_of_channels]
    elif image.ndim == 2 or image.shape[-1] == 1:
        if image.ndim == 2:
            image = np.expand_dims(image, -1)
        if number_of_channels > 1:
            image = np.tile(image, [1, 1, number_of_channels])
    if image_size is not None:
        image_float = image.astype(np.float32) / np.iinfo(np.uint8).max
        image_float = skimage.transform.resize(
            image_float, image_size, order=interpolation_order, mode="reflect")
        image = (image_float * np.iinfo("uint8").max).astype(np.uint8)
    return image


def decode_images_with_fnames(
        inputs: (tf.Tensor, Dict[str, tf.Tensor]),
        image_size: list,
        number_of_channels: Union[dict, int],
        cast_dtypes: Union[Dict[str, tf.DType], tf.DType, None] = None,
        rescale_float32: bool = True) -> Dict[str, tf.Tensor]:
    """
    Decode the images from file names

    Parameters
    ----------
    inputs
        dict with tensor / list / ndarray of file names
    image_size
        image size in format height / width
    number_of_channels
        number of channels to decode
    cast_dtypes
        dtype to cast to
    rescale_float32
        rescale to [0, 1] for tf.float32 dtype

    Returns
    -------
    dict with keys as in inputs and decoded images as values together
    with keys in form 'fname_$input_key' with corresponding file names
    """

    if not isinstance(inputs, dict):
        inputs = {ImageDataFields.images: inputs}
    if not isinstance(cast_dtypes, dict) and cast_dtypes is not None:
        cast_dtypes = {ImageDataFields.images: cast_dtypes}
    if not isinstance(number_of_channels, dict):
        num_channels_ = number_of_channels
        number_of_channels = defaultdict(lambda: num_channels_)
    result = {k: tf.image.decode_image(tf.read_file(v), number_of_channels[k])
              for k, v in inputs.items() if v is not None}
    for k in result:
        result[k] = tf.reshape(result[k], image_size + [number_of_channels[k]])
        if cast_dtypes is not None and k in cast_dtypes:
            dtype = cast_dtypes[k]
            rescale = (result[k].dtype == tf.uint8 and
                       cast_dtypes[k] == tf.float32 and
                       rescale_float32)
            result[k] = tf.cast(result[k], dtype)
            if rescale:
                result[k] /= 255.

    result.update({'{}_fnames'.format(k): v for k, v in inputs.items()})
    return result


def save_isolines(save_file_name: str,
                  depth_image: np.ndarray,
                  image: np.ndarray,
                  max_isodist: float = 30,
                  number_of_isolevels: int = 12,
                  border_offset: int = 10,
                  use_log_scale: bool = True,
                  dpi: int = 100):
    """
    Draw isolines according to depth and store image

    Parameters
    ----------
    save_file_name
        file name to save the image
    depth_image
        pixel wise depth, [m]
    image
        original image
    max_isodist
        max distance to draw the isolines on, [m]
    number_of_isolevels
        number of isolines
    use_log_scale
        if isolines should be drawn in log scale
    border_offset
        offset from border in pixels
    dpi
        dpi of saved figure
    """
    # pylint: disable=too-many-arguments
    # not possible to have less arguments without more complexity
    fig, subplot_axis = _get_figure_and_subplot(image, dpi)
    plt.imshow(image)

    if use_log_scale:
        levels = np.logspace(0, np.log10(max_isodist), number_of_isolevels)
    else:
        levels = np.arange(0, max_isodist * (1 + 1 / number_of_isolevels),
                           max_isodist / number_of_isolevels)
    contour_extent = (border_offset, depth_image.shape[1] - border_offset,
                      border_offset, depth_image.shape[0] - border_offset)

    depth_for_contour = (
        depth_image[border_offset:-border_offset, border_offset:-border_offset])
    contour_plot = plt.contour(
        depth_for_contour,
        extent=contour_extent,
        linewidths=1.5,
        levels=levels, cmap='gist_rainbow', extend='max')

    _add_colorbar(contour_plot, subplot_axis, number_of_isolevels,
                  use_log_scale)
    fig.savefig(save_file_name, dpi=dpi, transparent=False)


def _add_colorbar(contour_plot, subplot_axis, n_isoevel, use_log_scale):
    cbbox = inset_axes(subplot_axis, '10%', '88%', loc=7)
    for each_spine in cbbox.spines.values():
        each_spine.set_visible(False)
    cbbox.tick_params(axis='both', left='off', top='off', right='off',
                      bottom='off', labelleft='off', labeltop='off',
                      labelright='off', labelbottom='off')
    cbbox.set_facecolor([1, 1, 1, 0.9])
    cbaxes = inset_axes(cbbox, '20%', '95%', loc=6)
    colorbar = plt.colorbar(contour_plot, cax=cbaxes, format='%.1f m')
    colorbar.ax.tick_params(labelsize=16)
    colorbar.lines[0].set_linewidth(50)
    if use_log_scale:
        tick_locator = mpl_ticker.LogLocator(base=2, subs=(0.5,))
    else:
        tick_locator = mpl_ticker.MaxNLocator(nbins=n_isoevel + 1)
    colorbar.locator = tick_locator
    colorbar.update_ticks()


def _get_figure_and_subplot(image, dpi):
    plt.ioff()
    plt.switch_backend('agg')
    height, width = image.shape[:2]
    figsize = width / float(dpi), height / float(dpi)
    fig = plt.figure(figsize=figsize)
    subplot_axis = fig.add_axes([0, 0, 1, 1])
    subplot_axis.axis('off')
    return fig, subplot_axis


def save_16bit_png(save_file_name: str, image: np.ndarray):
    """
    Save depth image image as 16 bit png image
    image should be already cropped to [0, 1]

    Parameters
    ----------
    save_file_name
        file name to save
    image
        image as float
    """
    image_16bit = (65535 * image).astype(np.uint16)
    with open(save_file_name, 'wb') as png_file:
        writer = png.Writer(width=image_16bit.shape[1],
                            height=image_16bit.shape[0],
                            bitdepth=16,
                            greyscale=True)
        img_16bit_as_list = image_16bit.reshape(
            image_16bit.shape[0], -1).tolist()
        writer.write(png_file, img_16bit_as_list)
