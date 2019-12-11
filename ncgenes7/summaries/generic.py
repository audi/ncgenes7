# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

"""
Generic ModelSummaries
"""

import nucleus7 as nc7
import tensorflow as tf

from ncgenes7.utils.image_utils import concatenate_images


class BaseSummary(nc7.model.ModelSummary):
    """
    Base class for summary for storing scalars, images, audio and text to event
    files

    Name of each particular summary will be the name of instance of this class

    Parameters
    ----------
    image_concat_nrows
        number of rows for image concatenation
    image_concat_ncols
        number of columns for image concatenation
    image_rescale
        image values rescale factors (e.g. for raw image in uint8 format it
        will be 255.0); image values will be divided by this number;
        if provided, should be same as number of precessed images
    image_size
        if specified, all images will be resized to this dimension ([h, w]); if
        not specified and multiple images provided, image size of first image
        will be used
    image_grid_border_width
        number of pixels as grid border; color of the grid is blue

    Attributes
    ----------
    incoming_keys
        * scalar : (optional) scalar or dict or list, [bs]
        * images : (optional) image or dict or list, [bs, h, w, num_channels];
          if provided as list, they they will be concatenated and resized
          using nearest neighbor to size of first image;
          concatenation will happen row-wise
        * audio : (optional) audio signals or dict or list, [bs, signal_len];
        * text : (optional) text or dict or list, [bs, char_len]
        * histogram : (optional) histograms or dict or list
    """
    incoming_keys = [
        "_scalar",
        "_images",
        "_audio",
        "_text",
        "_histogram",
    ]
    dynamic_generated_keys = True

    def __init__(self, *,
                 image_concat_nrows=-1,
                 image_concat_ncols=1,
                 image_rescale=False,
                 image_size=None,
                 image_grid_border_width=3,
                 **summary_kwargs):
        super().__init__(**summary_kwargs)
        self.image_concat_nrows = image_concat_nrows
        self.image_concat_ncols = image_concat_ncols
        assert self.image_concat_nrows * self.image_concat_ncols < 0, (
            "Either nrows, or ncols should be defined, not both!!!")
        self.image_rescale = image_rescale
        self.image_size = image_size
        self.image_grid_border_width = image_grid_border_width

    def process(self, scalar=None, images=None, audio=None,
                text=None, histogram=None):
        # pylint: disable=arguments-differ
        # base class has more generic signature

        if isinstance(scalar, list):
            scalar = dict(enumerate(scalar))
        if isinstance(audio, list):
            audio = dict(enumerate(audio))
        if isinstance(text, list):
            text = dict(enumerate(text))
        if isinstance(histogram, list):
            histogram = dict(enumerate(histogram))

        summaries = {}
        data = {'scalar': scalar,
                'audio': audio,
                'text': text,
                "histogram": histogram}
        if images is not None and not isinstance(images, list):
            if self.image_size is not None:
                images = tf.image.resize_nearest_neighbor(
                    images, self.image_size)
            data['image'] = images
        for each_key, each_value in data.items():
            if isinstance(each_value, dict):
                for summary_name, summary_value in each_value.items():
                    summary_full_name = _get_summary_name(
                        each_key, self.name, summary_name)
                    summaries[summary_full_name] = summary_value
            else:
                if each_value is not None:
                    summary_full_name = _get_summary_name(each_key, self.name)
                    summaries[summary_full_name] = each_value
        if isinstance(images, list):
            with tf.variable_scope('concat_images'):
                image_concat = concatenate_images(
                    images, self.image_concat_nrows,
                    self.image_concat_ncols,
                    self.image_rescale,
                    self.image_grid_border_width,
                    image_size=self.image_size)
            summaries[_get_summary_name('image', self.name)] = image_concat
        return summaries


def _get_summary_name(summary_type, name, suffix=None):
    name = '_'.join([summary_type, name])
    if suffix is not None:
        name += '_' + str(suffix)
    return name
