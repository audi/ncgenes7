# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

"""
Convolution ops
"""
from typing import Union

import tensorflow as tf


def layer_normalize(
        x_raw: tf.Tensor,
        factor: Union[float, tf.Tensor],
        offset: Union[float, tf.Tensor],
        example_rank: int = 1,
        epsilon: float = 1e-5) -> tf.Tensor:
    """
    Layer normalization

    Parameters
    ----------
    x_raw
        data to normalize
    factor
        factor to use
    offset
        offset to use
    example_rank
        dimension on to normalize, usually 1
    epsilon
        small offset to prevent division by zero

    Returns
    -------
    x_norm
        normalized data

    References
    ----------
    "Layer Normalization", J. L. Ba, J. R. Kiros, G. E. Hinton, 2016
    arXiv:1607.06450
    """
    x_shape = x_raw.get_shape().as_list()
    x_shape_moment = x_shape[example_rank:]
    dimensions_to_calculate_over = (
        [example_rank + ii for ii in range(len(x_shape_moment))])

    mean, variance = tf.nn.moments(
        x_raw, dimensions_to_calculate_over, keep_dims=True)
    variance = tf.where(
        tf.is_nan(variance), tf.ones_like(variance), variance) + epsilon
    std = tf.sqrt(variance)
    variance_to_use = factor / std

    x = x_raw * variance_to_use + (offset - mean * variance_to_use)

    return x
