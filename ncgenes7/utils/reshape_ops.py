# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

"""
Helper functions for reshape operations
"""
from functools import wraps
from typing import Callable
from typing import Tuple

import tensorflow as tf


def maybe_reshape(tensor: tf.Tensor, new_rank: int
                  ) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Reshape a tensor to have the wanted rank. Additional dimensions will be
    added to the batch size

    Parameters
    ----------
    tensor
        tensor with rank greater or equal wanted rank
    new_rank
        the rank the tensor should have after reshaping

    Returns
    -------
    out_tensor
        tensor of wanted rank
    new_shape
        part of the initial shape which is now hidden in the batch size
    """
    initial_shape = tf.shape(tensor)

    if new_rank == 1:
        return tf.reshape(tensor, [-1]), initial_shape

    final_shape = tf.convert_to_tensor(
        tf.concat([[-1], initial_shape[-new_rank + 1:]], axis=0),
        tf.int32
    )
    out_tensor = tf.reshape(tensor, final_shape)

    new_shape = initial_shape[:-new_rank + 1]

    return out_tensor, new_shape


def maybe_restore(tensor: tf.Tensor, shape_now_in_batch: tf.Tensor
                  ) -> tf.Tensor:
    """
    Reshape a tensor such that the batch size is extended to shape_now_in_batch
    E.g. given a tensor [bs_old, c] the new tensor will have the size
    `[*shape_now_in_batch, c]`

    Parameters
    ----------
    tensor
        tensor with rank greater or equal wanted rank
    shape_now_in_batch
        tensor containing the dimensionality to reshape the batch size to

    Returns
    -------
    out_tensor
        the reshaped tensor
    """
    initial_shape = tf.shape(tensor)
    final_shape = tf.convert_to_tensor(
        tf.concat([shape_now_in_batch, initial_shape[1:]], axis=0)
    )
    out_tensor = tf.reshape(tensor, final_shape)
    return out_tensor


def get_conversion_function(
        input_format: str,
        output_format: str) -> Callable[[tf.Tensor], tf.Tensor]:
    """
    Get a conversion function from the input format to the output format

    Parameters
    ----------
    input_format
        The input format of the data you want to convert
    output_format
        The wanted data format

    Returns
    -------
    conversion_fn
        A function to convert the data from input_format to output_format
    """
    mapping_data_types_to_function = {
        'NTC': {
            'NHWC': _ntc_2_nhwc
        },
        'NHWC': {
            'NCHW': _nhwc_2_nchw
        },
        'NCHW': {
            'NHWC': _nchw_2_nhwc
        }
    }
    return mapping_data_types_to_function[input_format][output_format]


def squeeze_batch_dimension(*input_names_to_squeeze) -> Callable:
    """
    Class decorator to squeeze particular inputs inside of decorated
    method

    Parameters
    ----------
    input_names_to_squeeze
        names of input to be squeezed

    Returns
    -------
    decorator
        class decorator
    """

    def wrapper(function: Callable):
        @wraps(function)
        def wrapped(self, *args, **kwargs):
            kwargs_squeezed = dict()
            for k, each_arg in kwargs.items():
                if (k not in input_names_to_squeeze) or (each_arg is None):
                    kwargs_squeezed[k] = each_arg
                else:
                    kwargs_squeezed[k] = tf.squeeze(each_arg, [0])
            return function(self, *args, **kwargs_squeezed)

        return wrapped

    return wrapper


def expand_batch_dimension(*output_names_to_expand) -> Callable:
    """
    Class decorator to expand batch dimension inside of decorated method

    Parameters
    ----------
    output_names_to_expand
        list of output names to expand the dimension

    Returns
    -------
    decorator
        class decorator
    """

    def wrapper(function: Callable):
        @wraps(function)
        def wrapped(self, *args, **kwargs):
            result = function(self, *args, **kwargs)
            result_expanded = dict()
            for k, each_value in result.items():
                if (k not in output_names_to_expand) or (each_value is None):
                    result_expanded[k] = each_value
                else:
                    result_expanded[k] = tf.expand_dims(each_value, [0])
            return result_expanded

        return wrapped

    return wrapper


def _ntc_2_nhwc(in_data):
    """
    Convert data in 'NTC' format to 'NHWC' format by padding

    Parameters
    ----------
    in_data : tensor_like
        Input data, 'NTC'

    Returns
    -------
    out_data : tensor
        Output data, 'NHWC'
    """
    out_data = tf.expand_dims(in_data, axis=-3)
    return out_data


def _nhwc_2_nchw(in_data: tf.Tensor) -> tf.Tensor:
    out_data = tf.transpose(in_data, perm=[0, 3, 1, 2])
    return out_data


def _nchw_2_nhwc(in_data: tf.Tensor) -> tf.Tensor:
    out_data = tf.transpose(in_data, perm=[0, 2, 3, 1])
    return out_data
