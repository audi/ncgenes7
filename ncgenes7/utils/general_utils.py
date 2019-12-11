# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

"""
general utils
"""

from functools import wraps
from typing import Callable

from nucleus7.utils import nest_utils
import tensorflow as tf


def check_if_dynamic_keys_flat(function: Callable) -> Callable:
    """
    Decorator to check if the provided dynamic keys are not nested

    Parameters
    ----------
    function
        function to wrap

    Returns
    -------
    wrapped
        wrapped function
    """

    @wraps(function)
    def wrapped(self, **kwargs):
        dynamic_keys = [each_key for each_key in kwargs
                        if each_key not in self.incoming_keys_all]
        dynamic_keys = dict(zip(*[dynamic_keys, dynamic_keys]))
        dynamic_keys_flat = nest_utils.flatten_nested_struct(
            dynamic_keys)
        dynamic_nested_keys = [each_key for each_key in dynamic_keys_flat
                               if "//" in each_key]
        if dynamic_nested_keys:
            msg = ("{}: following keys of dynamic inputs are nested: {}!"
                   ).format(self.name, dynamic_nested_keys)
            raise ValueError(msg)
        return function(self, **kwargs)

    return wrapped


def broadcast_with_expand_to(tensor1, tensor2):
    """
    Broadcast one tensor to another with possible expanding of last dimensions
    of tensor1

    Parameters
    ----------
    tensor1
        tensor to broadcast
    tensor2
        tensor to broadcast to

    Returns
    -------
    broadcasted_tensor1
        tensor1 broadcasted to tensor 2 shape
    """
    if len(tensor1.shape) == len(tensor2.shape):
        return tf.broadcast_to(tensor1, tf.shape(tensor2))
    num_new_dims = len(tensor2.shape) - len(tensor1.shape)
    mask_slices = ([slice(None)] * len(tensor1.shape)
                   + [tf.newaxis] * num_new_dims)
    mask_broadcasted = tf.broadcast_to(tensor1[mask_slices], tf.shape(tensor2))
    return mask_broadcasted
