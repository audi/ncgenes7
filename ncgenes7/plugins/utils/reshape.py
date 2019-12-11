# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

"""
File containing utilities to reshape tensors
"""
from typing import Dict
from typing import Optional
from typing import Union

import nucleus7 as nc7
from nucleus7.utils import nest_utils
import tensorflow as tf

from ncgenes7.utils.reshape_ops import get_conversion_function
from ncgenes7.utils.reshape_ops import maybe_reshape
from ncgenes7.utils.reshape_ops import maybe_restore


class HideDimensionsInBatch(nc7.model.ModelPlugin):
    """
    Plugin hiding additional dimensions in the batch size.

    Parameters
    ----------
    target_rank
        Specifies the target rank. All other dimensions will be hidden in the
        batch size, e.g. Input [d1, d2, d3], target_rank 2 -> [d1*d2, d3]

    Attributes
    ----------
    incoming_keys
        * features : original tensor

    generated_keys
        * features : the input tensor with additional dimensions hidden
        * shape_in_batch : the shape which is now hidden in the batch
    """
    incoming_keys = [
        "features",
    ]
    generated_keys = [
        "features",
        "shape_in_batch",
    ]

    def __init__(self, *,
                 target_rank=4,
                 **plugin_kwargs):
        super().__init__(**plugin_kwargs)
        self.target_rank = target_rank

    def predict(self, features):
        # pylint: disable=arguments-differ
        # base class has more generic signature
        out_tensor, shape_in_batch = maybe_reshape(features, self.target_rank)
        out_dict = {'features': out_tensor,
                    'shape_in_batch': shape_in_batch}
        return out_dict


class ExtractDimensionsFromBatch(nc7.model.ModelPlugin):
    """
    Plugin extracting additional dimensions from the batch size

    There are possible ways to provide needed shape:
        * explicitly provide shape_in_batch (tensor with dimensions to extract
          from batch) to `shape_in_batch key`
        * provide a tensor, which has already the first dimensions as needed
          to `tensor_with_shape_in_batch` and provide `num_dims_in_batch` arg

    Parameters
    ----------
    num_dims_in_batch
        number of dimensions inside of batch, if tensor to infer the shape was
        provided

    Attributes
    ----------
    incoming_keys
        * features : original tensor
        * shape_in_batch : tensor specifying which dimensions have been hidden
        * tensor_with_shape_in_batch : tensor with the shape that must be
          unmasked up to num_dims_in_batch

    generated_keys
        * features : the input tensor with additional dimensions restored
    """
    incoming_keys = [
        "features",
        '_shape_in_batch',
        '_tensor_with_shape_in_batch',
    ]
    generated_keys = [
        "features",
    ]

    def __init__(self, num_dims_in_batch: Optional[int] = None,
                 **plugin_kwargs):
        super().__init__(**plugin_kwargs)
        self.num_dims_in_batch = num_dims_in_batch

    def predict(self, features, shape_in_batch: Optional[tf.Tensor] = None,
                tensor_with_shape_in_batch: Optional[tf.Tensor] = None):
        # pylint: disable=arguments-differ
        # base class has more generic signature
        if ((shape_in_batch is None and tensor_with_shape_in_batch is None)
                or (shape_in_batch is not None
                    and tensor_with_shape_in_batch is not None)):
            msg = ("{}: provide either shape_in_batch or "
                   "tensor_with_shape_in_batch").format(self.name)
            raise ValueError(msg)

        if (tensor_with_shape_in_batch is not None
                and self.num_dims_in_batch is None):
            msg = ("{}: Provide num_dims_in_batch to use tensor to infer the "
                   "batch hidden shape!")
            raise ValueError(msg)
        if tensor_with_shape_in_batch is not None:
            shape_in_batch = tf.shape(
                tensor_with_shape_in_batch)[:self.num_dims_in_batch]

        out_tensor = maybe_restore(features, shape_in_batch)
        out_dict = {'features': out_tensor}
        return out_dict


class DataFormatConverter(nc7.model.ModelPlugin):
    """
    Plugin to convert between data formats

    Parameters
    ----------
    input_format
        The input data format
    output_format
        The output data format

    Attributes
    ----------
    incoming_keys
        * features : original tensor

    generated_keys
        * features : the input tensor in the output_format
    """
    incoming_keys = ["features"]
    generated_keys = ["features"]

    def __init__(self,
                 *,
                 input_format='NTC',
                 output_format='NHWC',
                 **plugin_kwargs):
        super(DataFormatConverter, self).__init__(**plugin_kwargs)
        self.input_format = input_format
        self.output_format = output_format

    def predict(self, features):
        # pylint: disable=arguments-differ
        # base class has more generic signature
        fct = get_conversion_function(self.input_format, self.output_format)
        out_tensor = fct(features)
        out_dict = {'features': out_tensor}
        return out_dict


class ConcatPlugin(nc7.model.ModelPlugin):
    """
    Plugin to concat list inputs.

    It will take all inputs to it, which must be lists,
    e.g. {input1: [data1, data2, data3]} and concat it over the axis for this
    key

    Parameters
    ----------
    axis
        axis to concat the inputs; can be a dict with mapping of the input key
        (also nested with '//' separator) or a int, which will be then the same
        axis for all inputs; in case of axis mapping, 'default' key may exist
        and will be used if the key was not found inside of the axis mapping

    """
    dynamic_incoming_keys = True
    dynamic_generated_keys = True

    def __init__(self, axis: Union[Dict[str, int], int] = -1, **plugin_kwargs):
        super(ConcatPlugin, self).__init__(**plugin_kwargs)
        if isinstance(axis, int):
            axis = {"default": axis}
        self.axis = axis

    def predict(self, **inputs) -> Dict[str, tf.Tensor]:
        result_flat = {}
        default_axis = self.axis.get("default")
        inputs_flat = nest_utils.flatten_nested_struct(
            inputs, flatten_lists=False)
        for each_key, each_input_list in inputs_flat.items():
            if not isinstance(each_input_list, (list, tuple)):
                msg = ("{}: all inputs to concat must be lists or tuples! "
                       "(input for key {} is of type {})"
                       ).format(self.name, each_key, type(each_input_list))
                raise ValueError(msg)
            axis_for_key = self.axis.get(each_key, default_axis)
            if axis_for_key is None:
                msg = ("{}: axis for key {} was not provided and default key "
                       "does not exist!").format(self.name, each_key)
                raise ValueError(msg)
            if axis_for_key >= len(each_input_list[0].shape):
                msg = ("{}: axis {} for input key {} is not valid for"
                       "inputs with shape {}"
                       ).format(self.name, axis_for_key, each_key,
                                each_input_list[0].shape)
                raise ValueError(msg)
            inputs_concat = tf.concat(each_input_list, axis=axis_for_key)
            result_flat[each_key] = inputs_concat

        result = nest_utils.unflatten_dict_to_nested(result_flat)
        return result
