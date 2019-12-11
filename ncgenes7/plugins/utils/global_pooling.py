# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

"""
File containing global pooling plugins
"""
from typing import List
from typing import Union

import nucleus7 as nc7
import tensorflow as tf


class GlobalAveragePooling(nc7.model.ModelPlugin):
    """
    Plugin doing global average pooling

    Parameters
    ----------
    pool_dimensions
        int or list of int with dimensions to pool over
    keepdims
        If True, the pooled axes will remain but have dimension 1

    Attributes
    ----------
    incoming_keys
        * features : original tensor

    generated_keys
        * features : the input tensor after global average pooling
    """
    incoming_keys = ["features"]
    generated_keys = ["features"]

    def __init__(self,
                 pool_dimensions: Union[int, List[int]] = None,
                 keepdims: bool = False,
                 **plugin_kwargs):
        super(GlobalAveragePooling, self).__init__(**plugin_kwargs)
        self.pool_dimensions = pool_dimensions
        self.keepdims = keepdims

    def predict(self, *, features):
        # pylint: disable=arguments-differ
        # base class has more generic signature
        features_pooled = tf.reduce_mean(
            features, axis=self.pool_dimensions, keepdims=self.keepdims)

        result = {'features': features_pooled}
        return result
