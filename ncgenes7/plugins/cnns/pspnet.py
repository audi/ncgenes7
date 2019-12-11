# ==============================================================================
# Copyright @ 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

"""
PSPNet plugin
"""
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import nucleus7 as nc7
import tensorflow as tf

from ncgenes7.plugins.cnns.cnn_blocks import PSPNetLayer


class PSPNetPlugin(nc7.model.ModelPlugin):
    """
    Plugin using pyramid pooling module

    Parameters
    ----------
    filters
        number of feature maps in each convolution after pooling and after
        concatenation of psp features
    output_height
        height of the upsampled output; if negative, then it is a multiplier to
        input feature maps size; otherwise it is a size itself
    output_width
        width of the upsampled output; if negative, then it is a multiplier to
        input feature maps size; otherwise it is a size itself
    pool_sizes
        pool sizes to use; if it is negative, then it is the same as in the
        paper - bin sizes; otherwise it is a pool sizes; in case of bin sizes,
        the input spatial dimensions must be defined and so cannot be None

    Attributes
    ----------
    incoming_keys
        * feature_maps : tensor of shape  [batch_size, w, h, num_channels],
          tf.float32

    generated_keys
        * feature_maps : feature maps after processing

    References
    ----------
    Pyramid Scene Parsing Network
        https:arxiv.org/abs/1612.01105
    """

    incoming_keys = [
        "feature_maps",
    ]
    generated_keys = [
        "feature_maps",
    ]

    def __init__(self,
                 filters,
                 pool_sizes: Union[Tuple[int], List[int]],
                 output_height: Union[float, int],
                 output_width: Union[float, int],
                 kernel_size: int = 3,
                 num_classes: Optional[int] = None,
                 **plugin_kwargs):
        super(PSPNetPlugin, self).__init__(**plugin_kwargs)
        self.filters = filters
        self.pool_sizes = pool_sizes
        self.output_height = output_height
        self.output_width = output_width
        self.kernel_size = kernel_size
        self.num_classes = num_classes
        self._psp_layer = None

    def create_keras_layers(self):
        super().create_keras_layers()
        self._psp_layer = self.add_keras_layer(
            PSPNetLayer(filters=self.filters,
                        kernel_size=self.kernel_size,
                        pool_sizes=self.pool_sizes,
                        output_height=self.output_height,
                        output_width=self.output_width,
                        num_classes=self.num_classes,
                        activation=self.activation,
                        kernel_initializer=self.initializer))

    def predict(self, feature_maps) -> Dict[str, tf.Tensor]:
        # pylint: disable=arguments-differ
        # parent save method has more generic signature
        output = self._psp_layer(feature_maps)
        return {"feature_maps": output}
