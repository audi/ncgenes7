# ==============================================================================
# Copyright @ 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

"""
DUC module plugin
"""

from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import nucleus7 as nc7
import tensorflow as tf

from ncgenes7.plugins.cnns.cnn_blocks import DUC


class DUCPlugin(nc7.model.ModelPlugin):
    """
    DUC module plugin

    Parameters
    ----------
    kernel_size
        kernel size of DUC convolution
    num_classes
        number of classes
    stride_upsample
        see paper
    dilation_rates
        see paper
    image_height
        image height to resize after duc; will be added as default_placeholder,
        so can be changed during inference
    image_width
        image width to resize after duc; will be added as default_placeholder,
        so can be changed during inference

    Attributes
    ----------
    incoming_keys
        * feature_maps : tensor of shape  [batch_size, w, h, num_channels],
          tf.float32

    generated_keys
        * feature_maps : feature maps after processing

    References
    ----------
    DUC module
        https://arxiv.org/abs/1702.08502
    """
    incoming_keys = [
        "feature_maps",
    ]
    generated_keys = [
        "feature_maps",
    ]

    def __init__(self,
                 kernel_size: Union[int, list],
                 num_classes: int,
                 stride_upsample: int,
                 image_height: Optional[int] = None,
                 image_width: Optional[int] = None,
                 dilation_rates: Union[Tuple[int], List[int]] = (2, 4, 8, 16),
                 **plugin_kwargs):
        super(DUCPlugin, self).__init__(**plugin_kwargs)
        self.kernel_size = kernel_size
        self.num_classes = num_classes
        self.stride_upsample = stride_upsample
        self.dilation_rates = dilation_rates
        self.image_height = image_height
        self.image_width = image_width
        self._duc_layer = None

    def create_keras_layers(self):
        super().create_keras_layers()
        self._duc_layer = self.add_keras_layer(
            DUC(num_classes=self.num_classes,
                kernel_size=self.kernel_size,
                stride_upsample=self.stride_upsample,
                dilation_rates=self.dilation_rates,
                kernel_initializer=self.initializer))

    def predict(self, feature_maps) -> Dict[str, tf.Tensor]:
        # pylint: disable=arguments-differ
        # parent save method has more generic signature
        output = self._duc_layer(feature_maps)
        if self.image_height is not None and self.image_width is not None:
            image_height = self.add_default_placeholder(
                self.image_height, "image_height")
            image_width = self.add_default_placeholder(
                self.image_width, "image_width")
            output = tf.image.resize_bilinear(
                output, [image_height, image_width])
        return {"feature_maps": output}
