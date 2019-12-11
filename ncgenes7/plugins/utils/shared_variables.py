# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

"""
Plugins providing other plugins with shared variables
"""
import math

import nucleus7 as nc7
import tensorflow as tf

from ncgenes7.utils.convolution_ops import layer_normalize


class Proxies(nc7.model.ModelPlugin):
    """
    Plugin providing proxies for distance metric learning

    Parameters
    ----------
    num_proxies
        number of proxies to generate
    proxy_dim
        number of dimensions in proxie

    Attributes
    ----------
    generated_keys
        * proxies : the normalized proxies
    """
    incoming_keys = []
    generated_keys = ["proxies"]

    def __init__(self, *,
                 num_proxies=100,
                 proxy_dim=32,
                 **plugin_kwargs):
        super(Proxies, self).__init__(**plugin_kwargs)
        self.num_proxies = num_proxies
        self.proxy_dim = proxy_dim

    def predict(self):
        # pylint: disable=arguments-differ
        # base class has more generic signature
        proxies_raw = tf.get_variable(
            'Proxies', [self.num_proxies, self.proxy_dim])
        proxies = layer_normalize(
            proxies_raw, math.sqrt(1.0 / self.proxy_dim), 0)
        out_dict = {'proxies': proxies}
        return out_dict
