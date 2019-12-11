# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

"""
General ops
"""
import numpy as np


def softmax_np(logits: np.ndarray, axis=-1, keepdims=True):
    """
    softmax on the input array over axis

    Parameters
    ----------
    logits
        array with logits
    axis
        axis for softmax

    Returns
    -------
    softmax
        softmax
    """
    exp = np.exp(logits - np.amax(logits, axis=axis, keepdims=True))
    return exp / np.sum(exp, axis=axis, keepdims=keepdims)
