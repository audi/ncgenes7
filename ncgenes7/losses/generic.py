# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

"""
Implementation of generic ModelLoss
"""
from typing import Optional

import nucleus7 as nc7
import tensorflow as tf


class GenericTFLoss(nc7.model.ModelLoss):
    """
    Generic tensorflow loss

    Parameters
    ----------
    tf_fun_name
        name tensorflow loss, e.g. `absolute_difference`
    arg_name_predictions
        which argument name do predictions have inside of tf function, e.g.
        'logits' for 'softmax_cross_entropy' or 'predictions' (default) for
        mean_squared_error
    arg_name_labels
        which argument name do labels have inside of tf function, e.g.
        'onehot_labels' for 'softmax_cross_entropy' or 'labels' (default)
        for mean_squared_error
    tf_fun_kwargs
        some additional parameters to tf_fun

    Attributes
    ----------
    incoming_keys
        * predictions : predictions, e.g. logits, np.float
        * labels : labels,  np.float
    generated_keys
        * loss : calculated loss
    """
    incoming_keys = [
        "predictions",
        "labels",
    ]
    generated_keys = [
        "loss",
    ]

    def __init__(self, *,
                 tf_fun_name: str,
                 arg_name_predictions: str = 'predictions',
                 arg_name_labels: str = 'labels',
                 tf_fun_kwargs: Optional[dict] = None,
                 **loss_kwargs):
        super().__init__(**loss_kwargs)
        if not hasattr(tf.losses, tf_fun_name):
            raise AttributeError(
                'There is no function tf.losses.{}'.format(tf_fun_name))
        self.tf_fun_name = tf_fun_name
        self.tf_fun = getattr(tf.losses, tf_fun_name)
        self.arg_name_predictions = arg_name_predictions
        self.arg_name_labels = arg_name_labels
        self.tf_fun_kwargs = tf_fun_kwargs or {}
        assert self.arg_name_predictions != self.arg_name_labels, (
            'Labels and predictions cannot have same names!!!')

    def process(self, predictions, labels):
        # pylint: disable=arguments-differ
        # parent save method has more generic signature
        loss_kwargs = {self.arg_name_predictions: predictions,
                       self.arg_name_labels: labels}
        loss_kwargs.update(self.tf_fun_kwargs)
        loss = self.tf_fun(**loss_kwargs)
        return {'loss': loss}
