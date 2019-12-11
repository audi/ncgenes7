# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

"""
Implementation of classification losses which are not usable directly from
tf.losses
"""
from typing import Dict

import nucleus7 as nc7
import tensorflow as tf

EPSILON = 1e-08

class ClassificationFocalLoss(nc7.model.ModelLoss):
    """
    Focal loss calculation for classification when there is imbalance in terms
    of number of samples per class.

    FocalLoss(pt)=-alpha * (1-pt)^(gamma) * log(pt)

    Here gamma is focusing parameter and alpha is balancing parameter.
    This implementation works for multi-classification.

    Parameters
    ----------
    gamma
        a focusing parameter which smoothly adjusts the rate at which easy
        examples are down weighted
        When gamma is zero, it is equivalent to normal cross entropy
        Default is 2.0 as it is mentioned in paper because it works best.
    alpha
        alpha value for balancing the focal loss

    Attributes
    ----------
    incoming_keys
        * classification_softmax : softmax output of network in shape of
          [bs, num_classes], tf.float32
        * groundtruth_classes : labels - one hot class label in shape
          [bs, 1], tf.int32

    generated_keys
        * loss : calculated focal loss

    References
    ----------
    https://arxiv.org/pdf/1708.02002.pdf
    """
    incoming_keys = ["classification_softmax",
                     "groundtruth_classes"]
    generated_keys = ["loss"]

    def __init__(self, *,
                 gamma: float = 2.0,
                 alpha: float = 1.0,
                 **loss_kwargs):
        self.gamma = gamma
        self.alpha = alpha
        super().__init__(**loss_kwargs)

    def process(self, *,
                classification_softmax: tf.Tensor,
                groundtruth_classes: tf.Tensor) -> Dict[str, tf.Tensor]:
        # pylint: disable=arguments-differ
        # parent save method has more generic signature
        num_classes = tf.shape(classification_softmax)[1]
        softmax_probabilities = tf.add(classification_softmax, EPSILON)
        groundtruth_classes = tf.squeeze(groundtruth_classes, axis=-1)
        one_hot_labels = tf.one_hot(indices=groundtruth_classes,
                                    depth=num_classes)
        cross_entropy = -one_hot_labels * tf.log(softmax_probabilities)
        focal_weight = self.alpha * (1 - softmax_probabilities) ** self.gamma

        focal_loss = focal_weight * cross_entropy
        focal_loss = tf.reduce_mean(tf.reduce_max(focal_loss, axis=1))
        result = {'loss': focal_loss}
        return result
