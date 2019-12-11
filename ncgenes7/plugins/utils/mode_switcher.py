# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

"""
Feature mode switcher plugin
"""
from typing import Dict

import nucleus7 as nc7
from nucleus7.utils import nest_utils
import tensorflow as tf


class FeatureModeSwitcherPlugin(nc7.model.ModelPlugin):
    """
    Switches features according to the mode, e.g. if the mode if train, it
    will pass the train_features further and infer_features will be passed on
    other modes
    """
    incoming_keys = [
        "train_features",
        "infer_features",
    ]
    dynamic_generated_keys = True

    def predict(self, *,
                train_features: Dict[str, tf.Tensor],
                infer_features: Dict[str, tf.Tensor]
                ) -> Dict[str, tf.Tensor]:
        # pylint: disable=arguments-differ
        # base class has more generic signature
        self._validate_features(train_features, infer_features)
        if self.is_training:
            return train_features
        return infer_features

    def _validate_features(self, train_features, infer_features):
        train_features_flat = nest_utils.flatten_nested_struct(train_features)
        infer_features_flat = nest_utils.flatten_nested_struct(infer_features)
        train_features_keys = set(train_features_flat)
        infer_features_keys = set(infer_features_flat)
        not_existing_train_keys = infer_features_keys.difference(
            train_features_keys)
        not_existing_infer_keys = train_features_keys.difference(
            infer_features_keys)
        if not_existing_train_keys or not_existing_infer_keys:
            msg = ("{}: train and infer features differ! "
                   "(not existing train features: {}, "
                   "not existing infer features: {})"
                   ).format(self.name, not_existing_train_keys,
                            not_existing_infer_keys)
            raise ValueError(msg)
