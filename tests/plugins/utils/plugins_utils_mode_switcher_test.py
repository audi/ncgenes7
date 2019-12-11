# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

import tensorflow as tf

from ncgenes7.plugins.utils.mode_switcher import FeatureModeSwitcherPlugin


class TestFeatureModeSwitcherPlugin(tf.test.TestCase):

    def test_call(self):
        mode_switcher = FeatureModeSwitcherPlugin().build()
        train_features = {"key1": 1,
                          "key2": 2,
                          "key3": [3, 4],
                          "key4": {"sub1": 5, "sub2": 6}}
        infer_features = {"key1": 10,
                          "key2": 20,
                          "key3": [30, 40],
                          "key4": {"sub1": 50, "sub2": 60}}
        wrong_train_features = {"key1": 1,
                                "key2": 2,
                                "key3": [3, 4],
                                "key4": {"sub1": 5, "sub2": 6},
                                "not_existing_key": 1}
        wrong_infer_features = {"key1": 10}

        mode_switcher.mode = tf.estimator.ModeKeys.TRAIN
        with self.assertRaises(ValueError):
            mode_switcher(train_features=train_features,
                          infer_features=wrong_infer_features)
        with self.assertRaises(ValueError):
            mode_switcher(train_features=wrong_train_features,
                          infer_features=infer_features)

        mode_switcher.mode = tf.estimator.ModeKeys.TRAIN
        result_train = mode_switcher(train_features=train_features,
                                     infer_features=infer_features)
        mode_switcher.mode = tf.estimator.ModeKeys.EVAL
        result_eval = mode_switcher(train_features=train_features,
                                    infer_features=infer_features)
        mode_switcher.mode = tf.estimator.ModeKeys.PREDICT
        result_infer = mode_switcher(train_features=train_features,
                                     infer_features=infer_features)

        self.assertAllEqual(train_features,
                            result_train)
        self.assertAllEqual(infer_features,
                            result_eval)
        self.assertAllEqual(infer_features,
                            result_infer)
