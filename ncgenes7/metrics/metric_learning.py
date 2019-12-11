# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

"""
Evaluation metrics for distance metric learning
"""
import nucleus7 as nc7
import tensorflow as tf


class ProxyMetricLearningMetrics(nc7.model.ModelMetric):
    """
    Calculate accuracy for distance metric learning

    Parameters
    ----------

    Attributes
    ----------
    incoming_keys
        * proxy_estimates : predicted vectors in metric space, tf.float32,
          [bs, 1 or 4, proxy_dim]
        * origins : (optional) ground truth origins, tf.int, [bs]
        * proxies : (optional) the proxies

    generated_keys
        * proxy_accuracy : The calculated accuracy
    """
    incoming_keys = [
        "proxy_estimates",
        "_origins",
        "_proxies",
    ]
    generated_keys = [
        "proxy_accuracy",
    ]

    def process(self, proxy_estimates, origins=None, proxies=None):
        # pylint: disable=arguments-differ
        # base class has more generic signature
        result = dict()
        if origins is None or proxies is None:
            accuracy = self._build_testing(proxy_estimates)
        else:
            if self.is_training:
                accuracy = self._build_training(
                    proxy_estimates, proxies, origins)
            else:
                accuracy = self._build_testing(proxy_estimates)

        result['proxy_accuracy'] = accuracy
        return result

    @staticmethod
    def _build_training(proxy_estimates, proxies, origins):
        def _calc_loss(cur_rep):
            cur_diff = tf.reduce_sum(tf.square(cur_rep - proxies), axis=-1)
            out_lab = tf.to_int32(tf.argmin(cur_diff))
            return out_lab

        out_labs = tf.map_fn(
            _calc_loss, tf.squeeze(proxy_estimates), dtype=tf.int32)
        acc = tf.reduce_mean(tf.to_float(tf.equal(out_labs, origins)))
        return acc

    @staticmethod
    def _build_testing(proxy_estimates):
        out_embeddings = tf.unstack(proxy_estimates, axis=1, num=4)

        diff_01 = tf.reduce_sum(
            tf.square(out_embeddings[0] - out_embeddings[1]), axis=1)
        diff_23 = tf.reduce_sum(
            tf.square(out_embeddings[2] - out_embeddings[3]), axis=1)
        diff_02 = tf.reduce_sum(
            tf.square(out_embeddings[0] - out_embeddings[2]), axis=1)
        diff_03 = tf.reduce_sum(
            tf.square(out_embeddings[0] - out_embeddings[3]), axis=1)
        diff_12 = tf.reduce_sum(
            tf.square(out_embeddings[1] - out_embeddings[2]), axis=1)
        diff_13 = tf.reduce_sum(
            tf.square(out_embeddings[1] - out_embeddings[3]), axis=1)

        max_dist_true = tf.reduce_max(tf.stack([diff_01, diff_23], axis=1),
                                      axis=1)
        min_dist_false = tf.reduce_min(
            tf.stack([diff_02, diff_03, diff_12, diff_13], axis=1),
            axis=1
        )
        pred = tf.to_int32(tf.less(max_dist_true, min_dist_false))
        acc = tf.reduce_mean(tf.to_float(pred))

        return acc
