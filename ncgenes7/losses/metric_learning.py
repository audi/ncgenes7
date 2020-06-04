# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

"""
Losses for metric learning
"""
import nucleus7 as nc7
import tensorflow as tf


class ProxyEmbeddingLoss(nc7.model.ModelLoss):
    """
    Loss metric learning with proxy embedding

    Attributes
    ----------
    incoming_keys
        * proxy_estimate : calculated embedding,
          {tf.float32, [tf.float32]*n}, either [bs, embedding_dim] or
          [bs, 4, embedding_dim]
        * origins : (optional), numeric origin of the data
          ('class label'), {tf.int32, [tf.int32]*n}, [bs]
        * proxies : (optional), used proxies, {tf.float32, [tf.float32]*n},
          [number_proxies, dimension_proxy]
    generated_keys
        * loss_proxy_embedding : proxy embedding loss

    References
    ----------
    "No Fuss Distance Metric Learning using Proxies", Yair Movshovitz-Attias,
    Alexander Toshev, Thomas K. Leung, Sergey Ioffe, Saurabh Singh,
    arXiv:1703.07464
    """
    incoming_keys = [
        "proxy_estimate",
        "_origins",
        "_proxies",
    ]
    generated_keys = [
        "loss_proxy_embedding",
    ]

    def process(self, proxy_estimate, origins=None, proxies=None):
        # pylint: disable=arguments-differ
        # base class has more generic signature
        if origins is None or proxies is None:
            loss_all = self._non_proxy_path(proxy_estimate)
        else:
            if self.is_training:
                loss_all = self._proxy_path(
                    proxy_estimate, proxies, origins)
            else:
                loss_all = self._non_proxy_path(proxy_estimate)
        loss = tf.reduce_mean(loss_all)
        return {'loss_proxy_embedding': loss}

    @staticmethod
    def _proxy_path(proxy_estimate: tf.Tensor, proxies: tf.Tensor,
                    origins: tf.Tensor) -> tf.Tensor:
        """
        Calculate the loss using the proxies (for training)

        Parameters
        ----------
        proxy_estimate
            Contains the proxy estimates, [bs, dimension_proxy]
        proxies
            contains the proxies used during training,
            [number_proxies, dimension_proxies]
        origins
            Contains the numeric label of the origin, [bs]

        Returns
        -------
        loss
            calculated batch loss
        """

        def _calc_loss(merged_rep):
            cur_rep, cur_idx = merged_rep
            cur_idx = tf.to_int32(cur_idx)
            cur_diff = tf.square(cur_rep - proxies)
            cur_sum_diffs = tf.reduce_sum(cur_diff, axis=-1)
            cur_upper_loss = cur_sum_diffs[cur_idx]
            cur_mask = 1.0 - tf.to_float(
                tf.one_hot(cur_idx, depth=tf.shape(proxies)[0]))
            cur_loss = -tf.log(tf.exp(-cur_upper_loss) / tf.reduce_sum(
                tf.exp(-cur_sum_diffs) * cur_mask))
            return cur_loss

        proxy_estimate = tf.squeeze(proxy_estimate)

        loss = tf.map_fn(_calc_loss, (proxy_estimate, tf.to_double(origins)),
                         dtype=tf.float32)

        return loss

    @staticmethod
    def _non_proxy_path(proxy_estimates: tf.Tensor) -> tf.Tensor:
        """
        Calculate the metric learning loss (for validation)

        Parameters
        ----------
        proxy_estimates
            Contains the proxy estimates, [bs, 4, dimension_proxies]

        Returns
        -------
        loss
            the calculated loss
        """
        unstacked = tf.unstack(proxy_estimates, axis=1, num=4)

        diff_01 = tf.reduce_sum(tf.square(unstacked[0] - unstacked[1]), axis=1)
        diff_23 = tf.reduce_sum(tf.square(unstacked[2] - unstacked[3]), axis=1)
        diff_02 = tf.reduce_sum(tf.square(unstacked[0] - unstacked[2]), axis=1)
        diff_03 = tf.reduce_sum(tf.square(unstacked[0] - unstacked[3]), axis=1)
        diff_12 = tf.reduce_sum(tf.square(unstacked[1] - unstacked[2]), axis=1)
        diff_13 = tf.reduce_sum(tf.square(unstacked[1] - unstacked[3]), axis=1)

        max_dist_true = tf.reduce_max(tf.stack([diff_01, diff_23], axis=1),
                                      axis=1)
        min_dist_false = tf.reduce_min(
            tf.stack([diff_02, diff_03, diff_12, diff_13], axis=1),
            axis=1
        )

        loss = max_dist_true / (min_dist_false + 1e-5)
        return loss
