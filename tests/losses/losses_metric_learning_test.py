# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

"""
Test of metric learning losses
"""
from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from ncgenes7.losses.metric_learning import ProxyEmbeddingLoss


class TestProxyEmbeddingLoss(parameterized.TestCase, tf.test.TestCase):
    def test_loss_no_proxy(self):
        np.random.seed(4564)
        tf.reset_default_graph()

        loss_calcer = ProxyEmbeddingLoss(inbound_nodes=[])

        shape_in = [5, 4, 10]
        shape_in_quarter = [5, 1, 10]

        data_no_proxy = tf.placeholder(tf.float32, shape_in)
        loss_no_proxy = loss_calcer._non_proxy_path(data_no_proxy)

        with tf.Session() as sess:
            loss_eval = sess.run(
                loss_no_proxy, feed_dict={data_no_proxy: np.ones(shape_in)})

            self.assertAllClose(loss_eval, np.zeros([5]), rtol=2e-5, atol=2e-5,
                                msg='No proxy loss is not 0')
            self.assertAllClose(
                np.std(loss_eval), 0.0, msg='Variance of loss too high')

            zeros = np.zeros(shape_in_quarter)
            ones = np.ones(shape_in_quarter)
            twos = 2 * ones
            threes = 3 * ones
            data_feed = np.concatenate([zeros, ones, zeros, ones], axis=1)

            loss_eval = sess.run(
                loss_no_proxy, feed_dict={data_no_proxy: data_feed})
            self.assertAllClose(loss_eval, 1e6 * np.ones([5]), rtol=2e-3,
                                atol=2e-3, msg='No proxy loss is not 1e6')
            self.assertAllClose(
                np.std(loss_eval), 0.0, msg='Variance of loss too high')

            data_feed = np.concatenate([zeros, ones, twos, threes], axis=1)
            loss_eval = sess.run(
                loss_no_proxy, feed_dict={data_no_proxy: data_feed})
            self.assertAllClose(loss_eval, np.ones([5]), rtol=2e-5,
                                atol=2e-5, msg='No proxy loss is not 1')
            self.assertAllClose(
                np.std(loss_eval), 0.0, msg='Variance of loss too high')

    def test_loss_proxy(self):
        np.random.seed(4564)
        tf.reset_default_graph()

        loss_calcer = ProxyEmbeddingLoss(inbound_nodes=[])

        bs = 5
        shape_in = [bs, 2]

        prototypes = np.asarray(
            [[1., 0.], [0., 1.], [-1., 0.], [0., -1.]], dtype=np.float32)
        proxies = tf.constant(prototypes, dtype=tf.float32)

        data_proxie = tf.placeholder(tf.float32, shape_in)
        labs_proxie = tf.placeholder(tf.int32, [bs])
        loss_proxie = loss_calcer._proxy_path(
            data_proxie, proxies, labs_proxie
        )

        labs = [ii * np.ones([bs], dtype=np.int32) for ii in range(4)]

        with tf.Session() as sess:
            for ii in range(4):
                loss_list = []
                data = np.ones(shape_in, dtype=np.float32) * prototypes[ii, :]
                for lab in labs:
                    loss_all = sess.run(
                        loss_proxie,
                        feed_dict={data_proxie: data, labs_proxie: lab})
                    loss = np.mean(loss_all)
                    loss_list.append(loss)
                min_pos = np.argmin(loss_list)
                self.assertEqual(ii, min_pos, 'Wrong loss minimum')

    def test_dynamics(self):
        np.random.seed(4564)
        tf.reset_default_graph()

        # Set testing setting
        loss_calcer = ProxyEmbeddingLoss(inbound_nodes=[])
        loss_calcer.mode = 'eval'

        bs = 5
        shape_in = [bs, 4, 2]
        shape_in_single = [bs, 1, 2]

        in_data = tf.placeholder(tf.float32)
        prototypes = np.asarray(
            [[1., 0.], [0., 1.], [-1., 0.], [0., -1.]], dtype=np.float32)
        proxies = tf.constant(prototypes, dtype=tf.float32)
        labs_proxy = \
            tf.placeholder_with_default(tf.constant([0, 0], tf.int32), [None])

        loss = loss_calcer.process(
            in_data, labs_proxy, proxies)['loss_proxy_embedding']

        with tf.Session() as sess:
            loss_eval = sess.run(
                loss,
                feed_dict={in_data: np.ones(shape_in)})

            self.assertAllClose(loss_eval, np.zeros([]), rtol=2e-5, atol=2e-5,
                                msg='No proxy loss is not 0')
            zeros = np.zeros(shape_in_single)
            ones = np.ones(shape_in_single)
            twos = 2 * ones
            threes = 3 * ones
            data_feed = np.concatenate([zeros, ones, zeros, ones], axis=1)

            loss_eval = sess.run(
                loss, feed_dict={in_data: data_feed})
            self.assertAllClose(loss_eval, 2e5 * np.ones([]), rtol=2e-3,
                                atol=2e-3, msg='Loss is not 1e6')

            data_feed = np.concatenate([zeros, ones, twos, threes], axis=1)
            loss_eval = sess.run(
                loss, feed_dict={in_data: data_feed})
            self.assertAllClose(loss_eval, np.ones([]), rtol=2e-5,
                                atol=2e-5, msg='Loss is not 1')

        # From here on training setting
        tf.reset_default_graph()
        in_data = tf.placeholder(tf.float32)
        proxies = tf.constant(prototypes, dtype=tf.float32)
        labs_proxy = \
            tf.placeholder_with_default(tf.constant([0, 0], tf.int32), [None])

        loss_calcer.mode = 'train'
        loss = loss_calcer.process(
            in_data, labs_proxy, proxies)['loss_proxy_embedding']
        with tf.Session() as sess:
            labs = [ii * np.ones([bs], dtype=np.int32) for ii in range(4)]
            for ii in range(4):
                loss_list = []
                data = (np.ones(shape_in_single, dtype=np.float32) *
                        prototypes[ii, :])
                for lab in labs:
                    loss_eval = sess.run(
                        loss,
                        feed_dict={in_data: data, labs_proxy: lab})
                    loss_list.append(loss_eval)
                min_pos = np.argmin(loss_list)
                self.assertEqual(ii, min_pos, 'Wrong loss minimum')
