# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

"""
Tests for distance metric learning
"""
from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from ncgenes7.metrics.metric_learning import ProxyMetricLearningMetrics


class TestProxyEmbeddingAccuracy(parameterized.TestCase, tf.test.TestCase):
    def test_training(self):
        np.random.seed(4564)
        tf.reset_default_graph()

        proxy_estimates = tf.placeholder(tf.float32, [None, 1, 2])
        origins = tf.placeholder(tf.int32, [None])
        proxies = tf.constant(
            [[0, 1],
             [1, 0],
             [-1, 0]], dtype=tf.float32
        )

        metric = ProxyMetricLearningMetrics(inbound_nodes=[])

        acc_out = metric._build_training(proxy_estimates, proxies, origins)

        data = []
        oris = []
        exp_res = []

        # Case 0
        d = np.asarray([[[0, 1]], [[1, 0]], [[-1, 0]]], dtype=np.float32)
        o = np.asarray([0, 1, 2], dtype=np.int32)
        data.append(d)
        oris.append(o)
        exp_res.append(1.0)

        # Case 1
        d = np.asarray([[[0, 1]], [[1, 0]], [[-1, 0]]], dtype=np.float32)
        o = np.asarray([1, 1, 2], dtype=np.int32)
        data.append(d)
        oris.append(o)
        exp_res.append(2. / 3.)

        # Case 2
        d = np.asarray([[[0, 1]], [[1, 0]], [[-1, 0]]], dtype=np.float32)
        o = np.asarray([1, 1, 1], dtype=np.int32)
        data.append(d)
        oris.append(o)
        exp_res.append(1. / 3.)

        # Case 3
        d = np.asarray([[[0, 1]], [[1, 0]], [[-1, 0]]], dtype=np.float32)
        o = np.asarray([1, 0, 1], dtype=np.int32)
        data.append(d)
        oris.append(o)
        exp_res.append(0.)

        with tf.Session() as sess:
            for ii in range(len(data)):
                res_out = sess.run(
                    [acc_out],
                    feed_dict={proxy_estimates: data[ii],
                               origins: oris[ii]})[0]
                self.assertAllClose(res_out, exp_res[ii])

    def test_testing(self):
        np.random.seed(4564)
        tf.reset_default_graph()

        proxie_dim = 2

        proxy_estimates = tf.placeholder(tf.float32, [None, 4, proxie_dim])
        metric = ProxyMetricLearningMetrics(inbound_nodes=[])
        acc_out = metric._build_testing(proxy_estimates)

        with tf.Session() as sess:
            for kk in range(3):
                possible_results = np.random.normal(size=[4, 2])
                for ii in range(len(possible_results)):
                    l2 = np.sqrt(np.sum(np.square(possible_results[ii, :])))
                    possible_results[ii, :] /= l2

                trues = []
                falses = []
                mixed = []

                count_true = 0
                count_all = 0

                for ii in range(len(possible_results)):
                    for jj in range(ii + 1, len(possible_results)):
                        p1 = possible_results[ii, :] + \
                             0.01 * np.random.uniform(size=(proxie_dim,))
                        p2 = possible_results[ii, :] + \
                             0.01 * np.random.uniform(size=(proxie_dim,))
                        n1 = possible_results[jj, :] + \
                             0.01 * np.random.uniform(size=(proxie_dim,))
                        n2 = possible_results[jj, :] + \
                             0.01 * np.random.uniform(size=(proxie_dim,))

                        true_pair = np.stack([p1, p2, n1, n2], axis=0)
                        false_pair = np.stack([p1, n1, p2, n2], axis=0)

                        trues.append(true_pair)
                        falses.append(false_pair)

                        if np.random.uniform() > 0.5:
                            mixed.append(true_pair)
                            count_true += 1
                        else:
                            mixed.append(false_pair)

                        count_all += 1

                trues = np.stack(trues)
                falses = np.stack(falses)
                mixed = np.stack(mixed)

                res_true = sess.run([acc_out], {proxy_estimates: trues})[0]
                res_falses = sess.run([acc_out], {proxy_estimates: falses})[0]
                res_mixed = sess.run([acc_out], {proxy_estimates: mixed})[0]

                self.assertAllClose(res_true, 1.0)
                self.assertAllClose(res_falses, 0.0)
                self.assertAllClose(res_mixed, count_true / count_all)

    @parameterized.parameters([{'eval': True}, {'eval': False}])
    def test_dynamics(self, eval):
        if eval:
            np.random.seed(4564)
            tf.reset_default_graph()

            proxy_estimates = tf.placeholder(tf.float32, [None, None, 2])
            origins = tf.placeholder(tf.int32, [None])

            metric = ProxyMetricLearningMetrics(inbound_nodes=[])

            data_train = np.random.uniform(size=[5, 1, 2])
            data_test = np.random.uniform(size=[5, 4, 2])
            origin_data = np.random.uniform(3, size=[5]).astype(np.int32)

            # Test case with no proxies available
            metric.mode = 'eval'
            acc = metric.process(proxy_estimates)
            metric.mode = 'train'
            acc_train = metric.process(proxy_estimates)

            self.assertTrue('proxy_accuracy' in acc)

            with self.test_session() as sess:
                try:
                    _ = sess.run(
                        [acc_train],
                        feed_dict={
                            proxy_estimates: data_test,
                            origins: origin_data})
                except tf.errors.InvalidArgumentError:
                    self.assertTrue(False, msg='Raised unnecessary error')

                try:
                    _ = sess.run(
                        [acc],
                        feed_dict={
                            proxy_estimates: data_test,
                            origins: origin_data})
                except tf.errors.InvalidArgumentError:
                    self.assertTrue(False, msg='Raised unnecessary error')

                with self.assertRaises(tf.errors.InvalidArgumentError):
                    _ = sess.run(
                        [acc],
                        feed_dict={
                            proxy_estimates: data_train,
                            origins: origin_data})

                with self.assertRaises(tf.errors.InvalidArgumentError):
                    _ = sess.run(
                        [acc_train],
                        feed_dict={
                            proxy_estimates: data_train,
                            origins: origin_data})
        else:
            # Build dynamic graph
            np.random.seed(4564)
            tf.reset_default_graph()
            data_train = np.random.uniform(size=[5, 1, 2])
            data_test = np.random.uniform(size=[5, 4, 2])
            origin_data = np.random.uniform(3, size=[5]).astype(np.int32)
            proxy_estimates = tf.placeholder(tf.float32, [None, None, 2])
            origins = tf.placeholder(tf.int32, [None])
            proxies = tf.constant(
                [[0, 1],
                 [1, 0],
                 [-1, 0]], dtype=tf.float32
            )
            metric = ProxyMetricLearningMetrics(inbound_nodes=[])
            metric.mode = 'eval'
            acc = metric.process(proxy_estimates, origins, proxies)
            metric.mode = 'train'
            acc_train = metric.process(proxy_estimates, origins, proxies)
            self.assertTrue('proxy_accuracy' in acc)

            with self.test_session() as sess:
                try:
                    _ = sess.run(
                        [acc_train],
                        feed_dict={
                            proxy_estimates: data_train,
                            origins: origin_data})
                except tf.errors.InvalidArgumentError:
                    self.assertTrue(False, msg='Test path instead train path')

                try:
                    _ = sess.run(
                        [acc],
                        feed_dict={
                            proxy_estimates: data_test,
                            origins: origin_data})
                except tf.errors.InvalidArgumentError:
                    self.assertTrue(False, msg='Train path instead test path')

                with self.assertRaises(tf.errors.InvalidArgumentError):
                    _ = sess.run(
                        [acc],
                        feed_dict={
                            proxy_estimates: data_train,
                            origins: origin_data})

                with self.assertRaises(tf.errors.InvalidArgumentError):
                    _ = sess.run(
                        [acc_train],
                        feed_dict={
                            proxy_estimates: data_test,
                            origins: origin_data})
