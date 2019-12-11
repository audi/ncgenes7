# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from ncgenes7.losses.image import SSIMWithResizedLabels
from ncgenes7.losses.image import SoftmaxLossWithResizedLabels
from ncgenes7.losses.image import StereoDepthLoss


class TestStereoDepthLoss(parameterized.TestCase, tf.test.TestCase):

    @parameterized.parameters(
        {"use_2nd_order_gradients": True, "scales_n": 1},
        {"use_2nd_order_gradients": False, "scales_n": 1},
        {"use_2nd_order_gradients": True, "scales_n": 3})
    def test_process(self, use_2nd_order_gradients, scales_n):
        np.random.seed(4564)
        tf.reset_default_graph()

        loss = StereoDepthLoss(
            inbound_nodes=[], use_2nd_order_gradients=use_2nd_order_gradients
        ).build()
        loss.mode = 'train'

        image_left = tf.constant(np.random.rand(2, 20, 20, 3), tf.float32)
        image_right = tf.constant(np.random.rand(2, 20, 20, 3), tf.float32)
        image_reconstr_left = tf.constant(np.random.rand(2, 20, 20, 3),
                                          tf.float32)
        image_reconstr_right = tf.constant(np.random.rand(2, 20, 20, 3),
                                           tf.float32)

        disparity_left = tf.constant(np.random.rand(2, 20, 20, 1), tf.float32)
        disparity_right = tf.constant(np.random.rand(2, 20, 20, 1), tf.float32)
        disparity_inv_left = tf.constant(np.random.rand(2, 20, 20, 1),
                                         tf.float32)
        disparity_inv_right = tf.constant(np.random.rand(2, 20, 20, 1),
                                          tf.float32)

        def _add_scales(x, scales_n):
            x_list = [x]
            w, h = 20, 20
            for i in range(scales_n - 1):
                x_last = x_list[-1]
                offset_h = 2
                offset_w = 2
                x_list.append(x_last[:, offset_h:h - offset_h,
                              offset_w:w - offset_w, :])
                w, h = w - offset_w * 2, h - offset_h * 2
            return x_list

        if scales_n > 1:
            image_left = _add_scales(image_left, scales_n)
            image_right = _add_scales(image_right, scales_n)
            image_reconstr_left = _add_scales(image_reconstr_left, scales_n)
            image_reconstr_right = _add_scales(image_reconstr_right, scales_n)
            disparity_left = _add_scales(disparity_left, scales_n)
            disparity_right = _add_scales(disparity_right, scales_n)
            disparity_inv_left = _add_scales(disparity_inv_left, scales_n)
            disparity_inv_right = _add_scales(disparity_inv_right, scales_n)

        res = loss.process(images_left=image_left,
                           images_right=image_right,
                           images_left_reconstructed=image_reconstr_left,
                           images_right_reconstructed=image_reconstr_right,
                           disparity_images_left=disparity_left,
                           disparity_images_right=disparity_right,
                           disparity_images_left_inverted=disparity_inv_left,
                           disparity_images_right_inverted=disparity_inv_right)

        loss_keys_must = ['loss_appearance_matching',
                          'loss_disp_consistency',
                          'loss_disp_smoothness']
        if use_2nd_order_gradients:
            loss_keys_must.append('loss_disp_smoothness_2nd_order')
        self.assertSetEqual(set(res.keys()), set(loss_keys_must))


class TestSoftmaxLossWithResizedLabels(tf.test.TestCase):
    def test_process(self):
        logits = tf.constant(np.random.rand(2, 20, 10, 5), tf.float32)
        labels = tf.constant(np.random.randint(0, 5, size=[2, 10, 5, 1]),
                             tf.int32)
        loss = SoftmaxLossWithResizedLabels().build()
        res = loss.process(logits=logits, labels=labels)
        self.assertSetEqual({"loss"},
                            set(res.keys()))


class TestSSIMWithResizedLabels(tf.test.TestCase):
    def test_process(self):
        predictions = tf.constant(np.random.rand(2, 40, 20, 3), tf.float32)
        labels = tf.constant(np.random.rand(2, 30, 15, 3), tf.float32)
        loss = SSIMWithResizedLabels().build()
        res = loss.process(predictions=predictions, labels=labels)
        self.assertSetEqual({"loss"},
                            set(res.keys()))
